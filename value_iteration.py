import numpy as np
from itertools import product
from Assignment2Tools import prob_vector_generator

def build_state_space(D, S_max, Z_max, tau):
    states = []
    forecast_values = [-1] + list(range(D + 1))

    if tau == 0:
        for s in range(S_max + 1):
            for z in range(Z_max + 1):
                for d in range(D + 1):
                    states.append((s, z, d))
    else:
        for s in range(S_max + 1):
            for z in range(Z_max + 1):
                for d in range(D + 1):
                    for future_info in product(forecast_values, repeat=tau):
                        states.append((s, z, d) + future_info)

    state_to_index = {}
    for i in range(len(states)):
        state_to_index[states[i]] = i

    return states, state_to_index


def get_valid_actions(state, S_max, Z_max):
    s = state[0]
    z = state[1]
    d = state[2]

    effective_demand = z + d
    actions = []

    for next_servers_on in range(S_max + 1):
        min_serve = max(0, effective_demand - Z_max)
        max_serve = min(next_servers_on, effective_demand)

        for served in range(min_serve, max_serve + 1):
            actions.append((next_servers_on, served))

    return actions


def get_immediate_reward(state, action, theta, alpha):
    s = state[0]
    z = state[1]

    next_servers_on, served = action

    switched_on = max(0, next_servers_on - s)

    switching_cost = theta[0] * switched_on
    energy_cost = theta[1] * next_servers_on + theta[2] * (served ** 2)
    penalty_cost = alpha * z

    total_cost = switching_cost + energy_cost + penalty_cost

    return -total_cost


def get_forecast_options(current_future, D, phi, tau, lmbda):
    if tau == 0:
        return [([], 1.0)]

    all_options = []

    for k in range(tau):
        if k < tau - 1:
            old_value = current_future[k + 1]

            if old_value != -1:
                all_options.append([(old_value, 1.0)])
            else:
                temp = [(-1, 1.0 - lmbda[k])]
                for d in range(D + 1):
                    temp.append((d, lmbda[k] * phi[d]))
                all_options.append(temp)
        else:
            temp = [(-1, 1.0 - lmbda[tau - 1])]
            for d in range(D + 1):
                temp.append((d, lmbda[tau - 1] * phi[d]))
            all_options.append(temp)

    combined = [([], 1.0)]

    for options_for_one_position in all_options:
        new_combined = []
        for partial_values, partial_prob in combined:
            for value, prob in options_for_one_position:
                new_combined.append((partial_values + [value], partial_prob * prob))
        combined = new_combined

    return combined


def get_next_state_distribution(state, action, D, Z_max, phi, tau, lmbda):
    z = state[1]
    d = state[2]
    next_servers_on, served = action

    next_backlog = z + d - served

    if tau == 0:
        next_state_probs = {}
        for next_d in range(D + 1):
            next_state = (next_servers_on, next_backlog, next_d)
            next_state_probs[next_state] = next_state_probs.get(next_state, 0) + phi[next_d]
        return next_state_probs

    current_future = state[3:]

    if current_future[0] != -1:
        next_d_options = [(current_future[0], 1.0)]
    else:
        next_d_options = []
        for next_d in range(D + 1):
            next_d_options.append((next_d, phi[next_d]))

    future_options = get_forecast_options(current_future, D, phi, tau, lmbda)

    next_state_probs = {}

    for next_d, prob_d in next_d_options:
        for future_values, prob_f in future_options:
            next_state = (next_servers_on, next_backlog, next_d) + tuple(future_values)
            prob = prob_d * prob_f
            next_state_probs[next_state] = next_state_probs.get(next_state, 0) + prob

    return next_state_probs


def value_iteration(D, S_max, theta, phi, Z_max, alpha, tau, lmbda, beta, threshold, Kmin):
    states, state_to_index = build_state_space(D, S_max, Z_max, tau)

    V = np.zeros(len(states))
    policy = {}

    iteration = 0

    while True:
        V_new = np.zeros(len(states))
        delta = 0.0

        for i in range(len(states)):
            state = states[i]
            actions = get_valid_actions(state, S_max, Z_max)

            best_value = -np.inf
            best_action = None

            for action in actions:
                reward = get_immediate_reward(state, action, theta, alpha)
                next_state_probs = get_next_state_distribution(state, action, D, Z_max, phi, tau, lmbda)

                expected_future_value = 0.0
                for next_state, prob in next_state_probs.items():
                    next_index = state_to_index[next_state]
                    expected_future_value += prob * V[next_index]

                q_value = reward + beta * expected_future_value

                if q_value > best_value:
                    best_value = q_value
                    best_action = action

            V_new[i] = best_value
            policy[state] = best_action

            diff = abs(V_new[i] - V[i])
            if diff > delta:
                delta = diff

        V = V_new.copy()
        iteration += 1

        if iteration >= Kmin and delta < threshold:
            break

    V_optimal = {}
    for i in range(len(states)):
        V_optimal[states[i]] = V[i]

    return V_optimal, policy



D = 5                            # Maximum computational demand

S_max = 15                       # Maximum number of servers that can be on at any time.

theta = [5, 1, 0.25]             # Parameters associated with server cost.
                                 # theta_1 = 5, theta_2 = 1, and theta_3 = 0.25.


# Generate arrival probability distribution, phi, of computational demand.
mu_d = 2                  # Mean of phi. You can vary this between 0.1*D to 0.9*D
                          # where D is the maximum computational demand.    
                          
stddev_ratio = 0.6        # You can vary this between 0.1 to 0.9. Higher value of
                          # stddev_ratio means a higher standard deviation of phi.                          

stddev_d = stddev_ratio*np.sqrt(D*(D-mu_d))     # Standard deviation of phi.

phi = prob_vector_generator(D, mu_d, stddev_d)  # Arrival probability distribution.


# Parameters associated with deferabble computational demand                           
Z_max = 10                       # Maximum deferrable time.

alpha = 0.5                      # Parameters associated with penalty cost.


# Parameters associated wiht forecasting
tau = 3                   # Forecasting window

lmbda = [1, 0.75, 0.5]    # Probabilities associated with a successful forecast.


beta = 0.95 # Discount factor

# Convergence parameters
threshold = 0.1     # The absolute value of the difference between current and 
                    # updated value function for any state should be lesser than
                    # this threshold for convergence.       
                    
Kmin = 10           # Minimum number of iterations of iterative policy evaluation
                    # step of policy iteration.


if __name__ == "__main__":
    V_optimal_vi, policy_optimal_vi = value_iteration(
        D, S_max, theta, phi, Z_max, alpha, tau, lmbda, beta, threshold, Kmin
    )