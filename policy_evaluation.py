import numpy as np
from itertools import product
from Assignment2Tools import prob_vector_generator
from policy_iteration import policy_iteration

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


def average_cost_policy_evaluation(states, state_to_index, policy, variable_values, D, Z_max, phi, tau, lmbda, threshold, Kmin):
    H = np.zeros(len(states))

    reference_state_index = 0
    iteration = 0

    while True:
        H_new = np.zeros(len(states))

        for i in range(len(states)):
            state = states[i]
            action = policy[state]

            immediate_value = variable_values[state]

            next_state_probs = get_next_state_distribution(state, action, D, Z_max, phi, tau, lmbda)

            expected_future = 0.0
            for next_state, prob in next_state_probs.items():
                next_index = state_to_index[next_state]
                expected_future += prob * H[next_index]

            H_new[i] = immediate_value + expected_future

        mu = H_new[reference_state_index]
        H_new = H_new - mu

        delta = np.max(np.abs(H_new - H))
        H = H_new.copy()
        iteration += 1

        if iteration >= Kmin and delta < threshold:
            break

    return mu


def policy_evaluation_type1(D, S_max, theta, phi, Z_max, alpha, tau, lmbda, beta, threshold, Kmin):
    # Type 1 question chosen:
    # What is the expected switching-on cost per time slot under the optimal policy?

    V_optimal_pi, policy_optimal_pi = policy_iteration(
        D, S_max, theta, phi, Z_max, alpha, tau, lmbda, beta, threshold, Kmin
    )

    states, state_to_index = build_state_space(D, S_max, Z_max, tau)

    variable_values = {}
    for state in states:
        action = policy_optimal_pi[state]
        current_servers = state[0]
        next_servers_on = action[0]
        switching_on = max(0, next_servers_on - current_servers)
        variable_values[state] = theta[0] * switching_on

    expected_value = average_cost_policy_evaluation(
        states, state_to_index, policy_optimal_pi, variable_values,
        D, Z_max, phi, tau, lmbda, threshold, Kmin
    )

    return expected_value


def policy_evaluation_type2(D, S_max, theta, phi, Z_max, alpha, tau, lmbda, beta, threshold, Kmin):
    # Type 2 question chosen:
    # What is the probability that the number of servers on after action is greater than 8?

    server_threshold = 8

    V_optimal_pi, policy_optimal_pi = policy_iteration(
        D, S_max, theta, phi, Z_max, alpha, tau, lmbda, beta, threshold, Kmin
    )

    states, state_to_index = build_state_space(D, S_max, Z_max, tau)

    variable_values = {}
    for state in states:
        action = policy_optimal_pi[state]
        next_servers_on = action[0]

        if next_servers_on > server_threshold:
            variable_values[state] = 1.0
        else:
            variable_values[state] = 0.0

    probability = average_cost_policy_evaluation(
        states, state_to_index, policy_optimal_pi, variable_values,
        D, Z_max, phi, tau, lmbda, threshold, Kmin
    )

    return probability


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
    expected_value = policy_evaluation_type1(
        D, S_max, theta, phi, Z_max, alpha, tau, lmbda, beta, threshold, Kmin
    )

    probability = policy_evaluation_type2(
        D, S_max, theta, phi, Z_max, alpha, tau, lmbda, beta, threshold, Kmin
    )