import numpy as np
from Assignment2Tools import prob_vector_generator

def policy_evaluation_type1():
    # You must decide the necessary arguments of this function.
    
    # The function must return answer to one of the type 1 questions asked in
    # the document.
    pass


def policy_evaluation_type2():
    # You must decide the necessary arguments of this function.
    
    # The function must return answer to one of the type 2 questions asked in
    # the document.
    pass


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


# You must decide the necessary arguments of this function.
expected_value = policy_evaluation_type1()


# You must decide the necessary arguments of this function.
probability = policy_evaluation_type2()