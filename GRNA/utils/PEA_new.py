from scipy.optimize import minimize
import numpy as np

# Define the number of variables (change this to your desired n)
n = 6

# Initial guess for the variables
initial_guess = np.zeros(n)

# Define the equality constraints
def equality_constraints(vars):
    # Extract the values of C1 through Cn
    C_values = vars
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append(C_values[i] - C_values[j] - 0.4)
    return constraints

# Define the inequality constraints
def inequality_constraints(vars):
    # Extract the values of C1 through Cn
    C_values = vars
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append(C_values[i] - C_values[j])
    return constraints

# No bounds for the variables (removing the bounds definition)

# Combine constraints
constraints = [
    {'type': 'eq', 'fun': equality_constraints},
    {'type': 'ineq', 'fun': inequality_constraints}
]

# Use the SciPy minimize function to find a solution
result = minimize(lambda x: 0, initial_guess, constraints=constraints)

# Extract the solution
C_values = result.x

# Print the solution
for i in range(n):
    print(f"C{i+1} =", C_values[i])
