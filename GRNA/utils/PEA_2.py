from scipy.optimize import minimize

# Initial guess for the variables
initial_guess = [0, 0, 0]

# Define the equality constraints
def equality_constraints(vars):
    C1, C2, C3 = vars
    return [
        C1 - C2 - 0.4,
        C1 - C3 - 0.6,
        C2 - C3 - 0.2
    ]

# Define the inequality constraints
def inequality_constraints(vars):
    C1, C2, C3 = vars
    return [
        C1 - C2,  # C1 > C2
        C1 - C3,  # C1 < C3
        C2 - C3   # C2 < C3
    ]

# No bounds for the variables (removing the bounds definition)

# Combine constraints
constraints = [
    {'type': 'eq', 'fun': equality_constraints},
    {'type': 'ineq', 'fun': inequality_constraints}
]

# Use the SciPy minimize function to find a solution
result = minimize(lambda x: 0, initial_guess, constraints=constraints)

# Extract the solution
C1, C2, C3 = result.x

# Print the solution
print("C1 =", C1)
print("C2 =", C2)
print("C3 =", C3)
