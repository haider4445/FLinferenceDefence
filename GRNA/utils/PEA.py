def PETER_Equation_Attack(array = [0.9, 0.5, 0.3], ordering = [2, 1, 3], n = 3):
	import sympy as sp

	variables = sp.symbols('C1:%d' % (n + 1))

	# Define the list of differences (example)
	differences = [
		sp.Eq(variables[i] - variables[j], array[i] - array[j])  # Example difference between variables i and j
		for i in range(n)
		for j in range(i + 1, n)
	]


	ordering_constraints = []
	for i in range(n):
		for j in range(i + 1, n):
			if ordering[i] > ordering[j]:
				ordering_constraints.append(variables[i] > variables[j])
			else:
				ordering_constraints.append(variables[i] < variables[j])
	

	constraints = differences + ordering_constraints
	print(constraints)
	# Solve the system of difference equations
	solutions = sp.solve(constraints, variables)
	print(solutions)

	# Check if solutions are within the range [0, 1]
	valid_solutions = []
	for solution in solutions:
		if all(0 <= solution[var] <= 1 for var in variables):
			valid_solutions.append(solution)



	# Choose one set of values from valid solutions (the first one)
	chosen_solution = valid_solutions[0]

	# Create a list from the chosen solution
	values_list = [chosen_solution[var] for var in variables]

	# Print the chosen values
	print(values_list)

PETER_Equation_Attack()