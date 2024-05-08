def PETER_Equation_Attack(array=[0.5, 0.4, 0.1], ordering_signs = [3,2,1], n=3, sum_of_confidence_scores = 1):
    import sympy as sp
    import numpy as np

    array_permutations = range(0,n)

    # Create symbolic variables
    variables = sp.symbols('C1:%d' % (n + 1))

    # Generate equations based on the differences between the array elements
    differences = []
    for i in range(n):
      for j in range(i + 1, n):
        if ordering_signs[i]>ordering_signs[j]:
          differences.append(sp.Eq(variables[array_permutations[i]] - variables[array_permutations[j]], array[i] - array[j]))
        else:
          differences.append(sp.Eq(variables[array_permutations[i]] - variables[array_permutations[j]], array[j] - array[i]))

    # Add an additional equation for the sum of all variables equaling 1
    sum_equation = sp.Eq(sum(variables), sum_of_confidence_scores)
    differences.append(sum_equation)
    # Solve the equations
    solutions = sp.solve(differences, variables)
    return np.array(solutions)