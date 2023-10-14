import numpy as np

numpy_vector = np.array([[0.1, 0.2, 0.3], [-0.2, -0.3, 0.4]])

# Compute the ranking with the lowest value ranked as 1
ranking = np.argsort(np.argsort(numpy_vector, axis=1), axis=1) + 1

# Divide the ranking by 10
result = ranking / 10

print("Original Vector:")
print(numpy_vector)
print("\nRanking:")
print(ranking)
print("\nRanking Divided by 10:")
print(result)
