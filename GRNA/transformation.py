import numpy as np
import random


def generateTemplateMatrix(k):
	matrix = np.full((k,k), 2/k)
	for i in range(k):
		matrix[i][i] -= 1

	return matrix

def perturbedMatrix(matrix, pert):

	row_indices = [i for i in range(len(matrix))]
	for i in range(len(matrix)):
		randrow = random.choice(row_indices)
		matrix[randrow][i] -= 2**(pert)
		row_indices.remove(randrow)

		for j in row_indices:
			matrix[j][i] += 2**(pert)/(len(matrix))

	return matrix



def generateDerivedTemplateMatrix(k):

	zero_padding = np.zeros((k,k))
	basic_matrix = generateTemplateMatrix(k)

	derived_matrix1 = np.concatenate((basic_matrix, zero_padding.T), axis=1)
	derived_matrix2 = np.concatenate((zero_padding, basic_matrix.T), axis=1)
	derived_matrix = np.concatenate((derived_matrix1, derived_matrix2), axis=0)

	R = np.zeros(derived_matrix.shape)
	C = np.zeros(derived_matrix.shape)

	rand_list_R = [i for i in range(len(R))]
	rand_list_C = [i for i in range(len(C))]
	
	for i in range(len(R)):
		rand_num_R = random.choice(rand_list_R)
		rand_num_C = random.choice(rand_list_C)
		R[i][rand_num_R] = 1		
		C[i][rand_num_C] = 1
		rand_list_R.remove(rand_num_R)
		rand_list_C.remove(rand_num_C)


	# R = np.array([[1,0,0,0,0,0,0,0],
	# [0,0,0,0,0,1,0,0],
	# [0,0,1,0,0,0,0,0],
	# [0,0,0,0,0,0,1,0],
	# [0,0,0,0,1,0,0,0],
	# [0,1,0,0,0,0,0,0],
	# [0,0,0,1,0,0,0,0],
	# [0,0,0,0,0,0,0,1]])

	# C = np.array([[0,0,0,0,0,1,0,0],
	# [0,1,0,0,0,0,0,0],
	# [0,0,0,0,0,0,0,1],
	# [0,0,0,1,0,0,0,0],
	# [0,0,0,0,1,0,0,0],
	# [1,0,0,0,0,0,0,0],
	# [0,0,0,0,0,0,1,0],
	# [0,0,1,0,0,0,0,0]])
	
	randomized_derived_matrix = np.dot(R, derived_matrix)
	randomized_derived_matrix = np.dot(randomized_derived_matrix, C)

	return randomized_derived_matrix


r = np.array([[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]])

r = np.array([0.9853, 0.0147])
r = np.reshape(r, (-1, 1))

matrix = generateTemplateMatrix(1)
resultant2 = np.dot(r,matrix)
pert_matrix = perturbedMatrix(matrix, -4)
resultant = np.dot(r,pert_matrix)

print(np.linalg.norm(r[1]-r[0]))
print(np.linalg.norm(resultant[1]-resultant[0]))
print(np.linalg.norm(resultant2[1]-resultant2[0]))


# print(np.corrcoef(r[1], r[0]))
# print(np.corrcoef(resultant[1], resultant[0]))
# print(np.corrcoef(resultant2[1], resultant2[0]))

print(resultant)