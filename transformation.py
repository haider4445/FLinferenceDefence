import numpy as np
import random


def generateTemplateMatrix(k):
	matrix = np.full((k,k), 2/k)
	for i in range(k):
		matrix[i][i] -= 1

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

print(generateDerivedTemplateMatrix(5))
