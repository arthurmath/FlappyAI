import numpy as np
import random as rd


# layer1 = np.ones((5, 10))
# layer2 = 2 * np.ones((5, 10))



# lineCut = rd.randint(0, layer1.shape[0] - 1)

# # if len(layer1.shape) == 1:  # 1D case
# #     return np.hstack((layer1[:lineCut], layer2[lineCut:]))

# columnCut = rd.randint(0, layer1.shape[1] - 1)

# res = np.vstack((
#     layer1[:lineCut],
#     np.hstack((layer1[lineCut, :columnCut], layer2[lineCut, columnCut:])),
#     layer2[lineCut + 1 :],
#     ))


# # print(res)

# # [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# #  [1. 1. 1. 2. 2. 2. 2. 2. 2. 2.]
# #  [2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]
# #  [2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]]





# layer = np.zeros((5, 10))


# MUTATION_RATE = 0.9
# STD_MUTATION = 0.5


# mask = np.random.rand(*layer.shape) < MUTATION_RATE # Tableau de True et False
# print(mask)
# mutations = np.clip(np.random.normal(0, STD_MUTATION, size=layer.shape), -1, 1) # -1 < mutations < 1 (stabilité numérique)
# layer = np.where(mask, layer + mutations, layer)  # condition, valeur_si_vrai, valeur_si_faux (layer += mask * mutations)

# print(layer)







mat1 = np.matrix([[-0.41622082,  0.83259267,  0.39608799],
 [ 0.44481878, -0.56549026, -0.01302485],
 [ 0.98477944,  0.10564779,  0.99063276],
 [-0.84693273, -0.78955088,  0.53307816]])

mat2 = np.matrix([[-0.70887243,  0.28280024, -0.26095774],
 [-0.20537569, -0.07098581,  0.61362277],
 [-0.64413963,  0.29949137,  0.07058191],
 [-0.34582447,  0.09990064, -0.29576575]])

lineCut = 1
columnCut = 2


print(mat1[lineCut, :columnCut])
print(mat2[lineCut, columnCut:])

print(type(mat1))

print(isinstance(mat1, np.matrix))


np.hstack((mat1[lineCut, :columnCut], mat2[lineCut, columnCut:]))