import time

import numpy as np

from utils import equation_18_on_vector_of_j_elements

'''<constants>'''
# min HU
delta = -1024
# Mu for 9 components
MU = {0: 340 - delta, 1: 240 - delta, 2: 100 - delta, 3: 0 - delta, 4: -160 - delta, 5: -370 - delta, 6: -540 - delta,
      7: -810 - delta, 8: -987 - delta}
J = len(MU)
MAX_ITER = 20
ERR = np.Infinity
# Tolerance
TOL = 0.1
'''</constants>'''

img = np.load(f'''./sample/img.npy''')
# We know that img.shape is (280, 364,364)
# so we set the neighborhood size to 28
neighborhood_size = 28
Y = img[140, :, :] - delta

shape_of_theta = tuple(list(Y.shape) + [3, J])
shape_of_gamma = tuple(list(Y.shape) + [J])
not_normalized_phi = np.random.random(J)
sum_of_phis = sum(not_normalized_phi)
theta = np.zeros(shape=shape_of_theta)
random_alphas = list(range(1, J + 1))
theta[:, :, :, :] = np.array(
    [[i / sum_of_phis for i in not_normalized_phi], random_alphas, [MU[j] / random_alphas[j] for j in range(J)]])
gamma = np.zeros(shape=shape_of_gamma)
t1 = time.time_ns()
for i, a in enumerate(Y):
    for j, b in enumerate(a):
        to_be_appended_on_gamma = equation_18_on_vector_of_j_elements(b, theta[i, j]).reshape(1, -1)
        gamma[i, j] = to_be_appended_on_gamma / np.sum(to_be_appended_on_gamma)
print(f'It took {(time.time_ns() - t1) / 1000000000} seconds.')

n = 0
while ERR > TOL and n < MAX_ITER:
    n += 1
    for j in range(J):
        pass
