import numpy as np

from utility.utils import equation_18_on_vector_of_j_elements, ComputeThetaGammaBasedOnNeighborhood

# min HU
DELTA_2 = -1024
# Mu for 9 components
MU_2 = [340 - DELTA_2, 240 - DELTA_2, 100 - DELTA_2, 0 - DELTA_2, -160 - DELTA_2, -370 - DELTA_2, -540 - DELTA_2,
        -810 - DELTA_2, -987 - DELTA_2]
J_2 = len(MU_2)
MAX_ITER_2 = 20
# Tolerance
TOL_2 = 0.01


def run(y, neighborhood_size, non_central=False):
    if non_central:
        y = y - DELTA_2
    err = np.Infinity
    shape_of_theta = tuple(list(y.shape) + [3, J_2])
    shape_of_gamma = tuple(list(y.shape) + [J_2])
    not_normalized_phi = np.random.random(J_2)
    sum_of_phis = sum(not_normalized_phi)
    theta = np.zeros(shape=shape_of_theta)
    random_alphas = list(range(1, J_2 + 1))
    theta[:, :, :, :] = np.array(
        [[i / sum_of_phis for i in not_normalized_phi], random_alphas,
         [MU_2[j] / random_alphas[j] for j in range(J_2)]])
    gamma = np.zeros(shape=shape_of_gamma)
    for i, a in enumerate(y):
        for j, b in enumerate(a):
            to_be_appended_on_gamma = equation_18_on_vector_of_j_elements(b, theta[i, j]).reshape(1, -1)
            gamma[i, j] = to_be_appended_on_gamma / np.sum(to_be_appended_on_gamma)
    n = 0
    while err > TOL_2 and n < MAX_ITER_2:
        n += 1
        nbh = ComputeThetaGammaBasedOnNeighborhood(y, gamma, MU_2, neighborhood_size)
        nbh.compute_for_neighbors()
        new_theta = nbh.get_theta()
        new_gamma = nbh.get_gamma()
        print(np.linalg.norm(theta))
        err = np.linalg.norm(new_theta - theta) / np.linalg.norm(theta)
        theta = new_theta
        gamma = new_gamma
    return theta, gamma
