import numpy as np

from utility.utils import equation_18_on_vector_of_j_elements, ComputeThetaGammaBasedOn2DNeighborhood


# The second algorithm is pretty much the same as the first one
def run(y, mu, non_central=False, delta=-1025, max_iter=20, tol=0.01):
    size_of_j = len(mu)
    if non_central:
        y = y - delta
    err = np.Infinity
    shape_of_theta = (3, size_of_j)
    shape_of_gamma = tuple(list(y.shape) + [size_of_j])
    not_normalized_phi = np.random.random(size_of_j)
    sum_of_phis = sum(not_normalized_phi)
    theta = np.zeros(shape=shape_of_theta)
    random_alphas = list(range(10, size_of_j + 10))
    theta[:, :] = np.array([[i / sum_of_phis for i in not_normalized_phi], random_alphas,
                            [mu[j] / random_alphas[j] for j in range(size_of_j)]])
    gamma = np.zeros(shape=shape_of_gamma)
    for i, a in enumerate(y):
        for j, b in enumerate(a):
            to_be_appended_on_gamma = equation_18_on_vector_of_j_elements(b, theta).reshape(1, -1)
            gamma[i, j] = to_be_appended_on_gamma / np.sum(to_be_appended_on_gamma)
    n = 0
    while err > tol and n < max_iter:
        n += 1
        nbh = ComputeThetaGammaBasedOn2DNeighborhood(y, gamma, mu)
        nbh.compute_for_neighbors()
        new_gamma, new_theta = nbh.get_gamma_and_theta()
        print(n)
        err = np.linalg.norm(new_theta - theta) / np.linalg.norm(theta)
        theta = new_theta
        # TODO check why this sum is not equal to 1
        theta[0, :] = theta[0, :] / np.sum(theta[0, :])
        gamma = new_gamma
    return theta, gamma
