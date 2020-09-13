import numpy as np

from utility.utils import central_gamma_pdf


def _compute_next_gamma(y, size_of_j, shape_of_gamma, theta):
    # Eq. 18
    new_gamma = np.zeros(shape=shape_of_gamma)
    for i, a in enumerate(y):
        for j, b in enumerate(a):
            sum_of_j_elements = 0
            for k in range(size_of_j):
                val = theta[0, k] * central_gamma_pdf(y[i, j], alpha=theta[1, k], beta=theta[2, k])
                new_gamma[i, j, k] = val
                sum_of_j_elements += val
            new_gamma[i, j] = new_gamma[i, j] / sum_of_j_elements
    return new_gamma


def _compute_next_theta(y, mu, gamma):
    size_of_j = len(mu)
    first_form_summation = np.sum(gamma * (np.expand_dims(y, axis=-1) / mu), axis=(0, 1)).reshape(size_of_j)
    second_form_summation = np.sum(gamma * np.log(np.expand_dims(y, axis=-1) / mu), axis=(0, 1)).reshape(size_of_j)
    denominator_summation = np.sum(gamma, axis=(0, 1)).reshape(size_of_j)
    new_alpha = (first_form_summation - second_form_summation) / denominator_summation - 1
    new_beta = np.array(mu) / new_alpha
    new_phi = denominator_summation / y.size
    new_theta = np.array([new_phi, new_alpha, new_beta])
    return new_theta


# The second algorithm is pretty much the same as the first one
def run(y, mu, non_central=False, delta=-1030, max_iter=10, tol=0.01):
    size_of_j = len(mu)

    # centering the data
    if non_central:
        y = y - delta

    # initial guess of parameters
    # we assume that theta[0] = phi, theta[1] = alpha, theta[2] = beta
    shape_of_theta = (3, size_of_j)
    theta = np.zeros(shape=shape_of_theta)
    theta[:, :] = np.array([[1 / size_of_j] * size_of_j, [2] * size_of_j, [mu[j] / 2 for j in range(size_of_j)]])

    # compute initial gamma
    shape_of_gamma = tuple(list(y.shape) + [size_of_j])
    gamma = _compute_next_gamma(y, size_of_j, shape_of_gamma, theta)
    err = np.Infinity
    n = 0
    while err > tol and n < max_iter:
        n += 1
        new_theta = _compute_next_theta(y, mu, gamma)
        new_gamma = _compute_next_gamma(y, size_of_j, shape_of_gamma, theta)
        err = np.linalg.norm(new_theta - theta) / np.linalg.norm(theta)
        theta = new_theta
        gamma = new_gamma
        print(n)
    return theta, gamma
