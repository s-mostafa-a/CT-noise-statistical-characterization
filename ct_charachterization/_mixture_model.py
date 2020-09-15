import numpy as np

from ct_charachterization.utility.utils import central_gamma_pdf


def _compute_next_gamma(y, big_jay, shape_of_gamma, theta):
    # Eq. 18
    axis_for_sum = len(y.shape)
    new_gamma = np.zeros(shape=shape_of_gamma)
    for j in range(big_jay):
        new_gamma[..., j] = theta[0, j] * central_gamma_pdf(y, alpha=theta[1, j], beta=theta[2, j])
    summation = np.expand_dims(np.sum(new_gamma, axis=axis_for_sum), axis=-1)
    new_gamma = new_gamma / summation
    return new_gamma


def _compute_next_theta(y, centered_mu, gamma):
    axis = tuple(range(len(y.shape)))
    big_jay = len(centered_mu)
    first_form_summation = np.sum(gamma * (np.expand_dims(y, axis=-1) / centered_mu), axis=axis).reshape(big_jay)
    second_form_summation = np.sum(gamma * np.log(np.expand_dims(y, axis=-1) / centered_mu), axis=axis).reshape(
        big_jay)
    denominator_summation = np.sum(gamma, axis=axis).reshape(big_jay)
    # Eq. 24
    new_alpha = (first_form_summation - second_form_summation) / denominator_summation - 1
    # constraint: alpha[j] * beta[j] = mu[j]
    new_beta = np.array(centered_mu) / new_alpha
    # Eq. 22
    new_pi = denominator_summation / y.size
    new_theta = np.array([new_pi, new_alpha, new_beta])
    return new_theta


def run_first_algorithms(y: np.array, mu: np.array, delta=-1030, max_iter=10, tol=0.01, non_central=False):
    big_jay = len(mu)

    # centering the data
    if non_central:
        y = y - delta
        mu = mu - delta

    # initial guess of parameters
    # we assume that theta[0] = pi, theta[1] = alpha, theta[2] = beta
    shape_of_theta = (3, big_jay)
    theta = np.zeros(shape=shape_of_theta)
    theta[:, :] = np.array(
        [[1 / big_jay] * big_jay, [2] * big_jay, [mu[j] / 2 for j in range(big_jay)]])

    # compute initial gamma
    shape_of_gamma = tuple(list(y.shape) + [big_jay])
    gamma = _compute_next_gamma(y, big_jay, shape_of_gamma, theta)
    err = np.Infinity
    n = 0
    while err > tol and n < max_iter:
        n += 1
        new_theta = _compute_next_theta(y, mu, gamma)
        new_gamma = _compute_next_gamma(y, big_jay, shape_of_gamma, theta)
        err = np.linalg.norm(new_theta - theta) / np.linalg.norm(theta)
        theta = new_theta
        gamma = new_gamma
        print(f'iteration: {n}, error: {err}')
    return theta, gamma
