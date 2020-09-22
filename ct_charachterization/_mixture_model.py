import numpy as np

from ct_charachterization.utility.utils import central_gamma_pdf, broadcast_tile, split_matrix, sum_of_each_patch
from scipy.optimize import fsolve
from scipy.special import digamma
from functools import reduce


def _compute_next_gamma(y, big_jay, shape_of_gamma, theta):
    # Eq. 18
    axis_for_sum = len(y.shape)
    new_gamma = np.zeros(shape=shape_of_gamma)
    for j in range(big_jay):
        pi = theta[0, j]
        alpha = theta[1, j]
        beta = theta[2, j]
        times_to_br = tuple(np.array(np.array(y.shape) / np.array(pi.shape), dtype=int))
        pi = broadcast_tile(pi, times_to_br)
        alpha = broadcast_tile(alpha, times_to_br)
        beta = broadcast_tile(beta, times_to_br)
        new_gamma[..., j] = pi * central_gamma_pdf(y, alpha=alpha, beta=beta)
    summation = np.expand_dims(np.sum(new_gamma, axis=axis_for_sum), axis=-1)
    new_gamma = new_gamma / summation
    return new_gamma


def _compute_next_theta(y, centered_mu, gamma, theta):
    big_jay = len(centered_mu)
    alpha_for_one_j_shape = theta[1, 0, :].shape
    shape_to_be_patched = tuple(np.array(np.array(y.shape) / np.array(alpha_for_one_j_shape), dtype=int))
    size_of_each_neighborhood = reduce(lambda m, n: m * n, shape_to_be_patched)
    patched_y = split_matrix(y, shape_to_be_patched)
    patched_log_y = split_matrix(np.log(y), shape_to_be_patched)
    new_pi = np.empty(theta[0, ...].shape)
    new_alpha = np.empty(theta[1, ...].shape)
    new_beta = np.empty(theta[2, ...].shape)

    for j in range(big_jay):
        patched_gamma_j = split_matrix(gamma[..., j], shape_to_be_patched)
        first_numerator_summation = sum_of_each_patch(patched_gamma_j * patched_y / centered_mu[j])
        second_numerator_summation = sum_of_each_patch(patched_gamma_j * (patched_log_y - np.log(centered_mu[j])))
        denominator_summation = sum_of_each_patch(patched_gamma_j)
        # Eq. 24
        right_hand_side = (first_numerator_summation - second_numerator_summation) / denominator_summation - 1
        to_be_found = lambda alp: right_hand_side.ravel() - (np.log(alp) - digamma(alp))
        alpha_j = theta[1, j, ...]
        alpha_initial_guess = alpha_j.ravel()
        alpha_solution = fsolve(to_be_found, alpha_initial_guess)
        new_alpha[j, ...] = alpha_solution.reshape(alpha_for_one_j_shape)
        # constraint: alpha[j] * beta[j] = mu[j]
        new_beta[j, ...] = centered_mu[j] / alpha_j
        # Eq. 22
        new_pi[j, ...] = denominator_summation / size_of_each_neighborhood
    new_theta = np.array([new_pi, new_alpha, new_beta])
    return new_theta


def run_first_algorithms(y: np.array, mu: np.array, neighborhood_size: int, delta=-1030, max_iter=5, tol=0.00000001,
                         non_central=False):
    shape_of_alpha_for_each_j_in_each_location = []
    for ax in y.shape:
        assert ax % neighborhood_size == 0, f'''Input array's shape ({ax}) is not dividable to neighborhood size ({neighborhood_size}).'''  # noqa
        shape_of_alpha_for_each_j_in_each_location.append(ax // neighborhood_size)
    big_jay = len(mu)

    # centering the data
    if non_central:
        y = y - delta
        mu = mu - delta

    # initial guess of parameters
    # we assume that theta[0] = pi, theta[1] = alpha, theta[2] = beta

    shape_of_theta = tuple([3, big_jay] + shape_of_alpha_for_each_j_in_each_location)
    theta = np.zeros(shape=shape_of_theta)
    theta_before_expansion = np.array([[1 / big_jay] * big_jay, [2] * big_jay, [mu[j] / 2 for j in range(big_jay)]])
    for _ in range(len(shape_of_alpha_for_each_j_in_each_location)):
        theta_before_expansion = np.expand_dims(theta_before_expansion, axis=-1)
    theta[...] = theta_before_expansion
    # compute initial gamma
    shape_of_gamma = tuple(list(y.shape) + [big_jay])
    gamma = _compute_next_gamma(y, big_jay, shape_of_gamma, theta)
    err = np.Infinity
    n = 0
    while err > tol and n < max_iter:
        n += 1
        new_theta = _compute_next_theta(y, mu, gamma, theta)
        new_gamma = _compute_next_gamma(y, big_jay, shape_of_gamma, theta)
        err = np.linalg.norm(new_theta - theta) / np.linalg.norm(theta)
        theta = new_theta
        gamma = new_gamma
        print(f'iteration: {n}, error: {err}')
    return theta, gamma
