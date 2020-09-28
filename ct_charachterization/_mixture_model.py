import numpy as np

from ct_charachterization.utility.utils import central_gamma_pdf, broadcast_tile, block_matrix, \
    sum_over_each_neighborhood_on_blocked_matrix
from scipy.optimize import fsolve
from scipy.special import digamma
from functools import reduce


def _get_alphas_solution(right_hand_side, previous_alpha):
    alpha_optimizer = lambda alpha_var: right_hand_side - (np.log(alpha_var) - digamma(alpha_var))
    alpha_solution = fsolve(alpha_optimizer, previous_alpha)
    return alpha_solution


def _compute_next_gamma(y, theta, big_jay):
    # Eq. 18
    shape_of_gamma = tuple(list(y.shape) + [big_jay])
    new_gamma = np.empty(shape=shape_of_gamma, dtype=float)
    for j in range(big_jay):
        pi = theta[0, j]
        alpha = theta[1, j]
        beta = theta[2, j]
        times_to_br = tuple(np.array(np.array(y.shape) / np.array(pi.shape), dtype=int))
        pi = broadcast_tile(pi, times_to_br)
        alpha = broadcast_tile(alpha, times_to_br)
        beta = broadcast_tile(beta, times_to_br)
        new_gamma[..., j] = pi * central_gamma_pdf(y, alpha=alpha, beta=beta)
    summation = np.expand_dims(np.sum(new_gamma, axis=-1), axis=-1)
    new_gamma = new_gamma / summation
    return new_gamma


def _compute_next_theta(y, centered_mu, gamma, previous_alpha, y_shape_after_blocking):
    big_jay = len(centered_mu)
    shape_of_each_neighborhood = tuple(np.array(np.array(y.shape) / np.array(y_shape_after_blocking), dtype=int))
    size_of_each_neighborhood = reduce(lambda m, n: m * n, shape_of_each_neighborhood)

    new_pi = np.empty(shape=[big_jay] + y_shape_after_blocking)
    new_alpha = np.empty(shape=[big_jay] + y_shape_after_blocking)
    new_beta = np.empty(shape=[big_jay] + y_shape_after_blocking)

    blocked_y = block_matrix(mat=y, neighborhood_shape=shape_of_each_neighborhood)
    blocked_log_y = block_matrix(mat=np.log(y), neighborhood_shape=shape_of_each_neighborhood)

    for j in range(big_jay):
        blocked_gamma_j = block_matrix(mat=gamma[..., j], neighborhood_shape=shape_of_each_neighborhood)
        first_numerator_summation = sum_over_each_neighborhood_on_blocked_matrix(
            blocked_gamma_j * blocked_y / centered_mu[j])
        second_numerator_summation = sum_over_each_neighborhood_on_blocked_matrix(
            blocked_gamma_j * (blocked_log_y - np.log(centered_mu[j])))
        denominator_summation = sum_over_each_neighborhood_on_blocked_matrix(blocked_gamma_j)
        # Eq. 24
        right_hand_side = (first_numerator_summation - second_numerator_summation) / denominator_summation - 1
        # TODO: ravel and reshape work fine?
        alpha_initial_guess = previous_alpha[j, ...]
        vectorized_get_alphas_solution = np.vectorize(_get_alphas_solution)
        new_alpha[j, ...] = vectorized_get_alphas_solution(right_hand_side, alpha_initial_guess)
        # constraint: alpha[j] * beta[j] = mu[j]
        new_beta[j, ...] = centered_mu[j] / new_alpha[j, ...]
        # Eq. 22
        new_pi[j, ...] = denominator_summation / size_of_each_neighborhood
    new_theta = np.array([new_pi, new_alpha, new_beta])
    return new_theta


def run_first_algorithms(y: np.array, mu: np.array, neighborhood_size: int, delta=-1030, max_iter=5, tol=0.00000001,
                         non_central=False):
    y_shape_after_blocking = []
    for ax in y.shape:
        assert ax % neighborhood_size == 0, f'''Input array's shape ({ax}) is not dividable to neighborhood size ({neighborhood_size}).'''  # noqa
        y_shape_after_blocking.append(ax // neighborhood_size)
    big_jay = len(mu)

    # centering the data
    if non_central:
        y = y - delta
        mu = mu - delta

    # initial guess of parameters
    # we assume that theta[0] = pi, theta[1] = alpha, theta[2] = beta
    shape_of_theta = tuple([3, big_jay] + y_shape_after_blocking)
    theta = np.empty(shape=shape_of_theta, dtype=float)
    theta_before_expansion = np.array([[1 / big_jay] * big_jay, [2] * big_jay, [mu[j] / 2 for j in range(big_jay)]])
    for _ in range(len(y_shape_after_blocking)):
        theta_before_expansion = np.expand_dims(theta_before_expansion, axis=-1)
    theta[...] = theta_before_expansion
    # compute initial gamma
    gamma = _compute_next_gamma(y=y, theta=theta, big_jay=big_jay)
    err = np.Infinity
    n = 0
    while err > tol and n < max_iter:
        n += 1
        previous_alpha = theta[1, ...]
        new_theta = _compute_next_theta(y=y, centered_mu=mu, gamma=gamma, previous_alpha=previous_alpha,
                                        y_shape_after_blocking=y_shape_after_blocking)
        new_gamma = _compute_next_gamma(y=y, theta=theta, big_jay=big_jay)
        err = np.linalg.norm(new_theta - theta) / np.linalg.norm(theta)
        theta = new_theta
        gamma = new_gamma
        print(f'iteration: {n}, error: {err}')
    return theta, gamma
