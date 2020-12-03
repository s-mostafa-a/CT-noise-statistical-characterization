import numpy as onp

from .utility.utils import central_gamma_pdf, broadcast_tile, block_matrix, sum_over_each_neighborhood_on_blocked_matrix
from scipy.optimize import fsolve
from scipy.special import digamma
from functools import reduce


def _get_alphas_solution(right_hand_side, previous_alpha):
    alpha_optimizer = lambda alpha_var: right_hand_side - (onp.log(alpha_var) - digamma(alpha_var))
    alpha_solution = fsolve(alpha_optimizer, previous_alpha)
    return alpha_solution


def _compute_next_gamma(y, theta, big_jay):
    # Eq. 18
    shape_of_gamma = tuple(list(y.shape) + [big_jay])
    new_gamma = onp.empty(shape=shape_of_gamma, dtype=float)
    for j in range(big_jay):
        pi = theta[0, j]
        alpha = theta[1, j]
        beta = theta[2, j]
        times_to_br = tuple(onp.array(onp.array(y.shape) / onp.array(pi.shape), dtype=int))
        pi = broadcast_tile(pi, times_to_br)
        alpha = broadcast_tile(alpha, times_to_br)
        beta = broadcast_tile(beta, times_to_br)
        new_gamma[..., j] = pi * central_gamma_pdf(y, alpha=alpha, beta=beta)
    summation = onp.expand_dims(onp.sum(new_gamma, axis=-1), axis=-1)
    new_gamma = onp.nan_to_num(new_gamma / summation)
    return new_gamma


def _compute_next_theta(y, centered_mu, gamma, previous_alpha, y_shape_after_blocking):
    big_jay = len(centered_mu)
    shape_of_each_neighborhood = tuple(onp.array(onp.array(y.shape) / onp.array(y_shape_after_blocking), dtype=int))
    size_of_each_neighborhood = reduce(lambda m, n: m * n, shape_of_each_neighborhood)

    new_pi = onp.empty(shape=[big_jay] + y_shape_after_blocking)
    new_alpha = onp.empty(shape=[big_jay] + y_shape_after_blocking)
    new_beta = onp.empty(shape=[big_jay] + y_shape_after_blocking)

    blocked_y = block_matrix(mat=y, neighborhood_shape=shape_of_each_neighborhood)
    blocked_log_y = block_matrix(mat=onp.log(y), neighborhood_shape=shape_of_each_neighborhood)

    for j in range(big_jay):
        blocked_gamma_j = block_matrix(mat=gamma[..., j], neighborhood_shape=shape_of_each_neighborhood)
        first_numerator_summation = sum_over_each_neighborhood_on_blocked_matrix(
            blocked_gamma_j * blocked_y / centered_mu[j])
        second_numerator_summation = sum_over_each_neighborhood_on_blocked_matrix(
            blocked_gamma_j * (blocked_log_y - onp.log(centered_mu[j])))
        denominator_summation = sum_over_each_neighborhood_on_blocked_matrix(blocked_gamma_j)
        # Eq. 24
        right_hand_side = (first_numerator_summation - second_numerator_summation) / denominator_summation - 1
        # TODO: ravel and reshape work fine?
        alpha_initial_guess = previous_alpha[j, ...]
        vectorized_get_alphas_solution = onp.vectorize(_get_alphas_solution)
        new_alpha[j, ...] = vectorized_get_alphas_solution(right_hand_side, alpha_initial_guess)
        # constraint: alpha[j] * beta[j] = mu[j]
        new_beta[j, ...] = centered_mu[j] / new_alpha[j, ...]
        # Eq. 22
        new_pi[j, ...] = denominator_summation / size_of_each_neighborhood
    new_theta = onp.array([new_pi, new_alpha, new_beta])
    return new_theta


def run_first_algorithm(y: onp.array, mu: onp.array, neighborhood_size=0, delta=-1030, max_iter=5, tol=0.01,
                        non_central=False, initial_alpha=None):
    y_shape_after_blocking = []
    if neighborhood_size > 0:
        for ax in y.shape:
            assert ax % neighborhood_size == 0, f'''Input array's shape ({y.shape}) is not dividable to neighborhood size ({neighborhood_size}).'''  # noqa
            y_shape_after_blocking.append(ax // neighborhood_size)
    else:
        y_shape_after_blocking = [1 for _ in range(len(y.shape))]
    big_jay = len(mu)

    # centering the data
    if non_central:
        y = y - delta
        mu = mu - delta

    # initial guess of parameters
    if initial_alpha is None:
        initial_alpha = [2] * big_jay
    assert len(initial_alpha) == big_jay
    initial_beta = [mu[j] / initial_alpha[j] for j in range(big_jay)]
    initial_pi = [1 / big_jay] * big_jay
    # we assume that theta[0] = pi, theta[1] = alpha, theta[2] = beta
    shape_of_theta = tuple([3, big_jay] + y_shape_after_blocking)
    theta = onp.empty(shape=shape_of_theta, dtype=float)
    theta_before_expansion = onp.array([initial_pi, initial_alpha, initial_beta])
    for _ in range(len(y_shape_after_blocking)):
        theta_before_expansion = onp.expand_dims(theta_before_expansion, axis=-1)
    theta[...] = theta_before_expansion
    # compute initial gamma
    gamma = _compute_next_gamma(y=y, theta=theta, big_jay=big_jay)
    err = onp.Infinity
    n = 0
    while err > tol and n < max_iter:
        n += 1
        previous_alpha = theta[1, ...]
        new_theta = _compute_next_theta(y=y, centered_mu=mu, gamma=gamma, previous_alpha=previous_alpha,
                                        y_shape_after_blocking=y_shape_after_blocking)
        new_gamma = _compute_next_gamma(y=y, theta=theta, big_jay=big_jay)
        err = onp.linalg.norm(new_theta - theta) / onp.linalg.norm(theta)
        theta = new_theta
        gamma = new_gamma
        print(f'iteration: {n}, error: {err}')
    return theta, gamma
