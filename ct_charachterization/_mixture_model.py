import numpy as np

from ct_charachterization.utility.utils import central_gamma_pdf, broadcast_tile
from scipy.optimize import fsolve
from scipy.special import digamma


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
    axis = tuple(range(len(y.shape)))
    big_jay = len(centered_mu)
    first_form_summation = np.sum(gamma * (np.expand_dims(y, axis=-1) / centered_mu), axis=axis).reshape(big_jay)
    second_form_summation = np.sum(gamma * np.log(np.expand_dims(y, axis=-1) / centered_mu), axis=axis).reshape(
        big_jay)
    denominator_summation = np.sum(gamma, axis=axis).reshape(big_jay)
    # Eq. 24
    right_hand_side = (first_form_summation - second_form_summation) / denominator_summation - 1
    print(right_hand_side.shape)
    to_be_found = lambda alp: right_hand_side - (np.log(alp) - digamma(alp))
    for j in range(big_jay):
        alpha = theta[1, j, ...]
        times_to_br = tuple(np.array(np.array(y.shape) / np.array(alpha.shape), dtype=int))
        alpha = broadcast_tile(alpha, times_to_br)
        print(right_hand_side.shape)
        alpha_initial_guess = alpha
        exit(0)
        alpha_solution = fsolve(to_be_found, alpha_initial_guess)
    new_alpha = alpha_solution
    # constraint: alpha[j] * beta[j] = mu[j]
    new_beta = np.array(centered_mu) / new_alpha
    # Eq. 22
    new_pi = denominator_summation / y.size
    new_theta = np.array([new_pi, new_alpha, new_beta])
    return new_theta


def run_first_algorithms(y: np.array, mu: np.array, neighborhood_size: int, delta=-1030, max_iter=10, tol=0.01,
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
    theta[...] = np.expand_dims(np.array(
        [[1 / big_jay] * big_jay, [2] * big_jay, [mu[j] / 2 for j in range(big_jay)]]), axis=-1)
    # OGAY
    # compute initial gamma
    shape_of_gamma = tuple(list(y.shape) + [big_jay])
    gamma = _compute_next_gamma(y, big_jay, shape_of_gamma, theta)
    # not sure, error less OGAY
    new_theta = _compute_next_theta(y, mu, gamma, theta)
    exit(0)
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
