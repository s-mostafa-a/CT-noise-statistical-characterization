import numpy as np
from .utility.utils import broadcast_tile, block_matrix, sum_over_each_neighborhood_on_blocked_matrix, argmin_2d, \
    argmax_3d, argmax_2d
from ._second_algorithm import run_second_algorithm
from ct_charachterization.utility.utils import expand, contract
from matplotlib import pyplot as plt


def run_third_algorithm_gamma_instead_of_pi(y: np.array, mu: np.array, neighborhood_size: int, delta=-1030, max_iter=10,
                                            tol=0.01,
                                            constant_c=10, non_central=False):
    big_jay = len(mu)
    if non_central:
        mu = mu - delta
        y = y - delta

    first_shape = y.shape[0]
    second_shape = y.shape[1]
    half_neigh = int(neighborhood_size / 2)
    big_y = expand(small_img=y, neighborhood_size=neighborhood_size)
    big_y = big_y[half_neigh * neighborhood_size:(first_shape - half_neigh) * neighborhood_size,
            half_neigh * neighborhood_size:(second_shape - half_neigh) * neighborhood_size]
    theta, gamma = run_second_algorithm(big_y, mu=mu, neighborhood_size=neighborhood_size, delta=delta,
                                        max_iter=max_iter,
                                        tol=tol)
    shape_of_each_neighborhood = tuple([neighborhood_size for _ in big_y.shape])
    blocked_y = block_matrix(mat=big_y, neighborhood_shape=shape_of_each_neighborhood)
    blocked_radical_y = block_matrix(mat=np.sqrt(big_y), neighborhood_shape=shape_of_each_neighborhood)
    moments_size = tuple(list(big_y.shape) + [big_jay])
    first_local_sample_conditioned_moment = np.empty(moments_size, dtype=float)
    second_local_sample_conditioned_moment = np.empty(moments_size, dtype=float)
    for j in range(big_jay):
        blocked_gamma_j = block_matrix(mat=gamma[..., j], neighborhood_shape=shape_of_each_neighborhood)
        first_numerator_summation = np.nan_to_num(
            broadcast_tile(sum_over_each_neighborhood_on_blocked_matrix(blocked_gamma_j * blocked_radical_y),
                           shape_of_each_neighborhood))
        second_numerator_summation = np.nan_to_num(
            broadcast_tile(sum_over_each_neighborhood_on_blocked_matrix(blocked_gamma_j * blocked_y),
                           shape_of_each_neighborhood))
        denominator_summation = np.nan_to_num(
            broadcast_tile(sum_over_each_neighborhood_on_blocked_matrix(blocked_gamma_j), shape_of_each_neighborhood))
        first_local_sample_conditioned_moment[..., j] = np.nan_to_num(first_numerator_summation / denominator_summation)
        second_local_sample_conditioned_moment[..., j] = np.nan_to_num(
            second_numerator_summation / denominator_summation)
    first_local_sample_conditioned_moment = np.sum(first_local_sample_conditioned_moment * gamma, axis=-1)
    second_local_sample_conditioned_moment = np.sum(second_local_sample_conditioned_moment * gamma, axis=-1)
    local_sample_variance = second_local_sample_conditioned_moment - np.power(first_local_sample_conditioned_moment, 2)
    y_stab = (constant_c * (np.sqrt(big_y) - first_local_sample_conditioned_moment) / np.sqrt(
        local_sample_variance)) + second_local_sample_conditioned_moment
    return contract(big_img=y_stab, neighborhood_size=neighborhood_size)


def run_third_algorithm_expectation_at_the_end(y: np.array, mu: np.array, neighborhood_size=32, delta=-1030,
                                               max_iter=10, tol=0.01,
                                               constant_c=2, non_central=False):
    if non_central:
        mu = mu - delta
        y = y - delta
    first_shape = y.shape[0]
    second_shape = y.shape[1]
    big_y = expand(small_img=y, neighborhood_size=neighborhood_size)
    half_neigh = int(neighborhood_size / 2)
    big_y = big_y[half_neigh * neighborhood_size:(first_shape - half_neigh) * neighborhood_size,
            half_neigh * neighborhood_size:(second_shape - half_neigh) * neighborhood_size]
    big_jay = len(mu)
    theta, gamma = run_second_algorithm(big_y, mu=mu, neighborhood_size=neighborhood_size, delta=delta,
                                        max_iter=max_iter,
                                        tol=tol)
    pi = theta[0, ...]
    shape_of_each_neighborhood = tuple([neighborhood_size for _ in big_y.shape])
    blocked_y = block_matrix(mat=big_y, neighborhood_shape=shape_of_each_neighborhood)
    blocked_radical_y = block_matrix(mat=np.sqrt(big_y), neighborhood_shape=shape_of_each_neighborhood)
    moments_size = tuple([big_jay] + [int(i / neighborhood_size) for i in big_y.shape])
    first_local_sample_conditioned_moment = np.empty(moments_size, dtype=float)
    second_local_sample_conditioned_moment = np.empty(moments_size, dtype=float)
    variances = np.empty(moments_size, dtype=float)
    y_stab = np.empty(moments_size, dtype=float)
    for j in range(big_jay):
        blocked_gamma_j = block_matrix(mat=gamma[..., j], neighborhood_shape=shape_of_each_neighborhood)
        first_numerator_summation = sum_over_each_neighborhood_on_blocked_matrix(blocked_gamma_j * blocked_radical_y)
        second_numerator_summation = sum_over_each_neighborhood_on_blocked_matrix(blocked_gamma_j * blocked_y)
        denominator_summation = sum_over_each_neighborhood_on_blocked_matrix(blocked_gamma_j)
        # This does not affect the results. Just to remove warnings.
        denominator_summation[denominator_summation == 0] = 1
        first_local_sample_conditioned_moment[j, ...] = first_numerator_summation / denominator_summation
        second_local_sample_conditioned_moment[j, ...] = second_numerator_summation / denominator_summation
        vr = (second_local_sample_conditioned_moment[j, ...] - np.power(
            first_local_sample_conditioned_moment[j, ...], 2))
        # Safety
        vr[vr <= 0] = 1
        variances[j, ...] = vr
        y_stab[j, ...] = (constant_c * (np.sqrt(y[half_neigh:first_shape - half_neigh,
                                                half_neigh: second_shape - half_neigh]) -
                                        first_local_sample_conditioned_moment[j, ...]) / np.sqrt(
            variances[j, ...])) + second_local_sample_conditioned_moment[j, ...]
    y_stab = np.sum(y_stab * pi, axis=0)
    return y_stab


def run_third_algorithm_expectation_at_the_beginning(y: np.array, mu: np.array, neighborhood_size=32, delta=-1030,
                                                     max_iter=10, tol=0.01,
                                                     constant_c=2, non_central=False):
    if non_central:
        mu = mu - delta
        y = y - delta
    first_shape = y.shape[0]
    second_shape = y.shape[1]
    big_y = expand(small_img=y, neighborhood_size=neighborhood_size)
    half_neigh = int(neighborhood_size / 2)
    big_y = big_y[half_neigh * neighborhood_size:(first_shape - half_neigh) * neighborhood_size,
            half_neigh * neighborhood_size:(second_shape - half_neigh) * neighborhood_size]
    big_jay = len(mu)
    theta, gamma = run_second_algorithm(big_y, mu=mu, neighborhood_size=neighborhood_size, delta=delta,
                                        max_iter=max_iter,
                                        tol=tol)
    pi = theta[0, ...]
    shape_of_each_neighborhood = tuple([neighborhood_size for _ in big_y.shape])
    blocked_y = block_matrix(mat=big_y, neighborhood_shape=shape_of_each_neighborhood)
    blocked_radical_y = block_matrix(mat=np.sqrt(big_y), neighborhood_shape=shape_of_each_neighborhood)
    moments_size = tuple([big_jay] + [int(i / neighborhood_size) for i in big_y.shape])
    first_local_sample_conditioned_moment = np.empty(moments_size, dtype=float)
    second_local_sample_conditioned_moment = np.empty(moments_size, dtype=float)
    for j in range(big_jay):
        blocked_gamma_j = block_matrix(mat=gamma[..., j], neighborhood_shape=shape_of_each_neighborhood)
        first_numerator_summation = np.nan_to_num(
            sum_over_each_neighborhood_on_blocked_matrix(blocked_gamma_j * blocked_radical_y))
        second_numerator_summation = np.nan_to_num(
            sum_over_each_neighborhood_on_blocked_matrix(blocked_gamma_j * blocked_y))
        denominator_summation = sum_over_each_neighborhood_on_blocked_matrix(blocked_gamma_j)
        first_local_sample_conditioned_moment[j, ...] = first_numerator_summation / denominator_summation
        second_local_sample_conditioned_moment[j, ...] = second_numerator_summation / denominator_summation
    first_local_sample_conditioned_moment = np.sum(first_local_sample_conditioned_moment * pi, axis=0)
    second_local_sample_conditioned_moment = np.sum(second_local_sample_conditioned_moment * pi, axis=0)
    local_sample_variance = second_local_sample_conditioned_moment - np.power(first_local_sample_conditioned_moment, 2)
    y_stab = (constant_c * (np.sqrt(y[half_neigh:first_shape - half_neigh,
                                    half_neigh: second_shape - half_neigh]) - first_local_sample_conditioned_moment) / np.sqrt(
        local_sample_variance)) + second_local_sample_conditioned_moment
    return y_stab


def run_linear_combination_of_components(y: np.array, mu: np.array, neighborhood_size=32, delta=-1030, max_iter=10,
                                         tol=0.01, non_central=False):
    if non_central:
        mu = mu - delta
        y = y - delta
    first_shape = y.shape[0]
    second_shape = y.shape[1]
    big_y = expand(small_img=y, neighborhood_size=neighborhood_size)
    half_neigh = int(neighborhood_size / 2)
    big_y = big_y[half_neigh * neighborhood_size:(first_shape - half_neigh) * neighborhood_size,
            half_neigh * neighborhood_size:(second_shape - half_neigh) * neighborhood_size]
    theta, gamma = run_second_algorithm(big_y, mu=mu, neighborhood_size=neighborhood_size, delta=delta,
                                        max_iter=max_iter, tol=tol)
    pi = theta[0, ...]
    np.sqrt(y[half_neigh:first_shape - half_neigh, half_neigh: second_shape - half_neigh])
    combination = np.empty((first_shape - neighborhood_size, second_shape - neighborhood_size), dtype=float)
    for i in range(first_shape - neighborhood_size):
        for j in range(second_shape - neighborhood_size):
            combination[i, j] = np.sum(mu * pi[..., i, j])
    return combination
