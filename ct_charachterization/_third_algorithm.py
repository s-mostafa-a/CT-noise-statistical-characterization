import numpy as onp
from .utility.utils import broadcast_tile, block_matrix, sum_over_each_neighborhood_on_blocked_matrix, expand
from ._second_algorithm import run_second_algorithm
from ct_charachterization.utility.utils import expand, contract


def run_third_algorithm_gamma_instead_of_pi(y: onp.array, mu: onp.array, neighborhood_size=7, delta=-1030, max_iter=10,
                                            tol=0.01, constant_c=10, non_central=False):
    if non_central:
        mu = mu - delta
        y = y - delta
    half_neigh = int(neighborhood_size / 2)
    neigh_size_including_center = half_neigh * 2 + 1
    big_y = expand(small_img=y, half_neigh_size=half_neigh)
    big_jay = len(mu)
    theta, gamma = run_second_algorithm(big_y, mu=mu, neighborhood_size=neigh_size_including_center, delta=delta,
                                        max_iter=max_iter, tol=tol)
    shape_of_each_neighborhood = tuple([neigh_size_including_center for _ in big_y.shape])
    blocked_y = block_matrix(mat=big_y, neighborhood_shape=shape_of_each_neighborhood)
    blocked_radical_y = block_matrix(mat=onp.sqrt(big_y), neighborhood_shape=shape_of_each_neighborhood)
    moments_size = tuple(list(big_y.shape) + [big_jay])
    first_local_sample_conditioned_moment = onp.empty(moments_size, dtype=float)
    second_local_sample_conditioned_moment = onp.empty(moments_size, dtype=float)
    for j in range(big_jay):
        blocked_gamma_j = block_matrix(mat=gamma[..., j], neighborhood_shape=shape_of_each_neighborhood)
        first_numerator_summation = broadcast_tile(
            sum_over_each_neighborhood_on_blocked_matrix(blocked_gamma_j * blocked_radical_y),
            shape_of_each_neighborhood)
        second_numerator_summation = broadcast_tile(
            sum_over_each_neighborhood_on_blocked_matrix(blocked_gamma_j * blocked_y),
            shape_of_each_neighborhood)
        denominator_summation = broadcast_tile(sum_over_each_neighborhood_on_blocked_matrix(blocked_gamma_j),
                                               shape_of_each_neighborhood)
        denominator_summation[denominator_summation == 0] = 1
        first_local_sample_conditioned_moment[..., j] = first_numerator_summation / denominator_summation
        second_local_sample_conditioned_moment[..., j] = second_numerator_summation / denominator_summation

    first_local_sample_conditioned_moment = onp.sum(first_local_sample_conditioned_moment * gamma, axis=-1)
    second_local_sample_conditioned_moment = onp.sum(second_local_sample_conditioned_moment * gamma, axis=-1)
    local_sample_variance = second_local_sample_conditioned_moment - onp.power(first_local_sample_conditioned_moment, 2)
    y_stab = (constant_c * (onp.sqrt(big_y) - first_local_sample_conditioned_moment) / onp.sqrt(
        local_sample_variance)) + second_local_sample_conditioned_moment
    return contract(big_img=y_stab, half_neigh_size=half_neigh)


def run_third_algorithm_expectation_at_the_end(y: onp.array, mu: onp.array, neighborhood_size=7, delta=-1030,
                                               max_iter=10, tol=0.01,
                                               constant_c=2, non_central=False):
    if non_central:
        mu = mu - delta
        y = y - delta
    half_neigh = int(neighborhood_size / 2)
    y_lower = onp.array([half_neigh for _ in range(len(y.shape))])
    y_upper = onp.array(y.shape) - y_lower
    y_slices = []
    for i in range(len(y_lower)):
        y_slices.append(slice(y_lower[i], y_upper[i], 1))
    neigh_size_including_center = half_neigh * 2 + 1
    big_y = expand(small_img=y, half_neigh_size=half_neigh)
    big_jay = len(mu)
    theta, gamma = run_second_algorithm(big_y, mu=mu, neighborhood_size=neigh_size_including_center, delta=delta,
                                        max_iter=max_iter,
                                        tol=tol)
    pi = theta[0, ...]
    shape_of_each_neighborhood = tuple([neigh_size_including_center for _ in big_y.shape])
    blocked_y = block_matrix(mat=big_y, neighborhood_shape=shape_of_each_neighborhood)
    blocked_radical_y = block_matrix(mat=onp.sqrt(big_y), neighborhood_shape=shape_of_each_neighborhood)
    moments_size = tuple([big_jay] + [int(i / neigh_size_including_center) for i in big_y.shape])
    first_local_sample_conditioned_moment = onp.empty(moments_size, dtype=float)
    second_local_sample_conditioned_moment = onp.empty(moments_size, dtype=float)
    variances = onp.empty(moments_size, dtype=float)
    y_stab = onp.empty(moments_size, dtype=float)
    for j in range(big_jay):
        blocked_gamma_j = block_matrix(mat=gamma[..., j], neighborhood_shape=shape_of_each_neighborhood)
        first_numerator_summation = sum_over_each_neighborhood_on_blocked_matrix(blocked_gamma_j * blocked_radical_y)
        second_numerator_summation = sum_over_each_neighborhood_on_blocked_matrix(blocked_gamma_j * blocked_y)
        denominator_summation = sum_over_each_neighborhood_on_blocked_matrix(blocked_gamma_j)
        # This does not affect the results. Just to remove division by zero warnings.
        denominator_summation[denominator_summation == 0] = 1
        first_local_sample_conditioned_moment[j, ...] = first_numerator_summation / denominator_summation
        second_local_sample_conditioned_moment[j, ...] = second_numerator_summation / denominator_summation
        vr = (second_local_sample_conditioned_moment[j, ...] - onp.power(
            first_local_sample_conditioned_moment[j, ...], 2))
        # TODO This is safety, show it on the presentation
        vr[vr <= 0] = constant_c ** 2
        vr[vr ** 2 <= 0.5] = constant_c ** 2
        variances[j, ...] = vr
        y_stab[j, ...] = (constant_c * (onp.sqrt(y[y_slices]) -
                                        first_local_sample_conditioned_moment[j, ...]) / onp.sqrt(
            variances[j, ...])) + second_local_sample_conditioned_moment[j, ...]
    y_stab = onp.sum(y_stab * pi, axis=0)
    return y_stab


def run_third_algorithm_expectation_at_the_beginning(y: onp.array, mu: onp.array, neighborhood_size=7, delta=-1030,
                                                     max_iter=10, tol=0.01, constant_c=2, non_central=False):
    if non_central:
        mu = mu - delta
        y = y - delta
    half_neigh = int(neighborhood_size / 2)
    y_lower = onp.array([half_neigh for _ in range(len(y.shape))])
    y_upper = onp.array(y.shape) - y_lower
    y_slices = []
    for i in range(len(y_lower)):
        y_slices.append(slice(y_lower[i], y_upper[i], 1))
    neigh_size_including_center = half_neigh * 2 + 1
    big_y = expand(small_img=y, half_neigh_size=half_neigh)
    big_jay = len(mu)
    theta, gamma = run_second_algorithm(big_y, mu=mu, neighborhood_size=neigh_size_including_center, delta=delta,
                                        max_iter=max_iter,
                                        tol=tol)
    pi = theta[0, ...]
    shape_of_each_neighborhood = tuple([neigh_size_including_center for _ in big_y.shape])
    blocked_y = block_matrix(mat=big_y, neighborhood_shape=shape_of_each_neighborhood)
    blocked_radical_y = block_matrix(mat=onp.sqrt(big_y), neighborhood_shape=shape_of_each_neighborhood)
    moments_size = tuple([big_jay] + [int(i / neigh_size_including_center) for i in big_y.shape])
    first_local_sample_conditioned_moment = onp.empty(moments_size, dtype=float)
    second_local_sample_conditioned_moment = onp.empty(moments_size, dtype=float)
    for j in range(big_jay):
        blocked_gamma_j = block_matrix(mat=gamma[..., j], neighborhood_shape=shape_of_each_neighborhood)
        first_numerator_summation = sum_over_each_neighborhood_on_blocked_matrix(blocked_gamma_j * blocked_radical_y)
        second_numerator_summation = sum_over_each_neighborhood_on_blocked_matrix(blocked_gamma_j * blocked_y)
        denominator_summation = sum_over_each_neighborhood_on_blocked_matrix(blocked_gamma_j)
        # This does not affect the results. Just to remove division by zero warnings.
        denominator_summation[denominator_summation == 0] = 1
        first_local_sample_conditioned_moment[j, ...] = first_numerator_summation / denominator_summation
        second_local_sample_conditioned_moment[j, ...] = second_numerator_summation / denominator_summation
    first_local_sample_conditioned_moment = onp.sum(first_local_sample_conditioned_moment * pi, axis=0)
    second_local_sample_conditioned_moment = onp.sum(second_local_sample_conditioned_moment * pi, axis=0)
    local_sample_variance = second_local_sample_conditioned_moment - onp.power(first_local_sample_conditioned_moment, 2)
    y_stab = (constant_c * (onp.sqrt(y[y_slices]) - first_local_sample_conditioned_moment) / onp.sqrt(
        local_sample_variance)) + second_local_sample_conditioned_moment
    return y_stab


def run_linear_combination_of_components(y: onp.array, mu: onp.array, neighborhood_size=32, delta=-1030, max_iter=10,
                                         tol=0.01, non_central=False):
    if non_central:
        mu = mu - delta
        y = y - delta
    first_shape = y.shape[0]
    second_shape = y.shape[1]
    half_neigh = int(neighborhood_size / 2)
    big_y = expand(small_img=y, half_neigh_size=half_neigh)
    theta, gamma = run_second_algorithm(big_y, mu=mu, neighborhood_size=neighborhood_size, delta=delta,
                                        max_iter=max_iter, tol=tol)
    pi = theta[0, ...]
    onp.sqrt(y[half_neigh:first_shape - half_neigh, half_neigh: second_shape - half_neigh])
    combination = onp.empty((first_shape - neighborhood_size, second_shape - neighborhood_size), dtype=float)
    for i in range(first_shape - neighborhood_size):
        for j in range(second_shape - neighborhood_size):
            combination[i, j] = onp.sum(mu * pi[..., i, j])
    return combination
