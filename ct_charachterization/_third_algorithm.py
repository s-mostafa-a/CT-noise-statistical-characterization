import numpy as np
from .utility.utils import broadcast_tile, block_matrix, sum_over_each_neighborhood_on_blocked_matrix
from ._second_algorithm import run_second_algorithm


def run_third_algorithm(y: np.array, mu: np.array, neighborhood_size: int, delta=-1030, max_iter=10, tol=0.0000001,
                        constant_c=10, non_central=False):
    big_jay = len(mu)
    if non_central:
        mu = mu - delta
        y = y - delta
    theta, gamma = run_second_algorithm(y, mu=mu, neighborhood_size=neighborhood_size, delta=delta, max_iter=max_iter,
                                        tol=tol)
    pi = theta[0, ...]
    axes = list(range(len(pi.shape)))
    axes.pop(0)
    axes.append(0)
    pi = np.transpose(pi, axes=axes)
    # print('pi:', pi.shape)
    times_pi_to_be_broad_casted = np.array(np.array(y.shape) / np.array(pi.shape[:-1]), dtype=int)
    pi = broadcast_tile(pi, tuple(list(times_pi_to_be_broad_casted) + [1]))
    # print('pi:', pi.shape)
    shape_of_each_neighborhood = tuple([neighborhood_size for _ in y.shape])
    blocked_y = block_matrix(mat=y, neighborhood_shape=shape_of_each_neighborhood)
    blocked_radical_y = block_matrix(mat=np.sqrt(y), neighborhood_shape=shape_of_each_neighborhood)
    moments_size = tuple(list(y.shape) + [big_jay])
    first_local_sample_conditioned_moment = np.empty(moments_size, dtype=float)
    second_local_sample_conditioned_moment = np.empty(moments_size, dtype=float)
    for j in range(big_jay):
        blocked_gamma_j = block_matrix(mat=gamma[..., j], neighborhood_shape=shape_of_each_neighborhood)
        first_numerator_summation = broadcast_tile(
            sum_over_each_neighborhood_on_blocked_matrix(blocked_gamma_j * blocked_radical_y),
            shape_of_each_neighborhood)
        second_numerator_summation = broadcast_tile(
            sum_over_each_neighborhood_on_blocked_matrix(blocked_gamma_j * blocked_y), shape_of_each_neighborhood)
        denominator_summation = broadcast_tile(sum_over_each_neighborhood_on_blocked_matrix(blocked_gamma_j),
                                               shape_of_each_neighborhood)
        first_local_sample_conditioned_moment[..., j] = first_numerator_summation / denominator_summation
        second_local_sample_conditioned_moment[..., j] = second_numerator_summation / denominator_summation
    first_local_sample_conditioned_moment = np.sum(first_local_sample_conditioned_moment * gamma, axis=-1)
    second_local_sample_conditioned_moment = np.sum(second_local_sample_conditioned_moment * gamma, axis=-1)
    local_sample_variance = second_local_sample_conditioned_moment - np.power(first_local_sample_conditioned_moment, 2)
    y_stab = (constant_c * (np.sqrt(y) - first_local_sample_conditioned_moment) / np.sqrt(
        local_sample_variance)) + second_local_sample_conditioned_moment
    return y_stab
