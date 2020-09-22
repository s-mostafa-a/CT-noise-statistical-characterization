from scipy.special import gamma
import numpy as np

from functools import reduce


def _get_hashed_number(numbers_in_range_of_size: np.array, shape: tuple):
    reversed_bucket = tuple(reversed(shape))
    res = []
    for rt in range(len(reversed_bucket)):
        res.append(numbers_in_range_of_size % reversed_bucket[rt])
        numbers_in_range_of_size = numbers_in_range_of_size // reversed_bucket[rt]
    res = np.array(tuple(reversed(res))).T
    return res


def split_matrix(mat: np.array, small_block_shape: tuple):
    times = tuple(np.array(np.array(mat.shape) / np.array(small_block_shape), dtype=int))
    size = reduce(lambda x, y: x * y, times)
    range_of_size = np.array(list(range(size)))
    all_multi_dimensional_indices = _get_hashed_number(range_of_size, times)
    patches = np.empty(times, dtype=object)
    for multi_dimensional_index in all_multi_dimensional_indices:
        lower = multi_dimensional_index * small_block_shape
        upper = (multi_dimensional_index + 1) * small_block_shape
        slices = []
        for i in range(len(lower)):
            slices.append(slice(lower[i], upper[i], 1))
        patches[tuple(multi_dimensional_index)] = mat[tuple(slices)]
    return patches


def sum_of_each_patch(mat: np.array):
    size = mat.size
    range_of_size = np.array(list(range(size)))
    all_multi_dimensional_indices = _get_hashed_number(range_of_size, mat.shape)
    res = np.empty(mat.shape, dtype=mat[tuple(all_multi_dimensional_indices[0])].dtype)
    for multi_dimensional_index in all_multi_dimensional_indices:
        res[tuple(multi_dimensional_index)] = np.sum(mat[tuple(multi_dimensional_index)])
    return res


def non_central_gamma_pdf(x, alpha, beta, delta):
    assert x >= delta, f'''x must be more than or equal to delta. x: {x}, delta: {delta}'''
    y = x - delta
    return central_gamma_pdf(y=y, alpha=alpha, beta=beta)


def central_gamma_pdf(y, alpha, beta):
    assert (alpha > 0).all() and (
            beta > 0).all(), f'''Alpha and Beta must be more than zero. Alpha: {alpha}, Beta: {beta}'''
    form = np.power(y, (alpha - 1)) * np.exp(-y / beta)
    denominator = np.power(beta, alpha) * gamma(alpha)
    return form / denominator


def broadcast_tile(matrix, times: tuple):
    assert len(matrix.shape) == len(times), f'matrix.shape: {matrix.shape}, times: {times}'
    lsd = tuple([matrix.shape[i] * times[i] for i in range(len(times))])
    reshape_to = []
    for item in matrix.shape:
        reshape_to.append(item)
        reshape_to.append(1)
    reshape_to = tuple(reshape_to)
    final_shape = []
    for i in range(len(times)):
        final_shape.append(matrix.shape[i])
        final_shape.append(times[i])
    final_shape = tuple(final_shape)
    return np.broadcast_to(matrix.reshape(reshape_to), final_shape).reshape(lsd)
