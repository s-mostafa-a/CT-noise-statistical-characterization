from scipy.special import gamma
import numpy as np

from functools import reduce


def _get_hashed_number(numbers: np.array, bucket: tuple):
    reversed_bucket = tuple(reversed(bucket))
    res = []
    for rt in range(len(reversed_bucket)):
        res.append(numbers % reversed_bucket[rt])
        numbers = numbers // reversed_bucket[rt]
    res = np.array(tuple(reversed(res))).T
    return res


def split_matrix(arr, small_block_shape: tuple):
    times = tuple(np.array(np.array(arr.shape) / np.array(small_block_shape), dtype=int))
    whole_number_of_times = reduce(lambda x, y: x * y, times)
    numbers = np.array(list(range(whole_number_of_times)))
    all_indices = _get_hashed_number(numbers, times)
    patches = np.empty(times, dtype=object)
    for index in all_indices:
        lower = index * small_block_shape
        upper = (index + 1) * small_block_shape
        slices = []
        for i in range(len(lower)):
            slices.append(slice(lower[i], upper[i], 1))
        patches[tuple(index)] = arr[slices]
    return patches


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
