from scipy.special import gamma, loggamma
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


def block_matrix(mat: np.array, neighborhood_shape: tuple):
    shape_of_splitted_matrix = tuple(np.array(np.array(mat.shape) / np.array(neighborhood_shape), dtype=int))
    size = reduce(lambda x, y: x * y, shape_of_splitted_matrix)
    range_of_size = np.array(list(range(size)))
    all_multi_dimensional_indices = _get_hashed_number(range_of_size, shape_of_splitted_matrix)
    splitted = np.empty(shape_of_splitted_matrix, dtype=object)
    for multi_dimensional_index in all_multi_dimensional_indices:
        lower = multi_dimensional_index * neighborhood_shape
        upper = (multi_dimensional_index + 1) * neighborhood_shape
        slices = []
        for i in range(len(lower)):
            slices.append(slice(lower[i], upper[i], 1))
        splitted[tuple(multi_dimensional_index)] = mat[tuple(slices)]
    return splitted


def sum_over_each_neighborhood_on_blocked_matrix(mat: np.array):
    size = mat.size
    range_of_size = np.array(list(range(size)))
    all_multi_dimensional_indices = _get_hashed_number(range_of_size, mat.shape)
    res = np.empty(mat.shape, dtype=np.longdouble)
    for multi_dimensional_index in all_multi_dimensional_indices:
        res[tuple(multi_dimensional_index)] = np.sum(mat[tuple(multi_dimensional_index)])
    return res


def non_central_gamma_pdf(x, alpha, beta, delta):
    assert x >= delta, f'''x must be more than or equal to delta. x: {x}, delta: {delta}'''
    y = x - delta
    return central_gamma_pdf(y=y, alpha=alpha, beta=beta)


def argmax_3d(img: np.array):
    max1 = np.max(img, axis=0)
    argmax1 = np.argmax(img, axis=0)
    max2 = np.max(max1, axis=0)
    argmax2 = np.argmax(max1, axis=0)
    argmax3 = np.argmax(max2, axis=0)
    argmax3d = (argmax1[argmax2[argmax3], argmax3], argmax2[argmax3], argmax3)
    return argmax3d


def argmin_3d(img: np.array):
    min1 = np.min(img, axis=0)
    argmin1 = np.argmin(img, axis=0)
    min2 = np.max(min1, axis=0)
    argmin2 = np.argmin(min1, axis=0)
    argmin3 = np.argmin(min2, axis=0)
    argmin3d = (argmin1[argmin2[argmin3], argmin3], argmin2[argmin3], argmin3)
    return argmin3d


def argmax_2d(img: np.array):
    max1 = np.max(img, axis=0)
    argmax1 = np.argmax(img, axis=0)
    argmax2 = np.argmax(max1, axis=0)
    argmax2d = (argmax1[argmax2], argmax2)
    return argmax2d


def argmin_2d(img: np.array):
    min1 = np.min(img, axis=0)
    argmin1 = np.argmin(img, axis=0)
    argmin2 = np.argmax(min1, axis=0)
    argmin2d = (argmin1[argmin2], argmin2)
    return argmin2d


def central_gamma_log_pdf(y, alpha, beta):
    assert (alpha > 0).all() and (
            beta > 0).all(), f'''Alpha and Beta must be more than zero. Alpha: {alpha}, Beta: {beta}'''
    assert (y > 0).all(), f'''y value Must be more than zero. y: {y}'''
    a = (alpha - 1) * np.log(y)
    b = (y / beta)
    c = alpha * np.log(beta)
    d = loggamma(alpha)
    return a - b - c - d


def central_gamma_pdf(y, alpha, beta):
    assert (alpha > 0).all() and (
            beta > 0).all(), f'''Alpha and Beta must be more than zero. Alpha: {alpha}, Beta: {beta}'''
    # big number turns to np.inf then, big * 0.0 should be 0.0, but it takes np.nan!
    # so, we should change it with nan_to_num()
    form = np.nan_to_num(np.power(y, (alpha - 1)) * np.exp(-y / beta))
    denominator = np.nan_to_num(np.power(beta, alpha) * gamma(alpha))
    return np.nan_to_num(form / denominator)


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


def expand(small_img, half_neigh_size: int):
    for s in small_img.shape:
        assert half_neigh_size * 2 + 1 <= s, f'neighborhood_size: {half_neigh_size * 2 + 1}, shape: {small_img.shape}'
    padding = np.array([half_neigh_size for _ in small_img.shape])
    big_shape = tuple((np.array(small_img.shape) - padding * 2) * (half_neigh_size * 2 + 1))
    big_img = np.zeros(big_shape, dtype=float)
    size = reduce(lambda x, y: x * y, small_img.shape)
    range_of_size = np.array(list(range(size)))
    all_multi_dimensional_indices = _get_hashed_number(range_of_size, small_img.shape)
    for center_of_small in all_multi_dimensional_indices:
        center_of_big = (center_of_small - half_neigh_size) * (half_neigh_size * 2 + 1) + half_neigh_size
        small_lower = center_of_small - padding
        small_upper = center_of_small + padding + 1
        big_lower = center_of_big - padding
        big_upper = center_of_big + padding + 1
        if (small_lower < 0).any() or (small_upper > np.array(small_img.shape)).any():
            continue
        if (big_lower < 0).any() or (big_upper > np.array(big_img.shape)).any():
            continue
        small_slices = []
        big_slices = []
        for i in range(len(small_lower)):
            small_slices.append(slice(small_lower[i], small_upper[i], 1))
            big_slices.append(slice(big_lower[i], big_upper[i], 1))
        big_img[big_slices] = small_img[small_slices]
    return big_img


def contract(big_img, half_neigh_size):
    neighborhood_size = half_neigh_size * 2 + 1
    small_shape = tuple(np.array(np.array(big_img.shape) / neighborhood_size, dtype=int))
    small_img = np.empty(small_shape, dtype=float)
    size = reduce(lambda x, y: x * y, small_img.shape)
    range_of_size = np.array(list(range(size)))
    all_multi_dimensional_indices = _get_hashed_number(range_of_size, small_img.shape)
    for center_of_small in all_multi_dimensional_indices:
        center_of_big = center_of_small * (half_neigh_size * 2 + 1) + half_neigh_size
        small_img[tuple(center_of_small)] = big_img[tuple(center_of_big)]
    return small_img
