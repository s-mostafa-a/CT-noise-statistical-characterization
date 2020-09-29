import numpy as np
from ct_charachterization import run_first_algorithms
import matplotlib.pyplot as plt
from ct_charachterization.utility.utils import broadcast_tile, block_matrix, \
    sum_over_each_neighborhood_on_blocked_matrix, expand, contract


def run_third_algorithm(y: np.array, mu: np.array, neighborhood_size: int, delta=-1030, max_iter=10, tol=0.0000001,
                        constant_c=10, non_central=False):
    big_jay = len(mu)
    if non_central:
        mu = mu - delta
        y = y - delta
    theta, gamma = run_first_algorithms(y, mu=mu, neighborhood_size=neighborhood_size, delta=delta, max_iter=max_iter,
                                        tol=tol)
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
    local_sample_variance = second_local_sample_conditioned_moment - np.power(first_local_sample_conditioned_moment, 2)
    y_stab = (constant_c * (np.expand_dims(np.sqrt(y),
                                           axis=-1) - first_local_sample_conditioned_moment) / local_sample_variance) + second_local_sample_conditioned_moment  # noqa
    return y_stab


def _run_third_algorithm_without_fixed_neighborhood(y: np.array, mu: np.array, neighborhood_size: int, delta=-1030,
                                                    max_iter=10, tol=0.0000001, constant_c=10, non_central=False):
    big_y = expand(small_img=y, neighborhood_size=neighborhood_size)
    print('bigged')
    plt.imshow(big_y, cmap='gray')
    plt.show()
    big_y = big_y[16 * 32:112 * 32, 16 * 32:112 * 32]
    plt.imshow(big_y, cmap='gray')
    plt.show()
    big_y_stab = run_third_algorithm(y=big_y, mu=mu, neighborhood_size=neighborhood_size, delta=delta,
                                     max_iter=max_iter, tol=tol, constant_c=constant_c, non_central=non_central)
    print('stable')
    y_stab = np.empty(tuple(list([96, 96]) + [len(mu)]), dtype=float)
    for j in range(len(mu)):
        y_stab[..., j] = contract(big_img=big_y_stab[..., j], neighborhood_size=neighborhood_size)
    print('smalled')
    return y_stab


if __name__ == '__main__':
    # mu_5 = np.array([-1000, -700, -90, 50, 300])
    mu_5 = np.array([-870])
    # MU = np.array([340, 240, 100, 0, -160, -370, -540, -810, -987])
    img = np.load(f'''../resources/luna_cropped.npy''')
    print(img.shape)
    img = img[0:140, 0:140]
    plt.imshow(img, cmap='gray')
    plt.show()
    stabilized_y = _run_third_algorithm_without_fixed_neighborhood(img, mu_5, non_central=True, constant_c=10,
                                                                   neighborhood_size=70)
    plt.imshow(img, cmap='gray')
    plt.show()
    print(img.shape)
    sy = stabilized_y[..., 0]
    print(sy.shape)
    plt.imshow(sy, cmap='gray')
    plt.show()
    print(np.min(sy) - 1030, np.mean(sy) - 1030, np.max(sy) - 1030)
