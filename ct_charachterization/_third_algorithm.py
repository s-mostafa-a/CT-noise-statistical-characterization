import numpy as np
from ct_charachterization import run_first_algorithms
import matplotlib.pyplot as plt
from ct_charachterization.utility.utils import broadcast_tile


def run_third_algorithm(x, mu, delta=-1030, max_iter=10, tol=0.01, constant_c=10):
    big_jay = len(mu)
    centered_mu = mu - delta
    y = x - delta
    theta, gamma = run_first_algorithms(y, centered_mu=centered_mu, delta=delta, max_iter=max_iter, tol=tol)
    # sclm: sample_conditioned_local_moment
    whole_axises = tuple(range(len(gamma.shape)))
    shape_of_mini_matrices = [1 for _ in y.shape]
    shape_of_mini_matrices.append(big_jay)
    shape_of_mini_matrices = tuple(shape_of_mini_matrices)
    form_of_first_mini_sclm = np.sum(np.sqrt(np.expand_dims(y, axis=-1)) * gamma, axis=whole_axises[:-1]).reshape(
        shape_of_mini_matrices)
    form_of_second_mini_sclm = np.sum(np.expand_dims(y, axis=-1) * gamma, axis=whole_axises[:-1]).reshape(
        shape_of_mini_matrices)
    denominator_summation = np.sum(gamma, axis=whole_axises[:-1]).reshape(shape_of_mini_matrices)
    first_mini_sclm = form_of_first_mini_sclm / denominator_summation
    second_mini_sclm = form_of_second_mini_sclm / denominator_summation
    br_to_shape = list(y.shape)
    br_to_shape.append(1)
    br_to_shape = tuple(br_to_shape)
    first_sclm = np.sum(broadcast_tile(first_mini_sclm, br_to_shape) * theta[0, :], axis=whole_axises[-1])
    second_sclm = np.sum(broadcast_tile(second_mini_sclm, br_to_shape) * theta[0, :], axis=whole_axises[-1])
    var_of_radical_y = second_sclm - np.power(first_sclm, 2)
    stable_y = constant_c * (np.sqrt(y) - first_sclm) / np.sqrt(var_of_radical_y) + second_sclm
    return stable_y, theta, gamma


if __name__ == '__main__':
    MU = np.array([-1000, -700, -90, 50, 300])
    # MU = np.array([340, 240, 100, 0, -160, -370, -540, -810, -987])
    img = np.load(f'''../resources/my_lungs.npy''')
    stabilized_y, _, _ = run_third_algorithm(img, MU)
    print(np.min(stabilized_y), np.mean(stabilized_y), np.max(stabilized_y))
    plt.imshow(stabilized_y, cmap=plt.cm.bone)
    plt.show()

    plt.imshow(img, cmap=plt.cm.bone)
    plt.show()
