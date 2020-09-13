import numpy as np
from algorithms.second import run_second_algorithm
import matplotlib.pyplot as plt
import time
from utility.utils import broadcast_3d_tile


def run_third_algorithm(x, mu, delta=-1030, max_iter=10, tol=0.01, C=10):
    J = len(mu)
    centered_mu = mu - delta
    y = x - delta
    t1 = time.time_ns()
    theta, gamma = run_second_algorithm(y, centered_mu=centered_mu, delta=delta, max_iter=max_iter, tol=tol)
    print(f'{-(t1 - time.time_ns()) / 1000000000} sec spent for algorithm2')
    # sclm: sample_conditioned_local_moment
    form_of_first_mini_sclm = np.sum(np.sqrt(np.sqrt(np.expand_dims(y, axis=-1)) * gamma) * gamma, axis=(0, 1)).reshape(
        (1, 1, J))
    form_of_second_mini_sclm = np.sum(np.sqrt(np.expand_dims(y, axis=-1) * gamma) * gamma, axis=(0, 1)).reshape(
        (1, 1, J))
    denominator_summation = np.sum(gamma, axis=(0, 1)).reshape((1, 1, J))
    first_mini_sclm = form_of_first_mini_sclm / denominator_summation
    second_mini_sclm = form_of_second_mini_sclm / denominator_summation
    first_sclm = np.sum(broadcast_3d_tile(first_mini_sclm, y.shape[0], y.shape[1], 1) * theta[0, :], axis=2)
    second_sclm = np.sum(broadcast_3d_tile(second_mini_sclm, y.shape[0], y.shape[1], 1) * theta[0, :], axis=2)
    var_of_radical_y = second_sclm - np.power(first_sclm, 2)
    stable_y = C * (np.sqrt(y) - first_sclm) / np.sqrt(var_of_radical_y) + second_sclm
    return stable_y


if __name__ == '__main__':
    MU = np.array([-1000, -870, -75, 0])
    # MU = np.array([340, 240, 100, 0, -160, -370, -540, -810, -987])
    img = np.load(f'''../resources/my_lungs.npy''')
    stabilized_y = run_third_algorithm(img, MU)
    plt.imshow(stabilized_y, cmap=plt.cm.bone)
    plt.show()
