import numpy as np
from ct_charachterization import run_first_algorithm, _run_third_algorithm_without_fixed_neighborhood, \
    run_second_algorithm
import matplotlib.pyplot as plt


def test_first():
    MU = np.array([-1000, -700, -90, 50, 300])
    # length of pixels vector
    N_1 = 20
    np.random.seed(1)
    X_1 = np.random.randint(low=-1000, high=400 + 1, size=N_1)
    theta_1, gamma_1 = run_first_algorithm(y=X_1, mu=MU, non_central=True, neighborhood_size=10)
    print(theta_1.shape)


def test_second():
    MU = np.array([-1000, -700, -90, 50, 300])
    img = np.load(f'''./resources/2d_img.npy''')
    X_2 = img
    theta_2, gamma_2 = run_second_algorithm(y=X_2, mu=MU, non_central=True, neighborhood_size=32)
    print(theta_2.shape)


def test_third():
    mu_5 = np.array([-870])
    img = np.load(f'''./resources/luna_cropped.npy''')
    print(img.shape)
    img = img[0:128, 0:128]
    plt.imshow(img, cmap='gray')
    plt.show()
    plt.imshow(img[16:112, 16:112], cmap='gray')
    plt.show()
    stabilized_y = _run_third_algorithm_without_fixed_neighborhood(img, mu_5, non_central=True, constant_c=10,
                                                                   neighborhood_size=32)
    sy = stabilized_y[..., 0]
    print(sy.shape)
    plt.imshow(sy, cmap='gray')
    plt.show()
    print(np.min(sy) - 1030, np.mean(sy) - 1030, np.max(sy) - 1030)


if __name__ == '__main__':
    test_first()
    test_second()
    test_third()
