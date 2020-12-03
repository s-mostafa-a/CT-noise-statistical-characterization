import numpy as onp
from ct_charachterization import run_first_algorithm, run_second_algorithm, run_third_algorithm_gamma_instead_of_pi
import matplotlib.pyplot as plt


def test_first():
    mu = onp.array([-1000, -700, -90, 50, 300])
    # length of pixels vector
    n_1 = 20
    onp.random.seed(1)
    x_1 = onp.random.randint(low=-1000, high=400 + 1, size=n_1)
    theta_1, gamma_1 = run_first_algorithm(y=x_1, mu=mu, non_central=True, neighborhood_size=10)
    print(theta_1.shape)


def test_second():
    mu = onp.array([-1000, -700, -90, 50, 300])
    img = onp.load(f'''./resources/2d_img.npy''')
    x_2 = img
    theta_2, gamma_2 = run_second_algorithm(y=x_2, mu=mu, non_central=True, neighborhood_size=32)
    print(theta_2.shape)


def test_third():
    mu = onp.array([-987, -880, -540, -370, -160, 0, 100, 240, 340])
    img = onp.load(f'''./resources/luna_cropped.npy''')
    print(img.shape)
    img = img[0:128, 0:128]
    plt.imshow(img, cmap='gray')
    plt.show()
    stabilized_y = run_third_algorithm_gamma_instead_of_pi(img, mu, non_central=True, constant_c=10, neighborhood_size=32)
    sy = stabilized_y
    print(sy.shape)
    plt.imshow(sy, cmap='gray')
    plt.show()
    print(onp.min(sy) - 1030, onp.mean(sy) - 1030, onp.max(sy) - 1030)


if __name__ == '__main__':
    test_first()
    test_second()
    test_third()
