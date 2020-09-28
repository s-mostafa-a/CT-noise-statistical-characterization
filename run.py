import numpy as np
from ct_charachterization import run_first_algorithms, run_third_algorithm_without_fixed_neighborhood
import matplotlib.pyplot as plt

# MU = np.array([-1000, -700, -90, 50, 300])
# #################################
# # running the first algorithm
# # length of pixels vector
# N_1 = 20
# np.random.seed(1)
# X_1 = np.random.randint(low=-1000, high=400 + 1, size=N_1)
# theta_1, gamma_1 = run_first_algorithms(y=X_1, mu=MU, non_central=True, neighborhood_size=10)
# print(theta_1.shape)
# #################################
# # running the second algorithm
# img = np.load(f'''./resources/2d_img.npy''')
# X_2 = img
# theta_2, gamma_2 = run_first_algorithms(y=X_2, mu=MU, non_central=True, neighborhood_size=32)
# print(theta_2.shape)
mu_5 = np.array([-870])
img = np.load(f'''./resources/luna_cropped.npy''')
print(img.shape)
img = img[0:128, 0:128]
plt.imshow(img, cmap='gray')
plt.show()
plt.imshow(img[32:96, 32:96], cmap='gray')
plt.show()
stabilized_y = run_third_algorithm_without_fixed_neighborhood(img, mu_5, non_central=True, constant_c=10,
                                                              neighborhood_size=32)
sy = stabilized_y[..., 0]
print(sy.shape)
plt.imshow(sy, cmap='gray')
plt.show()
print(np.min(sy) - 1030, np.mean(sy) - 1030, np.max(sy) - 1030)

