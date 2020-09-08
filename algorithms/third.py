import numpy as np
import math
from algorithms.second import run as run_second_algorithm
import matplotlib.pyplot as plt

# min HU
from utility.utils import broadcast_tile

DELTA_3 = -1025
# Mu for 9 components
MU_3 = [340 - DELTA_3, 240 - DELTA_3, 100 - DELTA_3, 0 - DELTA_3, -160 - DELTA_3, -370 - DELTA_3, -540 - DELTA_3,
        -810 - DELTA_3, -987 - DELTA_3]
# MU_3 = [-1000 - DELTA_3, -870 - DELTA_3, -75 - DELTA_3, 0 - DELTA_3]
J_3 = len(MU_3)
NEIGHBORHOOD_SIZE = 512

img = np.load(f'''../resources/my_lungs.npy''')
# We know that img.shape is (364,364)
# so we set the neighborhood size to 28
X_3 = img
Y_3 = X_3 - DELTA_3
theta, gamma = run_second_algorithm(Y_3, NEIGHBORHOOD_SIZE)
C = 10
# sclm: sample_condition_local_moment
form_of_first_mini_sclm = np.ones((Y_3.shape[0] // NEIGHBORHOOD_SIZE, Y_3.shape[0] // NEIGHBORHOOD_SIZE, J_3))
form_of_second_mini_sclm = np.ones((Y_3.shape[0] // NEIGHBORHOOD_SIZE, Y_3.shape[0] // NEIGHBORHOOD_SIZE, J_3))
denominator_summation = np.ones((Y_3.shape[0] // NEIGHBORHOOD_SIZE, Y_3.shape[0] // NEIGHBORHOOD_SIZE, J_3))
for component in range(J_3):
    for i in range(Y_3.shape[0]):
        for j in range(Y_3.shape[1]):
            form_of_first_mini_sclm[i // NEIGHBORHOOD_SIZE, j // NEIGHBORHOOD_SIZE, component] += \
                math.sqrt(Y_3[i, j]) * gamma[i, j, component]
            form_of_second_mini_sclm[i // NEIGHBORHOOD_SIZE, j // NEIGHBORHOOD_SIZE, component] += \
                Y_3[i, j] * gamma[i, j, component]
            denominator_summation[i // NEIGHBORHOOD_SIZE, j // NEIGHBORHOOD_SIZE, component] += \
                gamma[i, j, component]
first_mini_sclm = form_of_first_mini_sclm / denominator_summation
second_mini_sclm = form_of_second_mini_sclm / denominator_summation
theta[:, :, 0, :] = theta[:, :, 0, :] / np.sum(theta[:, :, 0, :], axis=2).reshape((theta.shape[0], theta.shape[1], 1))
first_sclm = np.sum(broadcast_tile(first_mini_sclm, NEIGHBORHOOD_SIZE, NEIGHBORHOOD_SIZE, 1) * theta[:, :, 0, :],
                    axis=2)
second_sclm = np.sum(broadcast_tile(second_mini_sclm, NEIGHBORHOOD_SIZE, NEIGHBORHOOD_SIZE, 1) * theta[:, :, 0, :],
                     axis=2)
var_of_radical_y = second_sclm - np.power(first_sclm, 2)
stable_y = C * (np.sqrt(Y_3) - first_sclm) / np.sqrt(var_of_radical_y) + second_sclm

np.save('../resources/stable_my_lungs.npy', stable_y)
plt.imshow(stable_y, cmap=plt.cm.bone)
plt.show()

plt.imshow(Y_3, cmap=plt.cm.bone)
plt.show()
