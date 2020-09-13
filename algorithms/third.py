import numpy as np
import math
from algorithms.second import run as run_second_algorithm
import matplotlib.pyplot as plt
import time
# min HU
from utility.utils import broadcast_3d_tile

DELTA = -1030
# Mu for 9 components
# MU = [340 - DELTA, 240 - DELTA, 100 - DELTA, 0 - DELTA, -160 - DELTA, -370 - DELTA, -540 - DELTA,
#       -810 - DELTA, -987 - DELTA]
MU = [-1000 - DELTA, -870 - DELTA, -75 - DELTA, 0 - DELTA]
J = len(MU)

img = np.load(f'''../resources/my_lungs.npy''')

X = img
Y = X - DELTA
t1 = time.time_ns()
theta, gamma = run_second_algorithm(Y, mu=MU, max_iter=10)
print(f'{-(t1-time.time_ns())/1000000000} sec spent for algorithm2')
C = 10
# sclm: sample_conditioned_local_moment
form_of_first_mini_sclm = np.sum(np.sqrt(np.sqrt(np.expand_dims(Y, axis=-1)) * gamma) * gamma, axis=(0, 1)).reshape(
    (1, 1, J))
form_of_second_mini_sclm = np.sum(np.sqrt(np.expand_dims(Y, axis=-1) * gamma) * gamma, axis=(0, 1)).reshape((1, 1, J))
denominator_summation = np.sum(gamma, axis=(0, 1)).reshape((1, 1, J))
first_mini_sclm = form_of_first_mini_sclm / denominator_summation
second_mini_sclm = form_of_second_mini_sclm / denominator_summation
first_sclm = np.sum(broadcast_3d_tile(first_mini_sclm, Y.shape[0], Y.shape[1], 1) * theta[0, :], axis=2)
second_sclm = np.sum(broadcast_3d_tile(second_mini_sclm, Y.shape[0], Y.shape[1], 1) * theta[0, :], axis=2)
var_of_radical_y = second_sclm - np.power(first_sclm, 2)
stable_y = C * (np.sqrt(Y) - first_sclm) / np.sqrt(var_of_radical_y) + second_sclm

# np.save('../resources/stabled_my_lungs.npy', stable_y)
plt.imshow(stable_y, cmap=plt.cm.bone)
plt.show()

plt.imshow(Y, cmap=plt.cm.bone)
plt.show()
