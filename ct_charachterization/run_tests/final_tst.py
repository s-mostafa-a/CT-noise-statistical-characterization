import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.special import digamma
from glob import glob
import pydicom as dicom
import dicom_numpy
import SimpleITK as sitk
from matplotlib import pyplot as plt
from scipy.stats import mode
import matplotlib.patches as patches
from copy import deepcopy

from ct_charachterization import run_third_algorithm_expectation_at_the_beginning, \
    run_third_algorithm_expectation_at_the_end, run_third_algorithm_gamma_instead_of_pi

mu_9 = np.array([-987, -810, -540, -370, -160, 0, 100, 240, 340])
luna = np.load(f'../../resources/luna_cropped.npy')
# plt.imshow(luna, cmap='gray')
# plt.title("the original image")
# plt.show()

# y = luna[32:96, 32:96]
y = luna[96:160, 96:160]
plt.imshow(y, cmap='gray')
plt.title("the original image")
plt.show()

neighborhood_size = 8

resul = run_third_algorithm_expectation_at_the_beginning(y, mu_9, non_central=True, constant_c=10,
                                                         neighborhood_size=neighborhood_size, max_iter=5)
a = resul.shape[0]
b = resul.shape[1]
hn = int(neighborhood_size / 2)
plt.imshow(resul[hn:a - hn, hn:b - hn], cmap='gray')
plt.show()
resul = run_third_algorithm_expectation_at_the_end(y, mu_9, non_central=True, constant_c=10,
                                                   neighborhood_size=neighborhood_size, max_iter=5)
plt.imshow(resul[hn:a - hn, hn:b - hn], cmap='gray')
plt.show()
resul = run_third_algorithm_gamma_instead_of_pi(y, mu_9, non_central=True, constant_c=10,
                                                neighborhood_size=neighborhood_size, max_iter=5)
plt.imshow(resul[hn:a - hn, hn:b - hn], cmap='gray')
plt.show()

# flat = resul.flatten() - 1030
# ax = plt.subplot(1, 1, 1)
# ax.hist(flat, bins=list(np.arange(-1100, 500, 1)))
# plt.title("histogram")
# plt.show()
# print(f'min: {np.min(flat)}, mean: {np.mean(flat)}, max: {np.max(flat)}')
