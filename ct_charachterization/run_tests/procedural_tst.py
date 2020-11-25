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

from ct_charachterization import run_first_log_algorithm
from ct_charachterization.utility.utils import central_gamma_log_pdf

mu = np.array([-987, -810, -540, -370, -160, 0, 100, 240, 340])
luna = np.load(f'../../resources/luna_cropped.npy')
# plt.imshow(luna, cmap='gray')
# plt.title("the original image")
# plt.show()


y_1 = luna[32:96, 32:96]
plt.imshow(y_1, cmap='gray')
plt.title("the original image")
plt.show()
delta = np.min(y_1) - 1
# neighborhood_size = 8
mu = mu - delta

y_1 = y_1 - delta

global_theta, global_gamma = run_first_log_algorithm(y_1, mu, 0, delta, max_iter=10, tol=0, non_central=False)
global_alpha = global_theta[1, ...]
global_beta = global_theta[2, ...]
for j in range(len(mu)):
    wanted_alpha = global_alpha[j, ...]
    wanted_beta = global_beta[j, ...]
    xs = np.arange(delta + 1, 500, 1) - delta
    ys = 2500 * np.exp(central_gamma_log_pdf(xs, wanted_alpha, wanted_beta).ravel())
    plt.plot(xs + delta, ys, '-')
flat1 = deepcopy(y_1).flatten() + delta
plt.hist(flat1, bins=list(np.arange(-1100, 500, 1)), label='stabilized')
plt.legend(loc='upper right')
plt.show()

y_2 = luna[96:160, 96:160]
delta = np.min(y_2) - 10
# neighborhood_size = 8
mu = mu - delta
plt.imshow(y_2, cmap='gray')
plt.title("the original image")
plt.show()
y_2 = y_2 - delta
global_theta, global_gamma = run_first_log_algorithm(y_2, mu, 0, delta, max_iter=10, tol=0, non_central=False)
global_alpha = global_theta[1, ...]
global_beta = global_theta[2, ...]
for j in range(len(mu)):
    wanted_alpha = global_alpha[j, ...]
    wanted_beta = global_beta[j, ...]
    xs = np.arange(delta + 1, 500, 1) - delta
    ys = 2500 * np.exp(central_gamma_log_pdf(xs, wanted_alpha, wanted_beta).ravel())
    plt.plot(xs + delta, ys, '-')
flat1 = deepcopy(y_2).flatten() + delta
plt.hist(flat1, bins=list(np.arange(-1100, 500, 1)), label='stabilized')
plt.legend(loc='upper right')
plt.show()
