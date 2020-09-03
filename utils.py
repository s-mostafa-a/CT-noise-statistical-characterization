import math

from scipy import ndimage
import numpy as np
import SimpleITK as sitk


def non_central_gamma_pdf(x, alpha, beta, delta):
    assert x >= delta
    y = x - delta
    return central_gamma_pdf(y=y, alpha=alpha, beta=beta)


def central_gamma_pdf(y, alpha, beta):
    assert alpha > 0 and beta > 0
    form = math.pow(y, (alpha - 1)) * math.exp(-y / beta)
    denominator = math.pow(beta, alpha) * math.gamma(alpha)
    return form / denominator


def form_of_equation_18(y, phi, alpha, beta):
    return phi * central_gamma_pdf(y, alpha=alpha, beta=beta)


def broadcast_tile(matrix, h, w):
    m, n = matrix.shape[0] * h, matrix.shape[1] * w
    return np.broadcast_to(matrix.reshape(matrix.shape[0], 1, matrix.shape[1], 1),
                           (matrix.shape[0], h, matrix.shape[1], w)).reshape(m, n)


def equation_18_on_vector_of_j_elements(y_arr, mini_theta_arr):
    form_of_equation_18_vectorized = np.vectorize(form_of_equation_18)
    return form_of_equation_18_vectorized(y_arr, mini_theta_arr[0], mini_theta_arr[1], mini_theta_arr[2])


class CTScan(object):
    def __init__(self, path):
        path = path
        self._ds = sitk.ReadImage(path)
        self._spacing = np.array(list(reversed(self._ds.GetSpacing())))
        self._origin = np.array(list(reversed(self._ds.GetOrigin())))
        self._image = sitk.GetArrayFromImage(self._ds)

    def preprocess(self):
        self._resample()
        self._normalize()

    def get_image(self):
        return self._image

    def _resample(self):
        spacing = np.array(self._spacing, dtype=np.float32)
        new_spacing = [1, 1, 1]
        imgs = self._image
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = ndimage.interpolation.zoom(imgs, resize_factor, mode='nearest')
        self._image = imgs
        self._spacing = true_spacing

    def _normalize(self):
        MIN_BOUND = -1000
        MAX_BOUND = 400.
        self._image[self._image > MAX_BOUND] = MAX_BOUND
        self._image[self._image < MIN_BOUND] = MIN_BOUND


class Neighborhood:
    def __init__(self, Y, gamma, neighborhood_size):
        self._Y = Y
        self._gamma = gamma
        self._neighborhood_size = neighborhood_size

    def compute_for_neighbors(self):
        pass

    def get_matrix(self):
        pass
