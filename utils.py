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


def broadcast_tile(matrix, h, w, d):
    m, n, o = matrix.shape[0] * h, matrix.shape[1] * w, matrix.shape[2] * d
    return np.broadcast_to(matrix.reshape(matrix.shape[0], 1, matrix.shape[1], 1, matrix.shape[2], 1),
                           (matrix.shape[0], h, matrix.shape[1], w, matrix.shape[2], d)).reshape(m, n, o)


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


class ComputeBasedOnNeighborhood:
    def __init__(self, Y, gamma, mu, neighborhood_size):
        assert len(Y.shape) == 2
        assert Y.shape[0] % neighborhood_size == 0
        assert Y.shape[1] % neighborhood_size == 0
        self._Y = Y
        self._gamma = gamma
        self._neighborhood_size = neighborhood_size
        self._J = gamma.shape[2]
        self.mini_alpha = np.ones((Y.shape[0] // neighborhood_size, Y.shape[0] // neighborhood_size, self._J))
        self.mini_beta = np.ones((Y.shape[0] // neighborhood_size, Y.shape[0] // neighborhood_size, self._J))
        self.mini_phi = np.ones((Y.shape[0] // neighborhood_size, Y.shape[0] // neighborhood_size, self._J))
        self._mu = mu
        self._new_theta = None

    def compute_for_neighbors(self):
        first_form_summation = np.zeros(self.mini_alpha.shape)
        second_form_summation = np.zeros(self.mini_alpha.shape)
        denominator_summation = np.zeros(self.mini_alpha.shape)
        for component in range(self._J):
            for i in range(self._Y.shape[0]):
                for j in range(self._Y.shape[1]):
                    first_form_summation[i // self._neighborhood_size, j // self._neighborhood_size, component] += \
                        self._gamma[i, j, component] * self._Y[i, j] / self._mu[component]
                    second_form_summation[i // self._neighborhood_size, j // self._neighborhood_size, component] += \
                        self._gamma[i, j, component] * math.log(self._Y[i, j] / self._mu[component])
                    denominator_summation[i // self._neighborhood_size, j // self._neighborhood_size, component] += \
                        self._gamma[i, j, component]
        self.mini_alpha = (first_form_summation - second_form_summation) / denominator_summation - 1
        self.mini_beta = np.array(self._mu) / self.mini_alpha
        self.mini_phi = denominator_summation

    def get_theta(self):
        phi = broadcast_tile(self.mini_phi, self._neighborhood_size, self._neighborhood_size, 1)
        alpha = broadcast_tile(self.mini_alpha, self._neighborhood_size, self._neighborhood_size, 1)
        beta = broadcast_tile(self.mini_beta, self._neighborhood_size, self._neighborhood_size, 1)
        theta = np.array([phi, alpha, beta])
        self._new_theta = np.moveaxis(theta, [0, 1, 2, 3], [2, 0, 1, 3])
        return self._new_theta

    def get_gamma(self):
        gamma = np.zeros(shape=self._gamma.shape)
        for i, a in enumerate(self._Y):
            for j, b in enumerate(a):
                to_be_appended_on_gamma = equation_18_on_vector_of_j_elements(b, self._new_theta[i, j]).reshape(1, -1)
                gamma[i, j] = to_be_appended_on_gamma / np.sum(to_be_appended_on_gamma)
        return gamma
