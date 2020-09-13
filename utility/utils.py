import math

import numpy as np


def non_central_gamma_pdf(x, alpha, beta, delta):
    assert x >= delta, f'''x must be more-or-equal than delta. x: {x}, delta: {delta}'''
    y = x - delta
    return central_gamma_pdf(y=y, alpha=alpha, beta=beta)


def central_gamma_pdf(y, alpha, beta):
    assert alpha > 0 and beta > 0, f'''Alpha and Beta must be more than zero. Alpha: {alpha}, Beta: {beta}'''
    form = math.pow(y, (alpha - 1)) * math.exp(-y / beta)
    denominator = math.pow(beta, alpha) * math.gamma(alpha)
    return form / denominator


def form_of_equation_18(y, phi, alpha, beta):
    return phi * central_gamma_pdf(y, alpha=alpha, beta=beta)


def broadcast_3d_tile(matrix, h, w, d):
    m, n, o = matrix.shape[0] * h, matrix.shape[1] * w, matrix.shape[2] * d
    return np.broadcast_to(matrix.reshape(matrix.shape[0], 1, matrix.shape[1], 1, matrix.shape[2], 1),
                           (matrix.shape[0], h, matrix.shape[1], w, matrix.shape[2], d)).reshape(m, n, o)


def broadcast_2d_tile(matrix, h, w):
    m, n = matrix.shape[0] * h, matrix.shape[1] * w
    return np.broadcast_to(matrix.reshape(matrix.shape[0], 1, matrix.shape[1], 1),
                           (matrix.shape[0], h, matrix.shape[1], w)).reshape(m, n)


def equation_18_on_vector_of_j_elements(y_arr, mini_theta_arr):
    form_of_equation_18_vectorized = np.vectorize(form_of_equation_18)
    return form_of_equation_18_vectorized(y_arr, mini_theta_arr[0], mini_theta_arr[1], mini_theta_arr[2])


class ComputeThetaGammaBasedOn1DNeighborhood:
    def __init__(self, Y, gamma, mu):
        assert len(Y.shape) == 1, f'''The input array must be 1d'''
        self._Y = Y
        self._gamma = gamma
        self._J = gamma.shape[1]
        self.mini_alpha = np.ones((1, self._J))
        self.mini_beta = np.ones((1, self._J))
        self.mini_phi = np.ones((1, self._J))
        self._mu = mu
        self._new_theta = None

    def compute_for_neighbors(self):
        first_form_summation = np.zeros(self.mini_alpha.shape)
        second_form_summation = np.zeros(self.mini_alpha.shape)
        denominator_summation = np.zeros(self.mini_alpha.shape)
        for component in range(self._J):
            for i in range(self._Y.shape[0]):
                first_form_summation[0, component] += self._gamma[i, component] * self._Y[i] / self._mu[component]
                second_form_summation[0, component] += self._gamma[i, component] * math.log(
                    self._Y[i] / self._mu[component])
                denominator_summation[0, component] += self._gamma[i, component]
        self.mini_alpha = (first_form_summation - second_form_summation) / denominator_summation - 1
        self.mini_beta = np.array(self._mu) / self.mini_alpha
        self.mini_phi = denominator_summation

    def get_theta(self):
        theta = np.array([self.mini_phi, self.mini_alpha, self.mini_beta])
        self._new_theta = np.moveaxis(theta, [0, 1, 2], [1, 0, 2])
        return self._new_theta

    def get_gamma(self):
        gamma = np.zeros(shape=self._gamma.shape)
        for i, a in enumerate(self._Y):
            to_be_appended_on_gamma = equation_18_on_vector_of_j_elements(a, self._new_theta[0]).reshape(1, -1)
            gamma[i] = to_be_appended_on_gamma / np.sum(to_be_appended_on_gamma)
        return gamma


class ComputeThetaGammaBasedOn2DNeighborhood:
    def __init__(self, Y, gamma, mu):
        assert len(Y.shape) == 2, f'''The input image must be 2d'''
        self._Y = Y
        self._gamma = gamma
        self._J = gamma.shape[2]
        self._mu = mu
        self.mini_alpha = None
        self.mini_beta = None
        self.mini_phi = None
        self._new_theta = None

    def compute_for_neighbors(self):
        first_form_summation = np.sum(self._gamma * (np.expand_dims(self._Y, axis=-1) / self._mu), axis=(0, 1)).reshape(
            self._J)
        second_form_summation = np.sum(self._gamma * np.log(np.expand_dims(self._Y, axis=-1) / self._mu),
                                       axis=(0, 1)).reshape(self._J)
        denominator_summation = np.sum(self._gamma, axis=(0, 1)).reshape(self._J)
        self.mini_alpha = (first_form_summation - second_form_summation) / denominator_summation - 1
        self.mini_beta = np.array(self._mu) / self.mini_alpha
        self.mini_phi = denominator_summation

    def get_gamma_and_theta(self):
        new_theta = np.array([self.mini_phi, self.mini_alpha, self.mini_beta])
        gamma = np.zeros(shape=self._gamma.shape)
        for i, a in enumerate(self._Y):
            for j, b in enumerate(a):
                to_be_appended_on_gamma = equation_18_on_vector_of_j_elements(b, new_theta).reshape(1, -1)
                gamma[i, j] = to_be_appended_on_gamma / np.sum(to_be_appended_on_gamma)
        return gamma, new_theta
