import numpy as np

from algorithms._mixture_model import NonCentralGammaMixtureModel
from utility.utils import central_gamma_pdf


class SecondAlgorithm(NonCentralGammaMixtureModel):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _compute_next_gamma(y, size_of_j, shape_of_gamma, theta):
        # Eq. 18
        new_gamma = np.zeros(shape=shape_of_gamma)
        for i, a in enumerate(y):
            for j, b in enumerate(a):
                sum_of_j_elements = 0
                for k in range(size_of_j):
                    val = theta[0, k] * central_gamma_pdf(y[i, j], alpha=theta[1, k], beta=theta[2, k])
                    new_gamma[i, j, k] = val
                    sum_of_j_elements += val
                new_gamma[i, j] = new_gamma[i, j] / sum_of_j_elements
        return new_gamma

    @staticmethod
    def _compute_next_theta(y, mu, gamma):
        size_of_j = len(mu)
        first_form_summation = np.sum(gamma * (np.expand_dims(y, axis=-1) / mu), axis=(0, 1)).reshape(size_of_j)
        second_form_summation = np.sum(gamma * np.log(np.expand_dims(y, axis=-1) / mu), axis=(0, 1)).reshape(size_of_j)
        denominator_summation = np.sum(gamma, axis=(0, 1)).reshape(size_of_j)
        new_alpha = (first_form_summation - second_form_summation) / denominator_summation - 1
        new_beta = np.array(mu) / new_alpha
        new_phi = denominator_summation / y.size
        new_theta = np.array([new_phi, new_alpha, new_beta])
        return new_theta


run_second_algorithm = SecondAlgorithm().run
