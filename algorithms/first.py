import numpy as np

from algorithms._mixture_model import NonCentralGammaMixtureModel
from algorithms.utility.utils import central_gamma_pdf


class FirstAlgorithm(NonCentralGammaMixtureModel):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _compute_next_gamma(y, big_jay, shape_of_gamma, theta):
        # Eq. 18
        new_gamma = np.zeros(shape=shape_of_gamma)
        for i, a in enumerate(y):
            sum_of_j_elements = 0
            for j in range(big_jay):
                val = theta[0, j] * central_gamma_pdf(y[i], alpha=theta[1, j], beta=theta[2, j])
                new_gamma[i, j] = val
                sum_of_j_elements += val
            new_gamma[i] = new_gamma[i] / sum_of_j_elements
        return new_gamma


run_first_algorithm = FirstAlgorithm().run
