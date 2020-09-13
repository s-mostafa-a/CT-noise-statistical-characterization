from abc import abstractmethod
import numpy as np


class NonCentralGammaMixtureModel(object):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def _compute_next_gamma(y, size_of_j, shape_of_gamma, theta):
        pass

    @staticmethod
    @abstractmethod
    def _compute_next_theta(y, mu, gamma):
        pass

    def run(self, y, mu, non_central=False, delta=-1030, max_iter=10, tol=0.01):
        size_of_j = len(mu)

        # centering the data
        if non_central:
            y = y - delta

        # initial guess of parameters
        # we assume that theta[0] = phi, theta[1] = alpha, theta[2] = beta
        shape_of_theta = (3, size_of_j)
        theta = np.zeros(shape=shape_of_theta)
        theta[:, :] = np.array(
            [[1 / size_of_j] * size_of_j, [2] * size_of_j, [mu[j] / 2 for j in range(size_of_j)]])

        # compute initial gamma
        shape_of_gamma = tuple(list(y.shape) + [size_of_j])
        gamma = self._compute_next_gamma(y, size_of_j, shape_of_gamma, theta)
        err = np.Infinity
        n = 0
        while err > tol and n < max_iter:
            n += 1
            new_theta = self._compute_next_theta(y, mu, gamma)
            new_gamma = self._compute_next_gamma(y, size_of_j, shape_of_gamma, theta)
            err = np.linalg.norm(new_theta - theta) / np.linalg.norm(theta)
            theta = new_theta
            gamma = new_gamma
            print(n)
        return theta, gamma
