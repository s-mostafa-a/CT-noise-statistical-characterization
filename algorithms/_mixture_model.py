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
    def _compute_next_theta(y, centered_mu, gamma):
        axis = tuple(range(len(y.shape)))
        size_of_j = len(centered_mu)
        first_form_summation = np.sum(gamma * (np.expand_dims(y, axis=-1) / centered_mu), axis=axis).reshape(size_of_j)
        second_form_summation = np.sum(gamma * np.log(np.expand_dims(y, axis=-1) / centered_mu), axis=axis).reshape(
            size_of_j)
        denominator_summation = np.sum(gamma, axis=axis).reshape(size_of_j)
        new_alpha = (first_form_summation - second_form_summation) / denominator_summation - 1
        new_beta = np.array(centered_mu) / new_alpha
        new_pi = denominator_summation / y.size
        new_theta = np.array([new_pi, new_alpha, new_beta])
        return new_theta

    def run(self, y, centered_mu, non_central=False, delta=-1030, max_iter=10, tol=0.01):
        size_of_j = len(centered_mu)

        # centering the data
        if non_central:
            y = y - delta

        # initial guess of parameters
        # we assume that theta[0] = pi, theta[1] = alpha, theta[2] = beta
        shape_of_theta = (3, size_of_j)
        theta = np.zeros(shape=shape_of_theta)
        theta[:, :] = np.array(
            [[1 / size_of_j] * size_of_j, [2] * size_of_j, [centered_mu[j] / 2 for j in range(size_of_j)]])

        # compute initial gamma
        shape_of_gamma = tuple(list(y.shape) + [size_of_j])
        gamma = self._compute_next_gamma(y, size_of_j, shape_of_gamma, theta)
        err = np.Infinity
        n = 0
        while err > tol and n < max_iter:
            n += 1
            new_theta = self._compute_next_theta(y, centered_mu, gamma)
            new_gamma = self._compute_next_gamma(y, size_of_j, shape_of_gamma, theta)
            err = np.linalg.norm(new_theta - theta) / np.linalg.norm(theta)
            theta = new_theta
            gamma = new_gamma
            print('iteration: ', n)
        return theta, gamma
