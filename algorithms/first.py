import numpy as np
import math

from resources.utils import central_gamma_pdf

# min HU
DELTA_1 = -1024
# Mu for 9 components
MU_1 = {1: 340 - DELTA_1, 2: 240 - DELTA_1, 3: 100 - DELTA_1, 4: 0 - DELTA_1, 5: -160 - DELTA_1, 6: -370 - DELTA_1,
        7: -540 - DELTA_1,
        8: -810 - DELTA_1, 9: -987 - DELTA_1}
J_1 = len(MU_1)
MAX_ITER_1 = 20
# Tolerance
TOL_1 = 0.1


def run(y, non_central=False):
    number_of_rvs = len(y)
    if non_central:
        y = y - DELTA_1
    not_normalized_phi = np.random.random(J_1)
    sum_of_phis = sum(not_normalized_phi)
    theta = {'phi': [i / sum_of_phis for i in not_normalized_phi], 'alpha': list(range(1, J_1 + 1)),
             'beta': list(range(1, J_1 + 1))}
    gamma = []
    for i in range(number_of_rvs):
        form_for_bayes = []
        for j in range(J_1):
            form_for_bayes.append(
                theta['phi'][j] * central_gamma_pdf(y[i], alpha=theta['alpha'][j], beta=theta['beta'][j]))
        form_for_bayes = np.array(form_for_bayes)
        gamma.append(form_for_bayes / np.sum(form_for_bayes))
    gamma = np.array(gamma)
    assert gamma.shape == (number_of_rvs, J_1)
    n = 0
    err_1 = np.Infinity
    while err_1 > TOL_1 and n < MAX_ITER_1:
        n += 1
        alphas = []
        betas = []
        phis = []
        form_for_bayes = []
        for j in range(J_1):
            form_for_alpha = sum([gamma[i, j] * y[i] / MU_1[j + 1] for i in range(number_of_rvs)]) - sum(
                [gamma[i, j] * math.log(y[i] / MU_1[j + 1]) for i in range(number_of_rvs)])
            denom_for_alpha = sum([gamma[i, j] for i in range(number_of_rvs)])
            alpha = form_for_alpha / denom_for_alpha - 1
            beta = MU_1[j + 1] / alpha
            phi = 1 / n * denom_for_alpha
            form_for_bayes.append(
                np.array([phi * central_gamma_pdf(y[i], alpha=alpha, beta=beta) for i in range(number_of_rvs)]))
            alphas.append(alpha)
            betas.append(beta)
            phis.append(phi)
        form_for_bayes = np.array(form_for_bayes).T
        gamma = form_for_bayes / np.sum(form_for_bayes, axis=1).reshape(-1, 1)
        t_p = np.array([np.array(theta['alpha']), np.array(theta['beta']), np.array(theta['phi'])])
        t_n = np.array([np.array(alphas), np.array(betas), np.array(phis)])
        err_1 = np.linalg.norm(t_n - t_p) / np.linalg.norm(t_p)
        theta = {'phi': phis, 'alpha': alphas, 'beta': betas}
    return theta, gamma
