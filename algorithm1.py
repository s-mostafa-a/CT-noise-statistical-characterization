import numpy as np
import math

from utils import central_gamma_pdf

'''<constants>'''
# length of pixels vector
N = 10
X = np.random.randint(low=-1000, high=400 + 1, size=N)
# min HU
delta = -1024
# Mu for 9 components
MU = {1: 340 - delta, 2: 240 - delta, 3: 100 - delta, 4: 0 - delta, 5: -160 - delta, 6: -370 - delta, 7: -540 - delta,
      8: -810 - delta, 9: -987 - delta}
J = len(MU)
MAX_ITER = 20
ERR = np.Infinity
# Tolerance
TOL = 0.1
'''</constants>'''
Y = X - delta
not_normalized_phi = np.random.random(J)
sum_of_phis = sum(not_normalized_phi)
theta = {'phi': [i / sum_of_phis for i in not_normalized_phi], 'alpha': list(range(1, J + 1)),
         'beta': list(range(1, J + 1))}
lamda = []
for i in range(N):
    vec = []
    form_for_bayes = []
    for j in range(J):
        form_for_bayes.append(theta['phi'][j] * central_gamma_pdf(Y[i], alpha=theta['alpha'][j], beta=theta['beta'][j]))
    form_for_bayes = np.array(form_for_bayes)
    lamda.append(form_for_bayes / np.sum(form_for_bayes))
lamda = np.array(lamda)
assert lamda.shape == (N, J)
n = 0
while ERR > TOL and n < MAX_ITER:
    n += 1
    alphas = []
    betas = []
    phis = []
    form_for_bayes = []
    for j in range(J):
        form_for_alpha = sum([lamda[i, j] * Y[i] / MU[j + 1] for i in range(N)]) - sum(
            [lamda[i, j] * math.log(Y[i] / MU[j + 1]) for i in range(N)])
        denom_for_alpha = sum([lamda[i, j] for i in range(N)])
        alpha = form_for_alpha / denom_for_alpha - 1
        beta = MU[j + 1] / alpha
        phi = 1 / n * denom_for_alpha
        form_for_bayes.append(np.array([phi * central_gamma_pdf(Y[i], alpha=alpha, beta=beta) for i in range(N)]))
        alphas.append(alpha)
        betas.append(beta)
        phis.append(phi)
    form_for_bayes = np.array(form_for_bayes).T
    lamda = form_for_bayes / np.sum(form_for_bayes, axis=1).reshape(-1, 1)
    t_p = np.array([np.array(theta['alpha']), np.array(theta['beta']), np.array(theta['phi'])])
    t_n = np.array([np.array(alphas), np.array(betas), np.array(phis)])
    ERR = np.linalg.norm(t_n - t_p) / np.linalg.norm(t_p)
    theta = {'phi': phis, 'alpha': alphas, 'beta': betas}
    print('n:', n)
    print('Err:', ERR)
