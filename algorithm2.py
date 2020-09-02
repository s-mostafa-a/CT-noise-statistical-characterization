import numpy as np
import math

from utils import central_gamma_pdf, CTScan

'''<constants>'''
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

img = np.load(f'''./sample/img.npy''')
Y = img - delta
shpe = list(Y.shape) + [3, J]
not_normalized_phi = np.random.random(J)
sum_of_phis = sum(not_normalized_phi)
theta = np.zeros(shape=tuple(shpe))
theta[:, :, :, :, :] = np.array(
    [[i / sum_of_phis for i in not_normalized_phi], list(range(1, J + 1)), list(range(1, J + 1))])
