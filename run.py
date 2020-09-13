import numpy as np
from algorithms.first import run_first_algorithm
from algorithms.second import run_second_algorithm

DELTA = -1025

MU = [340 - DELTA, 240 - DELTA, 100 - DELTA, 0 - DELTA, -160 - DELTA, -370 - DELTA, -540 - DELTA,
      -810 - DELTA, -987 - DELTA]
#################################
# running the first algorithm
# length of pixels vector
N_1 = 10
X_1 = np.random.randint(low=-1000, high=400 + 1, size=N_1)
theta_1, gamma_1 = run_first_algorithm(y=X_1, centered_mu=MU, non_central=True)
print(theta_1)
#################################
# running the second algorithm
img = np.load(f'''./resources/2d_img.npy''')
X_2 = img
theta_2, gamma_2 = run_second_algorithm(y=X_2, centered_mu=MU, non_central=True, max_iter=3)
print(theta_2)
