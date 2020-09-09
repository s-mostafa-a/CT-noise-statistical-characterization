import numpy as np
from algorithms.first import run as run_first_algorithm
from algorithms.second import run as run_second_algorithm

DELTA = -1025

MU = [340 - DELTA, 240 - DELTA, 100 - DELTA, 0 - DELTA, -160 - DELTA, -370 - DELTA, -540 - DELTA,
      -810 - DELTA, -987 - DELTA]
#################################
# first algorithm

# length of pixels vector
N_1 = 10
X_1 = np.random.randint(low=-1000, high=400 + 1, size=N_1)
theta_1, gamma_1 = run_first_algorithm(y=X_1, mu=MU, non_central=True)
#################################
# second algorithm

img = np.load(f'''./resources/2d_img_2.npy''')
# We know that img.shape is (512,512)
# so we set the neighborhood size to 32
neighborhood_size = 32
X_2 = img

theta_2, gamma_2 = run_second_algorithm(y=X_2, mu=MU, non_central=True)
print(theta_2)
