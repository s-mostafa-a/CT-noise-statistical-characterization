import numpy as np
from ct_charachterization import run_first_algorithms

MU = np.array([-1000, -700, -90, 50, 300])
#################################
# running the first algorithm
# length of pixels vector
N_1 = 20
np.random.seed(1)
X_1 = np.random.randint(low=-1000, high=400 + 1, size=N_1)
theta_1, gamma_1 = run_first_algorithms(y=X_1, mu=MU, non_central=True, neighborhood_size=10)
print(theta_1.shape)
#################################
# running the second algorithm
img = np.load(f'''./resources/2d_img.npy''')
X_2 = img
theta_2, gamma_2 = run_first_algorithms(y=X_2, mu=MU, non_central=True, neighborhood_size=32)
print(theta_2.shape)
