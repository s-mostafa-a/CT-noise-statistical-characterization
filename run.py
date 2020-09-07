import numpy as np
from algorithms.first import run as run_first_algorithm
from algorithms.second import run as run_second_algorithm

#################################
# first algorithm

# length of pixels vector
N_1 = 10
X_1 = np.random.randint(low=-1000, high=400 + 1, size=N_1)
theta_1, gamma_1 = run_first_algorithm(X_1, non_central=True)
print(theta_1)
#################################
# second algorithm

img = np.load(f'''./resources/2d_img_2.npy''')
# We know that img.shape is (364,364)
# so we set the neighborhood size to 28
neighborhood_size = 32
X_2 = img

theta_2, gamma_2 = run_second_algorithm(X_2, neighborhood_size, non_central=True)
print(theta_2)
