import numpy as np
from algorithms.first import run as run_first_algorithm
from algorithms.second import run as run_second_algorithm

#################################
# first algorithm

# length of pixels vector
N_1 = 10
X_1 = np.random.randint(low=-1000, high=400 + 1, size=N_1)
theta_1 = run_first_algorithm(X_1)
print(theta_1)
#################################
# second algorithm

img = np.load(f'''./sample/img.npy''')
# We know that img.shape is (280, 364,364)
# so we set the neighborhood size to 28
neighborhood_size = 28
X_2 = img[140, :, :]

theta_2 = run_second_algorithm(X_2, neighborhood_size)
print(theta_2)
