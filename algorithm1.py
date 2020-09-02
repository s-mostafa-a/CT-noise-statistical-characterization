import numpy as np

'''constants'''
# length of pixels vector
N = 10
X = np.random.randint(low=-1000, high=400 + 1, size=N)
# min HU
delta = -1024
# Mu for 9 components
MU = {1: 340, 2: 240, 3: 100, 4: 0, 5: -160, 6: -370, 7: -540, 8: -810, 9: -987}
J = len(MU)
MAX_ITER = 20
ERR = np.Infinity
# Tolerance
TOL = 0.1
'''constants'''

Y = X - delta

print(type(MU), J)
