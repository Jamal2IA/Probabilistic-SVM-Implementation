import numpy as np
from numpy import linalg

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=100.0):
   # print(-linalg.norm(x-y)**2)
    x=np.asarray(x)
    y=np.asarray(y)
    return np.exp((-linalg.norm(x-y)**2) / (2 * (sigma ** 2)))
