import numpy as np
from typing import Callable

def softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    '''
    Computes the softmax distribution along the axis dimension of x.
    '''
    z = np.exp(x - x.max(axis=axis))
    return z/z.sum(axis=axis, keepdims=True)

class LaplaceProximalPoint:
    '''
    Implements the Laplace Proximal Point optimization algorithm, as specified in:

    Tibshirani, R. J., Fung, S. W., Heaton, H., & Osher, S. (2024). 
    Laplace Meets Moreau: Smooth Approximation to Infimal Convolutions Using Laplace's Method. 
    arXiv preprint arXiv:2406.02003.

    '''
    def __init__(self, f: Callable[[np.ndarray], float], x0: np.ndarray, lambd: float, delta: float, num_samples: int = 50):
        '''
        Inputs:
        x0 is the initial state and of dimension (n,) 
        lambd is the proximal operator weighting
        delta is the softmax temperature
        num_samples specifies the number of Gaussian samples 
        '''
        self.f = f
        self.x = x0
        self.lambd = lambd
        self.delta = delta
        self.num_samples = num_samples

    def step(self):
        y = np.sqrt(self.delta * self.lambd) * np.random.randn(self.num_samples, self.x.shape[0]) + self.x #(N,n)
        self.x = softmax(-np.apply_along_axis(self.f,-1,y)/self.delta) @ y
        return self.x