import numpy as np
from numpy import ndarray

from .SB import DSB


class DSB_lmmse(DSB):

    ''' Eq. 10 from arXiv:2105.10535 Ising Machines’ Dynamics and Regularization for Near-Optimal Large and Massive MIMO Detection '''
    ''' LMMSE-guigded SB in [arXiv:2210.14660] Simulated Bifurcation Algorithm for MIMO Detection '''

    def __init__(self, J, h=None, x=None, n_iter=1000, batch_size=1, dt=1, xi=None):
        super().__init__(J, h, x, n_iter, batch_size, dt, xi)

        self.initialize()

    def update(self, x_p:ndarray, r:float=0.15):
        # TODO: 如何加上正则项 $ H = s^T J s + h s T + r ||s - s_p||^2 $
        # x_p is the sub-optimal solution from LMMSE
        for i in range(self.n_iter):
            if self.h is None:
                self.y += (-(self.delta - self.p[i]) * self.x + self.xi * self.J.dot(np.sign(self.x))) * self.dt
            else:
                self.y += (-(self.delta - self.p[i]) * self.x + self.xi * (self.J.dot(np.sign(self.x)) + self.h)) * self.dt
            self.x += self.dt * self.y * self.delta
            cond = np.abs(self.x) > 1
            self.x = np.where(cond, np.sign(self.x), self.x)
            self.y = np.where(cond, np.zeros_like(self.y), self.y)
