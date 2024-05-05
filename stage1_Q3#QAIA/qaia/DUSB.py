from typing import List

import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix

from .SB import BSB

''' DU-SB in [arXiv:2306.16264] Deep Unfolded Simulated Bifurcation for Massive MIMO Signal Detection '''

# Eq. 15 ~ 16
σ = lambda x: 1 / (1 + np.exp(-x))
sw = lambda x: x * σ(x)
φ_s = lambda x, Λ=10: (1 / Λ) * (Λ * sw(x + 1) - Λ * sw(x - 1)) - 1
ψ_s = lambda x, A=100, B=1.01: σ(A * (np.abs(x) - B))


# for inference
class DUSB(BSB):

    def __init__(self, J:ndarray, h:ndarray, deltas:List[float], eta:float, x:ndarray=None, batch_size:int=100):
        super().__init__(J, h, x, len(deltas), batch_size, dt=-1, xi=None)

        # the pretrained optimal parameters :)
        self.Δ = deltas
        self.η = eta

        # Eq. 6 and 12
        # from essay
        #self.c_0: float = 2 * np.sqrt((self.N - 1) / csr_matrix.power(self.J, 2).sum())
        # from qaia lib
        self.c_0 = self.xi
        self.a = self.p

    def update(self):
        # Eq. 11 ~ 14, trainable parameters are $Δ_k$ (`dt`) and $η$
        for k, Δ_k in enumerate(self.Δ):
            self.y = self.y + Δ_k * (-(1 - self.a[k]) * self.x + self.η * self.c_0 * (self.J @ self.x + self.h))
            self.x = self.x + Δ_k * self.y
            self.x = φ_s(self.x)
            self.y = self.y * (1 - ψ_s(self.x))

    def update_hard(self):
        for k, Δ_k in enumerate(self.Δ):
            self.y = self.y + Δ_k * (-(1 - self.a[k]) * self.x + self.η * self.c_0 * (self.J @ self.x + self.h))
            self.x = self.x + Δ_k * self.y

            cond = np.abs(self.x) > 1
            self.x = np.where(cond, np.sign(self.x), self.x)          # limit x to vrng [-1, +1]
            self.y = np.where(cond, np.zeros_like(self.x), self.y)    # if |x|==1 is fully annealled, set y to zero
