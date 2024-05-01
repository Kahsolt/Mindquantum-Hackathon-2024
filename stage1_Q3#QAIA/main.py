from typing import *

import numpy as np
from numpy import ndarray

from qaia import NMFA, SimCIM, CAC, CFC, SFC, ASB, BSB, DSB, LQA


def to_ising(H:ndarray, y:ndarray, num_bits_per_symbol:int) -> Tuple[ndarray, ndarray]:
    '''
    Reduce MIMO detection problem into Ising problem.

    Reference
    ---------
    [1] Singh A K, Jamieson K, McMahon P L, et al. Ising machines’ dynamics and regularization for near-optimal mimo detection. 
        IEEE Transactions on Wireless Communications, 2022, 21(12): 11080-11094.

    Input
    -----
    H: [Nr, Nt], np.complex
        Channel matrix

    y: [Nr, 1], np.complex
        Received signal

    num_bits_per_symbol: int
        The number of bits per constellation symbol, e.g., 4 for QAM16.
    
    
    Output
    ------
    J: [2*Nt, 2*Nt], np.float
        The coupling matrix of Ising problem
    
    h: [2*Nt, 1], np.float
        The external field
        
    '''
    # the size of constellation
    M = 2**num_bits_per_symbol
    Nr, Nt = H.shape
    N = 2 * Nt
    rb = int(num_bits_per_symbol/2)
    qam_var = 1 / (2**(rb - 2)) * np.sum(np.linspace(1, 2**rb - 1, 2**(rb - 1))**2)
    I = np.eye(N)
    T = (2**(rb - 1 - np.arange(rb)))[:, np.newaxis, np.newaxis] * I[np.newaxis, ...]
    T = T.reshape(-1, N).T
    Nr, Nt = H.shape
    H_real = H.real
    H_imag = H.imag
    H_tilde = np.vstack([np.hstack([H_real, -H_imag]), np.hstack([H_imag, H_real])])
    y_tilde = np.concatenate([y.real, y.imag])
    # This is different from the original paper because we use normalized transmitted symbol
    z = y_tilde / np.sqrt(qam_var) - H_tilde @ T @ np.ones((N * rb, 1)) / qam_var + (np.sqrt(M) - 1) * H_tilde @ np.ones((N, 1)) / qam_var
    J = -2 * T.T @ H_tilde.T @ H_tilde @ T / qam_var
    diag_index = np.diag_indices_from(J)
    J[diag_index] = 0
    h = 2 * z.T @ H_tilde @ T
    # [Nr*nbps, Nr*nbps], [Nr*nbps, 1]
    return J, h.T


# 选手提供的Ising模型生成函数，可以用我们提供的to_ising
def ising_generator(H:ndarray, y:ndarray, num_bits_per_symbol:int, snr:float) -> Tuple[ndarray, ndarray]:
    return to_ising(H, y, num_bits_per_symbol)


# 选手提供的qaia MLD求解器，用mindquantum.algorithms.qaia
def qaia_mld_solver(J:ndarray, h:ndarray) -> ndarray:
    solver = LQA(J, h, batch_size=300, n_iter=100)
    solver.update()
    sample = np.sign(solver.x)      # [N=256, B]
    energy = solver.calc_energy()   # [1, B]
    opt_index = np.argmin(energy)
    solution = sample[:, opt_index] # [N=256], vset {-1, 1}
    return solution
