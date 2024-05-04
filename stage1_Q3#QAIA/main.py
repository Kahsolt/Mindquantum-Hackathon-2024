from typing import *

import numpy as np
from numpy import ndarray

from qaia import QAIA, NMFA, SimCIM, CAC, CFC, SFC, ASB, BSB, DSB, LQA


def to_ising(H:ndarray, y:ndarray, nbps:int) -> Tuple[ndarray, ndarray]:
    '''
    Reduce MIMO detection problem into Ising problem.

    Reference
    ---------
    [1] Singh A K, Jamieson K, McMahon P L, et al. Ising machines’ dynamics and regularization for near-optimal mimo detection. 
        IEEE Transactions on Wireless Communications, 2022, 21(12): 11080-11094.
    [2] Ising Machines’ Dynamics and Regularization for Near-Optimal Large and Massive MIMO Detection. arXiv: 2105.10535v3

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
    J: [rb*2*Nt, rb*2*Nt], np.float
        The coupling matrix of Ising problem
    h: [rb*2*Nt, 1], np.float
        The external field
    '''

    # the size of constellation, the M-QAM where M in {16, 64, 256}
    M = 2**nbps
    # n_elem at TX side (c=2 for real/imag, 1 symbol = 2 elem)
    Nr, Nt = H.shape
    N = 2 * Nt
    # n_bits/n_spins that one elem decodes to
    rb = nbps // 2

    # QAM variance for normalization
    # ref: https://dsplog.com/2007/09/23/scaling-factor-in-qam/
    #qam_var: float = 1 / (2**(rb - 2)) * np.sum(np.linspace(1, 2**rb - 1, 2**(rb - 1))**2)
    qam_var = 2 * (M - 1) / 3

    # Eq. 7 the transform matrix T
    I = np.eye(N)
    # [rb, N, N]
    T = (2**(rb - 1 - np.arange(rb)))[:, np.newaxis, np.newaxis] * I[np.newaxis, ...]
    # [rb*N, N] => [N, rb*N]
    T = T.reshape(-1, N).T

    # Eq. 4 and 5
    H_tilde = np.vstack([
        np.hstack([H.real, -H.imag]), 
        np.hstack([H.imag,  H.real]),
    ])
    y_tilde = np.concatenate([y.real, y.imag])

    # Eq. 8, J is symmetric with diag=0, J[i,j] signifies spin interaction of σi and σj in the Ising model
    # This is different from the original paper because we use normalized transmitted symbol
    J = -T.T @ H_tilde.T @ H_tilde @ T * (2 / qam_var)
    J[np.diag_indices_from(J)] = 0
    z = y_tilde / np.sqrt(qam_var) - H_tilde @ T @ np.ones((N * rb, 1)) / qam_var + (np.sqrt(M) - 1) * H_tilde @ np.ones((N, 1)) / qam_var
    h = 2 * z.T @ H_tilde @ T

    # [rb*N, rb*N], [rb*N, 1]
    return J, h.T

def to_ising_LM_SB(H:ndarray, y:ndarray, nbps:int, lmbd:int=1) -> Tuple[ndarray, ndarray]:
    ''' LM-SB in [arXiv:2306.16264] Deep Unfolded Simulated Bifurcation for Massive MIMO Signal Detection '''

    # the size of constellation, the M-QAM where M in {16, 64, 256}
    M = 2**nbps
    # n_elem at TX side (c=2 for real/imag, 1 symbol = 2 elem)
    Nr, Nt = H.shape
    N = 2 * Nt
    # n_bits/n_spins that one elem decodes to
    rb = nbps // 2

    # QAM variance for normalization
    qam_var = 2 * (M - 1) / 3

    # Eq. 7 the transform matrix T
    I = np.eye(N)
    # [rb, N, N]
    T = (2**(rb - 1 - np.arange(rb)))[:, np.newaxis, np.newaxis] * I[np.newaxis, ...]
    # [rb*N, N] => [N, rb*N]
    T = T.reshape(-1, N).T

    # Eq. 1
    H_tilde = np.vstack([
        np.hstack([H.real, -H.imag]), 
        np.hstack([H.imag,  H.real]),
    ])
    y_tilde = np.concatenate([y.real, y.imag])

    # Eq. 10
    U_λ = np.linalg.inv(H_tilde @ H_tilde.T + lmbd * I) / lmbd   # LMMSE-like part
    J = -T.T @ H_tilde.T @ U_λ @ H_tilde @ T * (2 / qam_var)
    J[np.diag_indices_from(J)] = 0
    z = y_tilde / np.sqrt(qam_var) - H_tilde @ T @ np.ones((N * rb, 1)) / qam_var + (np.sqrt(M) - 1) * H_tilde @ np.ones((N, 1)) / qam_var
    h = 2 * z.T @ U_λ.T @ H_tilde @ T

    # [rb*N, rb*N], [rb*N, 1]
    return J, h.T

def to_ising_DU_LM_SB(H:ndarray, y:ndarray, nbps:int) -> Tuple[ndarray, ndarray]:
    ''' DU-LM-SB in [arXiv:2306.16264] Deep Unfolded Simulated Bifurcation for Massive MIMO Signal Detection '''

def to_ising_MDI_MIMO(H:ndarray, y:ndarray, nbps:int) -> Tuple[ndarray, ndarray]:
    ''' MDI-MIMO from [2304.12830] Uplink MIMO Detection using Ising Machines: A Multi-Stage Ising Approach '''


def solver_qaia_lib(qaia_cls:type[QAIA], J:ndarray, h:ndarray) -> ndarray:
    solver = qaia_cls(J, h, batch_size=100, n_iter=100)
    solver.update()
    sample = np.sign(solver.x)      # [rb*N, B]
    energy = solver.calc_energy()   # [1, B]
    opt_index = np.argmin(energy)
    solution = sample[:, opt_index] # [rb*N], vset {-1, 1}
    return solution


# 选手提供的Ising模型生成函数，可以用我们提供的to_ising
def ising_generator(H:ndarray, y:ndarray, nbps:int, snr:float) -> Tuple[ndarray, ndarray]:
    return to_ising_LM_SB(H, y, nbps, lmbd=25)

# 选手提供的qaia MLD求解器，用mindquantum.algorithms.qaia
def qaia_mld_solver(J:ndarray, h:ndarray) -> ndarray:
    return solver_qaia_lib(BSB, J, h)
