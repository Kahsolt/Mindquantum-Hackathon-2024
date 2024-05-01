"""online judger for QAIA-MLD problem"""

import pickle
from time import time
from glob import glob

import numpy as np
from tqdm import tqdm

'''
本文件禁止改动!!!
'''

def compute_ber(solution, bits):
    '''
    Compute BER for the solution from QAIAs.

    Firstly, both the solution from QAIAs and generated bits should be transformed into gray-coded, 
    and then compute the ber.

    Reference
    ---------
    [1] Kim M, Venturelli D, Jamieson K. Leveraging quantum annealing for large MIMO processing in centralized radio access networks. 
        Proceedings of the ACM special interest group on data communication. 2019: 241-255.\
    
    Input
    -----
    solution: [Nt*nbps, ], np.int
        The binary array filled with ones and minus ones.

    bits: [Nt, nbps], np.int
        The binary array filled with ones and zeros.
    Ouput
    -----
    ber: np.float
        A scalar, the BER.
    '''
    ## convert the bits from sionna style to constellation style
    bits_constellation = 1 - np.concatenate([bits[..., 0::2], bits[..., 1::2]], axis=-1)
    num_bits_per_symbol = bits_constellation.shape[1]
    
    ## convert QuAMax transform to gray coded
    rb = num_bits_per_symbol//2
    bits_hat = solution.reshape(rb, 2, -1)
    bits_hat = np.concatenate([bits_hat[:, 0], bits_hat[:, 1]], 0)
    bits_hat = bits_hat.T.copy()
    # convert Ising {-1, 1} to QUBO {0, 1}
    bits_hat[bits_hat == -1] = 0
    # Differential bit encoding
    index = np.nonzero(bits_hat[:, rb-1] == 1)[0]
    output_bit = bits_hat.copy()
    bits_hat[index, rb:] = 1 - bits_hat[index, rb:]
    for i in range(num_bits_per_symbol - 1):
        output_bit[:, i + 1] = np.logical_xor(bits_hat[:, i], bits_hat[:, i + 1]).astype(np.float32)
    ber = np.mean(bits_constellation != output_bit)
    return ber


class Judger:
    """Judge contestant's algorithm with MLD test cases."""

    def __init__(self, test_cases):
        self.test_cases = test_cases

    @staticmethod
    def infer(ising_generator, qaia_mld_solver, H, y, num_bits_per_symbol, snr):
        J, h = ising_generator(H, y, num_bits_per_symbol, snr)
        bits = qaia_mld_solver(J, h)
        return bits

    def benchmark(self, ising_gen, qaia_mld_solver):
        avgber = 0
        for i, case in enumerate(tqdm(self.test_cases)):
            H, y, bits_truth, num_bits_per_symbol, snr, ZF_ber = case
            bits_decode = self.infer(ising_gen, qaia_mld_solver, H, y, num_bits_per_symbol, snr)
            ber = compute_ber(bits_decode, bits_truth)
            avgber += ber
            print(f'[case {i}] ans: {ber}, ref: {ZF_ber}')
        avgber /= len(self.test_cases)
        return avgber


if __name__ == "__main__":
    from main import ising_generator, qaia_mld_solver

    dataset = []
    filelist = glob(f'MLD_data/*.pickle')
    # filelist = ['MLD_data/16x16_snr10.pickle']
    for filename in filelist:
        with open(filename, 'rb') as fh:
            data = pickle.load(fh)
        dataset.append([data['H'], data['y'], data['bits'], data['num_bits_per_symbol'], data['SNR'], data['ZF_ber']])

    # 测试选手的平均ber，越低越好
    judger = Judger(dataset)
    t = time()
    avgber = judger.benchmark(ising_generator, qaia_mld_solver)
    print(f'>> time cost: {time() - t:.2f}')
    print(f">> avg. BER = {avgber:.5f}")
