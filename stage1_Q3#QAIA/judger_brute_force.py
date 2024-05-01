from typing import *
from judger import *

from ModulationPy import PSKModem, QAMModem
from numpy import ndarray

# - [ ] 直接暴力枚举 QAM16 求 argmin{x} |y-H*x|^2
# - [x] 直接反解 x => sign(x)

cache: Dict[int, QAMModem] = {}

TABLE_QUMAX = {
    '0000': complex(-3, -3),
    '0001': complex(-3, -1),
    '0010': complex(-3, 1),
    '0011': complex(-3, 3),
    '0100': complex(-1, -3),
    '0101': complex(-1, -1),
    '0110': complex(-1, 1),
    '0111': complex(-1, 3),
    '1000': complex(1, -3),
    '1001': complex(1, -1),
    '1010': complex(1, 1),
    '1011': complex(1, 3),
    '1100': complex(3, -3),
    '1101': complex(3, -1),
    '1110': complex(3, 1),
    '1111': complex(3, 3),
}

def QuAMax(bits:ndarray) -> ndarray:
    bit_str = ''.join([str(b) for b in bits])
    return np.asarray([TABLE_QUMAX[bit_str]])


TABLE_INTERM = {
    '0000': complex(-3, -3),
    '0001': complex(-3, -1),
    '0010': complex(-3, 1),
    '0011': complex(-3, 3),
    '0111': complex(-1, -3),
    '0110': complex(-1, -1),
    '0101': complex(-1, 1),
    '0100': complex(-1, 3),
    '1000': complex(1, -3),
    '1001': complex(1, -1),
    '1010': complex(1, 1),
    '1011': complex(1, 3),
    '1111': complex(3, -3),
    '1110': complex(3, -1),
    '1101': complex(3, 1),
    '1100': complex(3, 3),
}

def InterM(bits:ndarray) -> ndarray:
    bit_str = ''.join([str(b) for b in bits])
    return np.asarray([TABLE_INTERM[bit_str]])


class Judger:

    def __init__(self, test_cases):
        self.test_cases = test_cases

    @staticmethod
    def infer(H, y, bits_truth, num_bits_per_symbol, snr):
        M = 2**num_bits_per_symbol
        if M not in cache:
            cache[M] = QAMModem(M, bin_input=True, soft_decision=False, bin_output=True)
        modem = cache[M]

        x = np.stack([modem.demodulate(it) for it in y], axis=0)
        breakpoint()

        return x

    def benchmark(self):
        avgber = 0
        for i, case in enumerate(tqdm(self.test_cases)):
            H, y, bits_truth, num_bits_per_symbol, snr, ZF_ber = case
            if num_bits_per_symbol != 4: continue
            if snr != 20: continue
            bits_decode = self.infer(H, y, bits_truth, num_bits_per_symbol, snr)
            ber = compute_ber(bits_decode, bits_truth)
            avgber += ber
            print(f'[case {i}] ans: {ber}, ref: {ZF_ber}')
        avgber /= len(self.test_cases)
        return avgber


if __name__ == '__main__':
    dataset = []
    filelist = glob(f'MLD_data/*.pickle')
    for filename in filelist:
        with open(filename, 'rb') as fh:
            data = pickle.load(fh)
        dataset.append([data['H'], data['y'], data['bits'], data['num_bits_per_symbol'], data['SNR'], data['ZF_ber']])

    # 该数据集的理论 BER 下限
    judger = Judger(dataset)
    t = time()
    avgber = judger.benchmark()
    print(f'>> time cost: {time() - t:.2f}')
    print(f">> avg. BER = {avgber:.5f}")
