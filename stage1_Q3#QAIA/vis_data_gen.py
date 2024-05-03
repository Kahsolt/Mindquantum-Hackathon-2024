#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/03

# 理解数据是如何造出来的: y = H @ x + n
# - 只要噪声 n 能被完全估计出来，那么 ZF 方法就是完全可行的 (但这不可能)
# - 只要噪声 n 的估计是错误的，ZF 会导致很大的误差积累

import matplotlib.pyplot as plt

from run_baseline import *

# https://stackoverflow.com/questions/15505514/binary-numpy-array-to-list-of-integers
bits_to_number = lambda bits: bits.dot(1 << np.arange(bits.shape[-1] - 1, -1, -1))


def test_data_gen(idx:int):
    fp = f'MLD_data/{idx}.pickle'
    with open(fp, 'rb') as fh:
        data = pkl.load(fh)
        H: ndarray = data['H']
        y: ndarray = data['y']
        bits: ndarray = data['bits'].astype(np.int32)
        nbps: int = data['num_bits_per_symbol']
        SNR: int = data['SNR']
        ZF_ber: float = data['ZF_ber']

    print('H.shape:', H.shape)
    print('y.shape:', y.shape)
    print('bits.shape:', bits.shape)
    print('nbps:', nbps)
    print('SNR:', SNR)
    print('ZF_ber:', ZF_ber)

    mapper = get_mapper(nbps)
    b = tf.convert_to_tensor(bits, dtype=tf.int32)
    x: ndarray = mapper(b).cpu().numpy()
    color = bits_to_number(bits)

    # SNR(dB) := 10*log10(P_signal/P_noise) ?= Var(signal) / Var(noise)
    sigma = np.var(b) / SNR
    noise = np.random.normal(scale=sigma**0.5, size=x.shape)
    y_hat = H @ x + noise
    
    # 0. perfect recon, when noise is known (impossible)
    #x_hat = np.linalg.inv(H) @ (y_hat - noise)
    # 1. ZF-method (seemingly ok in cases)
    x_hat = np.linalg.inv(H) @ y_hat
    # 2. ZF-method with resampled noise (even also seemingly ok in cases)
    #noise2 = np.random.normal(scale=sigma**0.5, size=x.shape)
    #x_hat = np.linalg.inv(H) @ (y_hat - noise2)
    # 3. ZF-method with GT data (not ok in almost all cases, because y=y_hat+noise3, the noise will be amplified by H')
    #noise2 = np.random.normal(scale=sigma**0.5, size=x.shape)
    #x_hat = np.linalg.inv(H) @ (y - noise2)

    plt.subplot(221) ; plt.scatter(x.real,     x.imag,     c=color, cmap='Spectral') ; plt.title('x = QAM(bits)')
    plt.subplot(222) ; plt.scatter(y_hat.real, y_hat.imag, c=color, cmap='Spectral') ; plt.title('y_hat = H @ x + noise')
    plt.subplot(223) ; plt.scatter(x_hat.real, x_hat.imag, c=color, cmap='Spectral') ; plt.title('x_hat = inv(H) @ y_hat')
    plt.subplot(224) ; plt.scatter(y.real,     y.imag,     c=color, cmap='Spectral') ; plt.title('y (GT)')
    plt.suptitle(f'id={idx} Ht={H.shape[1]} SNR={SNR} nbps={nbps}')
    plt.show()


if __name__ == '__main__':
    for i in range(150):
        test_data_gen(i)
