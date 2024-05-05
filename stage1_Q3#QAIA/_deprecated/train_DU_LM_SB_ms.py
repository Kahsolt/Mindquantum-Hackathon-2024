#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/05 

# MindSpore has very bad CPU and dynamic shape support!!! :(
# FIXME: deprecated, DO NOT USE!!

import math
import random
import pickle as pkl
from pathlib import Path
from argparse import ArgumentParser
from typing import Tuple

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops.functional as F
from mindspore.nn.optim import Adam
from mindspore import context
from mindspore import Tensor, Parameter
from tqdm import tqdm

context.set_context(device_target='CPU', mode=ms.PYNATIVE_MODE, pynative_synchronize=True)

BASE_PATH = Path(__file__).parent
LOG_PATH = BASE_PATH / 'log' ; LOG_PATH.mkdir(exist_ok=True)

# Eq. 15 ~ 16 from arXiv:2306.16264
σ = lambda x: 1 / (1 + F.exp(-x))
sw = lambda x: x * σ(x)
φ_s = lambda x, Λ=10: (1 / Λ) * (Λ * sw(x + 1) - Λ * sw(x - 1)) - 1
ψ_s = lambda x, A=100, B=1.01: σ(A * (F.abs(x) - B))


# for training
class DU_LM_SB(nn.Cell):

  ''' arXiv:2306.16264 Deep Unfolded Simulated Bifurcation for Massive MIMO Signal Detection '''

  def __init__(self, T:int, batch_size:int=100):
    super().__init__()

    self.T = T
    self.batch_size = batch_size
    # Eq. 4
    self.a = F.linspace(0, 1, T)

    # the T+2 trainable parameters :)
    self.Δ = Parameter(F.ones([T],   dtype=ms.float32), requires_grad=True)
    self.η = Parameter(Tensor([1.0], dtype=ms.float32), requires_grad=True)
    self.λ = Parameter(Tensor([1.0], dtype=ms.float32), requires_grad=True)

  def to_ising(self, H:Tensor, y:Tensor, nbps:int) -> Tuple[Tensor, Tensor]:
    # the size of constellation, the M-QAM where M in {16, 64, 256}
    M = 2**nbps
    # n_elem at TX side (c=2 for real/imag, 1 symbol = 2 elem)
    Nr, Nt = H.shape
    N = 2 * Nt
    # n_bits/n_spins that one elem decodes to
    rb = nbps // 2

    # QAM variance for normalization
    qam_var = 2 * (M - 1) / 3

    # Eq. 7 the transform matrix T from arXiv:2105.10535
    I = F.eye(N)
    # [rb, N, N]
    T: Tensor = (2**(rb - 1 - F.arange(rb)))[:, None, None] * I[None, ...]
    # [rb*N, N] => [N, rb*N]
    T = T.reshape(-1, N).T

    # Eq. 1
    H_tilde = F.vstack([
      F.hstack([H.real(), -H.imag()]), 
      F.hstack([H.imag(),  H.real()]),
    ])
    y_tilde = F.cat([y.real(), y.imag()])

    # Eq. 10
    U_λ = F.inverse(H_tilde @ H_tilde.T + self.λ * I)     # LMMSE-like part
    J = -T.T @ H_tilde.T @ U_λ @ H_tilde @ T * (2 / qam_var)
    J = J * (1 - F.eye(J.shape[0]))    # mask diagonal to zeros
    z = y_tilde / math.sqrt(qam_var) - H_tilde @ T @ F.ones([N * rb, 1]) / qam_var + (math.sqrt(M) - 1) * H_tilde @ F.ones([N, 1]) / qam_var
    h = 2 * z.T @ U_λ.T @ H_tilde @ T

    # [rb*N, rb*N], [rb*N, 1]
    return J, h.T

  def construct(self, H:Tensor, y:Tensor, bits:Tensor, nbps:int) -> Tensor:
    ''' LM part '''
    J, h = self.to_ising(H, y, nbps)

    ''' DU-SB part '''
    # Eq. 6 and 12
    B = self.batch_size
    N = J.shape[0]
    c_0: float = 2 * math.sqrt((N - 1) / (J**2).sum())

    # rand init x and y
    x = 0.02 * (F.rand(N, B) - 0.5)
    y = 0.02 * (F.rand(N, B) - 0.5)

    # Eq. 11 ~ 14
    for k, Δ_k in enumerate(self.Δ):
      y = y + Δ_k * (-(1 - self.a[k]) * x + self.η * c_0 * (J @ x + h))
      x = x + Δ_k * y
      x = φ_s(x)
      y = y * (1 - ψ_s(x))

    spins = φ_s(x)   # ~F.sign, [rb*c*Nt=256, B=100]

    if 'reformat to QuAMax according to judger.compute_ber()':
      rb = nbps // 2
      spins = F.reshape(spins, (rb, 2, -1, B))    # [rb, c=2, Nt, B]
      spins = F.permute(spins, (3, 2, 1, 0))  # [B, Nt, c=2, rb]
      spins = F.reshape(spins, (B, -1, 2*rb)) # [B, Nt, 2*rb]
      bits_hat = (spins + 1) / 2                    # Ising to QUBO

    loss = F.mse_loss(Tensor.mean(bits_hat, axis=0), bits)
    return loss


def train(args):
  ''' Data '''
  dataset = []
  for idx in tqdm(range(150)):
    with open(f'MLD_data/{idx}.pickle', 'rb') as fh:
      data = pkl.load(fh)
      H = data['H']
      y = data['y']
      bits = data['bits']
      nbps: int = data['num_bits_per_symbol']
      SNR: int = data['SNR']
      dataset.append([H, y, bits, nbps, SNR])

  ''' Model '''
  model = DU_LM_SB(args.n_iter, args.batch_size)
  optim = Adam(model.trainable_params(), args.lr)
  grad_fn = ms.value_and_grad(model, None, optim.parameters, has_aux=False)

  ''' Train '''
  model.set_train()
  for steps in tqdm(range(args.steps)):
    if steps % len(dataset) == 0:
      random.shuffle(dataset)
    sample = dataset[steps % len(dataset)]

    H, y, bits, nbps, SNR = sample
    H    = Tensor.from_numpy(H)   .astype(ms.complex64)
    y    = Tensor.from_numpy(y)   .astype(ms.complex64)
    bits = Tensor.from_numpy(bits).astype(ms.float32)

    loss, grads = grad_fn(H, y, bits, nbps)
    optim(grads)

    if steps % 1 == 0:
      print(f'>> [step {steps}] loss: {loss.item()}')

  ''' Ckpt '''
  print('param_dict:', model.parameters_dict())
  ms.save_checkpoint(model.parameters_dict(), LOG_PATH / 'DU-LM-SB.ckpt')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-T', '--n_iter', default=10, type=int)
  parser.add_argument('-B', '--batch_size', default=2000, type=int)
  parser.add_argument('--steps', default=1000, type=int)
  parser.add_argument('--lr', default=2e-4, type=float)
  args = parser.parse_args()

  train(args)
