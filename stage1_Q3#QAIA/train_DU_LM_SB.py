#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/05 

import math
import json
import random
import pickle as pkl
from pathlib import Path
from argparse import ArgumentParser
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch import Tensor
from torch.nn import Parameter
import torch.storage
from tqdm import tqdm
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

from run_baseline import modulate_and_transmit

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'    # CPU is even faster :(

BASE_PATH = Path(__file__).parent
LOG_PATH = BASE_PATH / 'log' ; LOG_PATH.mkdir(exist_ok=True)

# Eq. 15 ~ 16 from arXiv:2306.16264
σ = F.sigmoid
sw = lambda x: x * σ(x)
φ_s = lambda x, Λ=10: (1 / Λ) * (Λ * sw(x + 1) - Λ * sw(x - 1)) - 1
ψ_s = lambda x, A=100, B=1.01: σ(A * (torch.abs(x) - B))


# for training
class DU_LM_SB(nn.Module):

  ''' arXiv:2306.16264 Deep Unfolded Simulated Bifurcation for Massive MIMO Signal Detection '''

  def __init__(self, T:int, batch_size:int=100):
    super().__init__()

    self.T = T
    self.batch_size = batch_size
    # Eq. 4
    self.a = torch.linspace(0, 1, T)

    # the T+2 trainable parameters :)
    self.Δ = Parameter(torch.ones  ([T],    dtype=torch.float32), requires_grad=True)
    self.η = Parameter(torch.tensor([1.0],  dtype=torch.float32), requires_grad=True)
    self.λ = Parameter(torch.tensor([25.0], dtype=torch.float32), requires_grad=True)

  def forward(self, H:Tensor, y:Tensor, nbps:int) -> Tensor:
    ''' LM part '''
    J, h = to_ising(H, y, nbps, self.λ)

    ''' DU-SB part '''
    # Eq. 6 and 12
    B = self.batch_size
    N = J.shape[0]
    # from essay
    c_0: float = 2 * math.sqrt((N - 1) / (J**2).sum())
    # from qaia lib
    #c_0: Tensor = 0.5 * math.sqrt(N - 1) / torch.sqrt((J**2).sum())

    # rand init x and y
    x = 0.02 * (torch.rand(N, B, device=H.device) - 0.5)
    y = 0.02 * (torch.rand(N, B, device=H.device) - 0.5)

    # Eq. 11 ~ 14
    for k, Δ_k in enumerate(self.Δ):
      y = y + Δ_k * (-(1 - self.a[k]) * x + self.η * c_0 * (J @ x + h))
      x = x + Δ_k * y
      x = φ_s(x)
      y = y * (1 - ψ_s(x))

    # [B=100, rb*c*Nt=256]
    spins = x.T

    return spins


def to_ising(H:Tensor, y:Tensor, nbps:int, lmbd:Tensor) -> Tuple[Tensor, Tensor]:
  ''' pytorch version of to_ising_LM_SB() '''
  from main import to_ising_LM_SB
  assert to_ising_LM_SB

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
  I = torch.eye(N, device=H.device)
  # [rb, N, N]
  T: Tensor = (2**(rb - 1 - torch.arange(rb, device=H.device)))[:, None, None] * I[None, ...]
  # [rb*N, N] => [N, rb*N]
  T = T.reshape(-1, N).T

  # Eq. 1
  H_tilde = torch.vstack([
    torch.hstack([H.real, -H.imag]), 
    torch.hstack([H.imag,  H.real]),
  ])
  y_tilde = torch.cat([y.real, y.imag])

  # Eq. 10
  U_λ = torch.linalg.inv(H_tilde @ H_tilde.T + lmbd * I) / lmbd   # LMMSE-like part, with our fix
  H_tilde_T = H_tilde @ T
  J = -H_tilde_T.T @ U_λ @ H_tilde_T * (2 / qam_var)
  J = J * (1 - torch.eye(J.shape[0], device=H.device))    # mask diagonal to zeros
  z = (y_tilde - H_tilde_T @ torch.ones([N * rb, 1], device=H.device) + (math.sqrt(M) - 1) * H_tilde @ torch.ones([N, 1], device=H.device)) / math.sqrt(qam_var)
  h = 2 * H_tilde_T.T @ (U_λ @ z)

  # [rb*N, rb*N], [rb*N, 1]
  return J, h


def ber_loss(spins:Tensor, bits:Tensor) -> Tensor:
  ''' differentiable version of compute_ber() '''
  from judger import compute_ber
  assert compute_ber

  # convert the bits from sionna style to constellation style
  # Sionna QAM16 map: https://nvlabs.github.io/sionna/examples/Hello_World.html
  bits_constellation = 1 - torch.cat([bits[..., 0::2], bits[..., 1::2]], dim=-1)

  # Fig. 2 from arXiv:2001.04014, the QuAMax paper converting QuAMax to gray coded
  nbps = bits_constellation.shape[1]
  rb = nbps // 2
  spins = torch.reshape(spins, (rb, 2, -1))  # [rb, c=2, Nt]
  spins = torch.permute(spins, (2, 1, 0))    # [Nt, c=2, rb]
  spins = torch.reshape(spins, (-1, 2*rb))   # [Nt, 2*rb]
  bits_hat = (spins + 1) / 2                 # Ising {-1, +1} to QUBO {0, 1}

  # QuAMax => intermediate code
  bits_final = bits_hat.clone()                           # copy b[0]
  index = torch.nonzero(bits_hat[:, rb-1] > 0.5)[:, -1]   # select even columns
  bits_hat[index, rb:] = 1 - bits_hat[index, rb:]         # invert bits of high part (flip upside-down)
  # Differential bit encoding, intermediate code => gray code (constellation-style)
  for i in range(1, nbps):                                # b[i] = b[i] ^ b[i-1]
    x = bits_hat[:, i] + bits_hat[:, i - 1]
    x_dual = 2 - x
    bits_final[:, i] = torch.where(x <= x_dual, x, x_dual)
  # calc BER
  return F.mse_loss(bits_final, bits_constellation)


def make_random_transmit(bits_shape:torch.Size, H:ndarray, nbps:int, SNR:int) -> Tuple[ndarray, ndarray]:
  ''' transmit random bits through given channel mix H '''
  bits = np.random.uniform(size=bits_shape) < 0.5
  bits = bits.astype(np.float32)
  x, y = modulate_and_transmit(bits, H, nbps, SNR)
  return bits, y


def train(args):
  print('device:', device)
  print('hparam:', vars(args))
  exp_name = f'DU-LM-SB_T={args.n_iter}_lr={args.lr}{"_overfit" if args.overfit else ""}'

  ''' Data '''
  dataset = []
  for idx in tqdm(range(150)):
    if idx > args.limit > 0: break
    with open(f'MLD_data/{idx}.pickle', 'rb') as fh:
      data = pkl.load(fh)
      H = data['H']
      y = data['y']
      bits = data['bits']
      nbps: int = data['num_bits_per_symbol']
      SNR: int = data['SNR']
      dataset.append([H, y, bits, nbps, SNR])

  ''' Model '''
  model = DU_LM_SB(args.n_iter, args.batch_size).to(device)
  optim = Adam(model.parameters(), args.lr)

  ''' Ckpt '''
  init_step = 0
  losses = []
  if args.load:
    print(f'>> resume from {args.load}')
    ckpt = torch.load(args.load, map_location=device)
    init_step = ckpt['steps']
    losses.extend(ckpt['losses'])
    model.load_state_dict(ckpt['model'])
    optim.load_state_dict(ckpt['optim'])

  ''' Train '''
  model.train()
  try:
    for steps in tqdm(range(init_step, args.steps)):
      if not args.no_shuffle and steps % len(dataset) == 0:
        random.shuffle(dataset)
      sample = dataset[steps % len(dataset)]

      H, y, bits, nbps, SNR = sample
      if not args.overfit:
        bits, y = make_random_transmit(bits.shape, H, nbps, SNR)
      H    = torch.from_numpy(H)   .to(device, torch.complex64)
      y    = torch.from_numpy(y)   .to(device, torch.complex64)
      bits = torch.from_numpy(bits).to(device, torch.float32)

      optim.zero_grad()
      spins = model(H, y, nbps)
      loss = torch.stack([ber_loss(sp, bits) for sp in spins]).mean()
      loss.backward()
      optim.step()

      if not 'debug best pred':
        with torch.no_grad():
          from judger import compute_ber
          soluts = torch.sign(spins).detach().cpu().numpy()
          bits_np = bits.cpu().numpy()
          ber = [compute_ber(solut, bits_np) for solut in soluts]
          print('ber:', ber)
          breakpoint()

      if (steps + 1) % 50 == 0:
        losses.append(loss.item())
        print(f'>> [step {steps + 1}] loss: {losses[-1]}')
  except KeyboardInterrupt:
    pass

  ''' Ckpt '''
  ckpt = {
    'steps': steps,
    'losses': losses,
    'model': model.state_dict(),
    'optim': optim.state_dict(),
  }
  torch.save(ckpt, LOG_PATH / f'{exp_name}.pth')

  with torch.no_grad():
    params = {
      'deltas': model.Δ.detach().cpu().numpy().tolist(),
      'eta':    model.η.detach().cpu().item(),
      'lmbd':   model.λ.detach().cpu().item(),
    }
    print('params:', params)

  with open(LOG_PATH / f'{exp_name}.json', 'w', encoding='utf-8') as fh:
    json.dump(params, fh, indent=2, ensure_ascii=False)

  plt.plot(losses)
  plt.tight_layout()
  plt.savefig(LOG_PATH / f'{exp_name}.png', dpi=600)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-T', '--n_iter', default=10, type=int)
  parser.add_argument('-B', '--batch_size', default=256, type=int)
  parser.add_argument('--steps', default=3000, type=int)
  parser.add_argument('--lr', default=1e-2, type=float)
  parser.add_argument('--load', help='ckpt to resume')
  parser.add_argument('-L', '--limit', default=-1, type=int, help='limit dataset n_sample')
  parser.add_argument('--overfit', action='store_true', help='overfit to given dataset')
  parser.add_argument('--no_shuffle', action='store_true', help='no shuffle dataset')
  parser.add_argument('--log_every', default=50, type=int)
  args = parser.parse_args()

  if args.overfit:
    print('[WARN] you are trying to overfit to the given dataset!')

  train(args)
