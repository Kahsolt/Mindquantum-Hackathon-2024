#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/06 

import pickle as pkl

import torch
import numpy as np

from train_DU_LM_SB import DU_LM_SB, to_ising
from qaia import DUSB
from main import to_ising_LM_SB


def run(idx:int):
  with open(f'MLD_data/{idx}.pickle', 'rb') as fh:
    data = pkl.load(fh)
    H = data['H']
    y = data['y']
    bits = data['bits']
    nbps: int = data['num_bits_per_symbol']
    SNR: int = data['SNR']

  deltas = [1.0] * 100
  eta = 1.0
  lmbd = 25.0

  J, h = to_ising_LM_SB(H, y, nbps, lmbd)
  solver = DUSB(J, h, deltas, eta, batch_size=100)
  solver.update()
  x = solver.x

  model = DU_LM_SB(T=len(deltas), batch_size=100)
  model.Δ.data = torch.tensor(deltas)
  model.η.data = torch.tensor([eta])
  model.λ.data = torch.tensor([lmbd])
  HH = torch.from_numpy(H)
  yy = torch.from_numpy(y)
  JJ, hh = to_ising(HH, yy, nbps=nbps, lmbd=model.λ.data)
  spins = model(HH, yy, nbps)

  assert np.allclose(J, JJ.numpy())
  assert np.allclose(h, hh.numpy())
  assert np.isclose(x.max(), spins.max().item(), atol=1e-3)
  assert np.isclose(x.min(), spins.min().item(), atol=1e-3)
  
  print('Test passed')


if __name__ == '__main__':
  run(0)
