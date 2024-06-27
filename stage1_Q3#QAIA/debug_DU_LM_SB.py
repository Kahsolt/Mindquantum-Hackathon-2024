#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/06/27 

# 调查 DU-LM-SB 训练 loss 可以低至 0.08，为什么 推理误差还是高达 0.199？
# - 训练-推理的代码是一致的，没问题
# - 问题还是这种建模方式本身没有泛化性，同一套 H 但不同的 bits 训练出来的模型泛化不到另一组 bits，怎么回事呢？

import json
import pickle as pkl
from glob import glob
from typing import *

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from train_LM_SB import DU_LM_SB, to_ising_ext as to_ising_ext_train, ber_loss
from main import LOG_PATH, to_ising_ext as to_ising_ext_infer
from judger import compute_ber
from qaia.DUSB import DUSB

DU_LM_SB_weights = LOG_PATH / 'DU-LM-SB_T=10_lr=0.0001.json'

''' Load pretrained weights '''
with open(DU_LM_SB_weights, 'r', encoding='utf-8') as fh:
  params = json.load(fh)
  deltas: List[float] = params['deltas']
  eta: float = params['eta']
  lmbd: float = params['lmbd']

model = DU_LM_SB(T=10, batch_size=1)
model.Δ.data[:] = torch.tensor(deltas)
model.η.data[:] = torch.tensor([eta])
model.λ.data[:] = torch.tensor([lmbd])


@torch.inference_mode()
def run():
  ber_loss_list = []
  ber_score_list = []
  ber_ZF_list = []

  for i, fp in enumerate(glob(f'MLD_data/*.pickle')):
    with open(fp, 'rb') as fh:
      data = pkl.load(fh)
    H, y, bits_truth, nbps, snr, ZF_ber = data['H'], data['y'], data['bits'], data['num_bits_per_symbol'], data['SNR'], data['ZF_ber']
    H_t = torch.from_numpy(H)
    y_t = torch.from_numpy(y)
    lmbd_t = torch.tensor(lmbd)
    bits_t = torch.from_numpy(bits_truth)
    ber_ZF_list.append(ZF_ber)

    # J and h
    J_t, h_t = to_ising_ext_train(H_t, y_t, nbps, lmbd=lmbd_t)
    J_i, h_i = to_ising_ext_infer(H, y, nbps, lmbd=lmbd)
    try:
      assert np.allclose(J_t, J_t.numpy(), atol=1e-5)
      assert np.allclose(h_i, h_t.numpy(), atol=1e-5)
    except AssertionError:
      print('>> [to_ising_ext] match failed')
      breakpoint()

    # SB solver
    solver = DUSB(J_i, h_i, deltas, eta, batch_size=1)
    solver.update()
    res_i = np.sign(solver.x[:, 0])
    ber_infer = compute_ber(res_i, bits_truth)
    ber_score_list.append(ber_infer)

    res_t = model(H_t, y_t, nbps)
    ber_train = ber_loss(res_t, bits_t)
    ber_loss_list.append(ber_train.item())

    res_t_spin = torch.sign(res_t)
    ok = res_t_spin.numpy() == res_i
    print(f'[{i}] infer BER: {ber_infer:.5f} train BER_loss: {ber_train:.5f}, match rate: {ok.sum() / res_i.size:.3%}')

  plt.clf()
  plt.plot(ber_score_list, 'b',    label='infer')
  plt.plot(ber_loss_list,  'r',    label='train')
  plt.plot(ber_ZF_list,    'grey', label='ZF')
  plt.suptitle('BER')
  plt.legend()
  plt.tight_layout()
  plt.show()


run()
