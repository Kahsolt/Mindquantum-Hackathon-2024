#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/07 

import json
import random
from copy import deepcopy
from re import compile as Regex
from pathlib import Path
from argparse import ArgumentParser

import mindspore as ms
import mindspore.ops.functional as F
from mindspore.nn.optim import Adam
from mindspore.nn import TrainOneStepCell
from mindspore import context
from mindquantum.simulator import Simulator
from mindquantum.core.operators import Hamiltonian
from mindquantum.framework import MQAnsatzOnlyLayer
import numpy as np
from tqdm import tqdm

from vis_transfer_data import load_lookup_table, LookupTable
from utils.qcirc import qaoa_hubo, build_ham_high
from score import load_data
from main import ave_D, order, trans_gamma, rescale_factor, load_finetuned_lookup_table

context.set_context(device_target='CPU', mode=ms.PYNATIVE_MODE, pynative_synchronize=True)

BASE_PATH = Path(__file__).parent
LOG_PATH = BASE_PATH / 'log' ; LOG_PATH.mkdir(exist_ok=True)

R_ITER = Regex('iter=(\d+)')


def _cvt_lookup_table(lookup_table:LookupTable) -> LookupTable:
  ret = deepcopy(lookup_table)
  for p, data in ret.items():
    for k, v in data.items():
      ret[p][k] = v.tolist()
  return ret


def train(args):
  ''' Data '''
  dataset = []
  for propotion in [0.3, 0.6, 0.9]:
    for k in range(2, 6):
      for coef in ['std', 'uni', 'bimodal']:
        for r in range(10):
          Jc_dict = load_data(f"data/k{k}/{coef}_p{propotion}_{r}.json")
          dataset.append(Jc_dict)

  if args.load:
    lookup_table = load_finetuned_lookup_table(args.load)
    try:
      init_iter = int(R_ITER.findall(args.load)[0])
    except:
      init_iter = 0
  else:
    lookup_table = load_lookup_table()
    init_iter = 0
    with open(LOG_PATH / 'lookup_table-origial.json', 'w', encoding='utf-8') as fh:
      json.dump(_cvt_lookup_table(lookup_table), fh, indent=2, ensure_ascii=False)

  ''' Simulator '''
  Nq = 12
  sim = Simulator('mqvector', n_qubits=Nq)

  ''' Train '''
  for iter in tqdm(range(init_iter, args.iters), initial=init_iter, total=args.iters):
    if iter % len(dataset) == 0:
      random.shuffle(dataset)
    # random pick a ham and circuit depth
    Jc_dict = dataset[iter % len(dataset)]
    p = random.choice([4, 8])

    ham = Hamiltonian(build_ham_high(Jc_dict))
    D = ave_D(Jc_dict, Nq)
    k = min(order(Jc_dict), 6)

    # vqc
    gamma_params = [f'g{i}' for i in range(p)]
    beta_params  = [f'b{i}' for i in range(p)]
    circ = qaoa_hubo(Jc_dict, Nq, gamma_params, beta_params, p=p)

    # init_p
    params = lookup_table[p][k]
    gammas, betas = np.split(params, 2)
    factor = rescale_factor(Jc_dict) * 1.275    # rescale gamma
    gammas = trans_gamma(gammas, D) * factor

    # align order with qcir (param => weights)
    init_weights = []
    for pname in circ.params_name:
      which = pname[0]
      idx = int(pname[1:])
      if which == 'g':
        init_weights.append(gammas[idx])
      else:
        init_weights.append(betas[idx])
    init_weights = ms.tensor(init_weights)

    # train
    grad_ops = sim.get_expectation_with_grad(ham, circ)
    net = MQAnsatzOnlyLayer(grad_ops, weight=init_weights)
    opt = Adam(net.trainable_params(), learning_rate=args.lr)
    train_step = TrainOneStepCell(net, opt)
    for step in range(args.steps):
      loss = train_step()
      if step % 10 == 0:
        print(f'>> [step {step}] loss: {loss.item()}')

    L1 = F.l1_loss(net.weight, init_weights)
    print(f'>> [dist] L1: {L1.mean()}, Linf: {L1.max()}')

    # de-align order to lookup_table (weights => param)
    tuned_weights = net.weight.asnumpy()
    tuned_gammas = [None] * p
    tuned_betas  = [None] * p
    for pname, pvalue in zip(circ.params_name, tuned_weights):
      which = pname[0]
      idx = int(pname[1:])
      if which == 'g':
        tuned_gammas[idx] = pvalue
      else:
        tuned_betas[idx] = pvalue

    # update lookup table
    tuned_gammas = np.asarray(tuned_gammas)
    tuned_gammas /= factor * np.arctan(1 / np.sqrt(D - 1))  # inv rescale gamma
    tuned_betas = np.asarray(tuned_betas)
    tuned_params = np.concatenate([tuned_gammas, tuned_betas])
    lookup_table[p][k] = (1 - args.dx) * params + args.dx * tuned_params

    # tmp ckpt
    if (iter + 1) % 100 == 0:
      with open(LOG_PATH / f'lookup_table-iter={iter+1}.json', 'w', encoding='utf-8') as fh:
        json.dump(_cvt_lookup_table(lookup_table), fh, indent=2, ensure_ascii=False)

  ''' Ckpt '''
  with open(LOG_PATH / 'lookup_table.json', 'w', encoding='utf-8') as fh:
    json.dump(_cvt_lookup_table(lookup_table), fh, indent=2, ensure_ascii=False)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--iters', default=10000, type=int)
  parser.add_argument('--steps', default=100, type=int)
  parser.add_argument('--lr', default=1e-5, type=float)
  parser.add_argument('--dx', default=0.1, type=float)
  parser.add_argument('--load')
  args = parser.parse_args()

  train(args)
