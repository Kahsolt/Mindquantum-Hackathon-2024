#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/07 

import numpy as np
from mindquantum.simulator import Simulator
from mindquantum.core.operators import Hamiltonian

from utils.qcirc import qaoa_hubo, build_ham_high
from score import load_data
from main import main, LOG_PATH

from code import interact

save_fp = LOG_PATH / 'score_mat.npy'

def run():
  Nq = 12
  sim = Simulator('mqvector', n_qubits=Nq)

  # [prop, k, coef, r, depth(p)]
  score_mat = np.zeros([3, 4, 3, 10, 2], dtype=np.float32)

  for pid, propotion in enumerate([0.3, 0.6, 0.9]):
    for kid, k in enumerate(range(2, 6)):
      for cid, coef in enumerate(['std', 'uni', 'bimodal']):
        for r in range(10):
          fp = f"data/k{k}/{coef}_p{propotion}_{r}.json"
          Jc_dict = load_data(fp)
          ham = Hamiltonian(build_ham_high(Jc_dict))

          for did, depth in enumerate([4, 8]):
            gamma_List, beta_List = main(Jc_dict, depth, Nq)
            circ = qaoa_hubo(Jc_dict, Nq, gamma_List, beta_List, p=depth)
            E = sim.get_expectation(ham, circ).real
            score_mat[pid, kid, cid, r, did] = E

  np.save(save_fp, score_mat)


if not save_fp.exists():
  run()

score_mat = np.load(save_fp)

if 'stats':
  [score_mat[i, ...]         .mean() for i in range(3)]
  [score_mat[i, ...]         .std()  for i in range(3)]
  [score_mat[:, i, ...]      .mean() for i in range(4)]
  [score_mat[:, i, ...]      .std()  for i in range(4)]
  [score_mat[:, :, i, ...]   .mean() for i in range(3)]
  [score_mat[:, :, i, ...]   .std()  for i in range(3)]
  [score_mat[:, :, :, i, ...].mean() for i in range(10)]
  [score_mat[:, :, :, i, ...].std()  for i in range(10)]
  [score_mat[..., i]         .mean() for i in range(2)]
  [score_mat[..., i]         .std()  for i in range(2)]

# [prop] edge density
#   avg: [-127.36577, -157.32594, -135.90237]
#   std: [71.151474, 88.70964, 119.21811]
# [k] order of ising-model
#   avg: [-68.222496, -125.817276, -138.61066, -228.14171]
#   std: [29.53053, 50.32695, 109.61016, 91.023994]
# [coef] edge weight probdist
#   avg: [-89.83214, -114.52684, -216.2351]
#   std: [78.56908, 56.96428, 96.05193]
# [r] sample index
#   avg: [-139.10056, -139.65471, -140.61537, -140.91692, -141.76175, -138.6221, -138.42047, -141.0248, -141.98668, -139.87698]
#   std: [96.0376, 97.350685, 93.64047, 96.520065, 95.76261, 93.9899, 96.85584, 96.792404, 99.28371, 93.06068]
# [p] circuit depth
#   avg: [-126.21123, -154.18483]
#   std: [82.35353, 106.02304]

interact(local=globals())
