#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/06/04 

# 测试各线路性能：无噪优化，含噪测量，找到最优ansatz
#  - HEA(3) 能制备较为准确的态，也相对抗噪
#  - CC(1) 就能制备很准确的态了，但含噪条件下比 HAE(3) 差 (why?)

import pickle as pkl
from pathlib import Path

from tqdm import tqdm

from solution import *
import simulator ; simulator.init_shots_counter()

BASE_PATH = Path(__file__).parent
IMG_PATH = BASE_PATH / 'img' ; IMG_PATH.mkdir(exist_ok=True)
save_fp = IMG_PATH / f'{Path(__file__).stem}.pkl'
save_img_fp = IMG_PATH / f'{Path(__file__).stem}.png'

ANSATZS = {
  **{f'HAE({i})':       lambda mol: get_hae_ry_circit        (mol, i) for i in range(1, 6+1)},
  **{f'HAEc({i})':      lambda mol: get_hae_ry_compact_circit(mol, i) for i in range(1, 6+1)},
  **{f'CC({i})':        lambda mol: get_cnot_centric_circit  (mol, i) for i in range(1, 6+1)},
  **{f'pCHC({i}, {t})': lambda mol: get_pchc_circuit         (mol, i, t) for i in range(2, 4+1) for t in ['s', 'd', 'sd']},
}

def run():
  ''' Data '''
  molecule = [
    ['H', [0, 0, 0.0]],
    ['H', [0, 0, 1.0]],
    ['H', [0, 0, 2.0]],
    ['H', [0, 0, 3.0]],
  ]
  mol = generate_molecule(molecule)
  ham = get_molecular_hamiltonian(mol)
  const, split_ham = split_hamiltonian(ham)
  split_ham = prune_hamiltonian(split_ham)

  ''' Records '''
  if save_fp.exists():
    with open(save_fp, 'rb') as fh:
      stats = pkl.load(fh)
  else:
    stats = {}

  ''' Experiments '''
  N_SHOTS = 1000
  N_OPTIM_TRIAL = 5
  for name, get_circ in tqdm(ANSATZS.items()):
    if name in stats: continue
    print(f'[{name}]')
    circ = get_circ(mol)

    ''' Noiseless Optimize '''
    fmin = 99999
    fval = None
    pr = None
    for _ in range(N_OPTIM_TRIAL):
      fval, pr_new = get_best_params(circ, ham, method='Adam', init='randn')
      if fval < fmin:
        fmin = fval
        pr = pr_new
    print('   best fval:', fval)
    circ, pr = prune_circuit(circ, pr)

    ''' Noiseless Measure '''
    sim = Simulator('mqvector', mol.n_qubits)
    exp_ideal = sim.get_expectation(Hamiltonian(ham), circ, pr=pr).real
    sim.apply_circuit(circ, pr)
    qs = sim.get_qs()

    ''' Noisy Measure '''
    sim_noisy = HKSSimulator('mqvector', mol.n_qubits)
    exp_qmeas = const
    for coeff, ops in tqdm(split_ham):
      exp = measure_single_ham(sim_noisy, circ, pr, ops, N_SHOTS)
      exp_qmeas += exp * coeff

    ''' Noisy Measure (empty-ref) '''
    pr_empty = ParameterResolver(dict(zip(circ.params_name, np.zeros(len(circ.params_name)))))
    exp_qmeas_ref = const
    for coeff, ops in tqdm(split_ham):
      exp = measure_single_ham(sim_noisy, circ, pr_empty, ops, N_SHOTS)
      exp_qmeas_ref += exp * coeff

    stats[name] = {
      'qs': qs,
      'E_ideal': exp_ideal,
      'E_noisy': exp_qmeas,
      'E_noisy_ref': exp_qmeas_ref,
    }

    with open(save_fp, 'wb') as fh:
      pkl.dump(stats, fh)

  E_ideal_list = []
  E_noisy_list = []
  E_noisy_ref_list = []
  for name, stat in stats.items():
    E_ideal = stat["E_ideal"]
    E_noisy = stat["E_noisy"]
    E_noisy_ref = stat["E_noisy_ref"]

    E_ideal_list.append(E_ideal)
    E_noisy_list.append(E_noisy)
    E_noisy_ref_list.append(E_noisy_ref)

    print(f'[{name}]')
    print(f'  E_ideal: {E_ideal}')
    print(f'  E_noisy: {E_noisy}')
    print(f'  ΔE: {E_noisy - E_ideal}')
    print(f'  E_noisy_ref: {E_noisy_ref}')
    print(f'  ΔE_ref: {E_noisy - E_noisy_ref}')

  ''' Plot '''
  import matplotlib.pyplot as plt
  plt.plot(E_ideal_list, label='E_ideal')
  plt.plot(E_noisy_list, label='E_noisy')
  plt.plot(E_noisy_ref_list, label='E_noisy_ref')
  plt.legend()
  plt.tight_layout()
  plt.savefig(save_img_fp, dpi=400)


if __name__ == '__main__':
  run()
