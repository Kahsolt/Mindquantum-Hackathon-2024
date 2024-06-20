#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/06/21

# 调查加载了预训练权重的时候那该死的训练第一步的随机性是从哪儿来的
# - openfermionpyscf 构造哈密顿量时顺序太不稳定导致的
# - 线路生成的态和 qubit 秩序是一致的

from solution import *

E_to_ham: Dict[float, List[QubitOperator]] = {}

molecule = [
  ['H', [0, 0, 0.0]],
  ['H', [0, 0, 1.0]],
  ['H', [0, 0, 2.0]],
  ['H', [0, 0, 3.0]],
]

for _ in range(10):
  mol = generate_molecule(molecule)
  ham = get_molecular_hamiltonian(mol)      # <- 每次构造哈密顿量可能会有小区别, wtf!!
  circ = get_hae_ry_circit(mol, depth=3)
  p0 = get_hae_ry_circit_pretrained_params(depth=3)

  sim = Simulator('mqvector', circ.n_qubits)
  grad_ops = sim.get_expectation_with_grad(Hamiltonian(ham), circ)
  net = MQAnsatzOnlyLayer(grad_ops)
  net.weight.set_data(Tensor.from_numpy(p0))
  E = net().item()

  const, split_ham = split_hamiltonian(ham)
  split_ham = approx_merge_hamiltonian(split_ham)
  ham_norm = combine_hamiltonian(const, split_ham)
  if not 'show eigen':
    H = ham_norm.matrix().todense()
    evs = np.linalg.eigvalsh(H)
    print('E_approx:', evs[0])    # -2.176284252393974 < -2.166387, 这个近似其实引入了系统误差
  grad_ops = sim.get_expectation_with_grad(Hamiltonian(ham_norm), circ)
  net = MQAnsatzOnlyLayer(grad_ops)
  net.weight.set_data(Tensor.from_numpy(p0))
  E_norm = net().item()
  print('E:', E, 'E_norm:', E_norm)

  E_str = f'{E:.8f}'
  if E_str not in E_to_ham:
    E_to_ham[E_str] = []
  E_to_ham[E_str].append(ham_norm)

print('len(E_to_ham):', len(E_to_ham))
keys = list(E_to_ham.keys())
print('keys:', keys)
ls0 = E_to_ham[keys[0]]
ls1 = E_to_ham[keys[1]]

diff = ls0[0] - ls1[0]
print(diff.compress(1e-8))
