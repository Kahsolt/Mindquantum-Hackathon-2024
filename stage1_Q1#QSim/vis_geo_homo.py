#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/06/21 

# 查看空间结构上等价但表达不同的 geo 对 energy 和 ham 的影响
# - energy 只有浮点尾数的微小精度区别；写法一 (有一个原子在原点) 是数值最稳定的，无随机性
# - ham 的构造有一定概率不同，主要是 X 和 Y 的对调 (以及相位负号的对调)

from solution import *

def get_ham_and_E(molecule) -> Tuple[QubitOperator, Tuple[float, float]]:
  mol = generate_molecule(molecule)
  ham = get_molecular_hamiltonian(mol)
  return ham, (mol.hf_energy, mol.fci_energy)

hams: List[QubitOperator] = []

molecule = [
  ['H', [0, 0, 0.0]],
  ['H', [0, 0, 1.0]],
  ['H', [0, 0, 2.0]],
  ['H', [0, 0, 3.0]],
]
ham, (E_hf, E_fci) = get_ham_and_E(molecule)
print('E_hf:', E_hf, 'E_fci:', E_fci)
hams.append(ham)

molecule = [
  ['H', [0, 0, 1.0]],
  ['H', [0, 0, 2.0]],
  ['H', [0, 0, 3.0]],
  ['H', [0, 0, 4.0]],
]
ham, (E_hf, E_fci) = get_ham_and_E(molecule)
print('E_hf:', E_hf, 'E_fci:', E_fci)
hams.append(ham)

molecule = [
  ['H', [0, 0, -1.5]],
  ['H', [0, 0, -0.5]],
  ['H', [0, 0,  0.5]],
  ['H', [0, 0,  1.5]],
]
ham, (E_hf, E_fci) = get_ham_and_E(molecule)
print('E_hf:', E_hf, 'E_fci:', E_fci)
hams.append(ham)

print('diff 0-1:', (hams[0] - hams[1]).compress(1e-8))
print('diff 0-2:', (hams[0] - hams[2]).compress(1e-8))
print('diff 1-2:', (hams[1] - hams[2]).compress(1e-8))
