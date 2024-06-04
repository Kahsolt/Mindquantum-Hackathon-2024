#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/06/03 

# 查看 CHC 线路结构

from qiskit_nature.second_q.circuit.library.ansatzes.utils import generate_vibration_excitations
from qiskit_nature.second_q.circuit.library.ansatzes.chc import CHC

num_qubits = 8
num_particles = (4, 4)
excitations = [
  *generate_vibration_excitations(1, num_particles),
  *generate_vibration_excitations(2, num_particles),
]
ansatz = CHC(num_qubits, excitations)

# 对于含噪环境，这个线路可能还是太深了。。。
print()
print(ansatz)
print('len(ansatz):', len(ansatz))  # => 228
print('len(parameters):', ansatz.num_parameters) # => 15
