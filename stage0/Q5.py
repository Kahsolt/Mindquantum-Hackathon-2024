#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/08

from mindquantum import *

circ = Circuit()
circ += H.on(0)
circ += RX(0.1).on(1)
circ += RY(0.1).on(1, ctrl_qubits=0)

print(circ)

# X0 Y1
ham = Hamiltonian(QubitOperator('X0 Y1'))

sim = Simulator('mqvector', 2)
expectation = sim.get_expectation(ham, circ)

print(expectation)
