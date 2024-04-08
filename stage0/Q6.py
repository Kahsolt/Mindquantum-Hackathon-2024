#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/08

from mindquantum.algorithm.nisq import RYCascade

# 请补充如下函数
def ansatz_circuit():
  circ = RYCascade(n_qubits=4, depth=1).circuit
  return circ

# 以下为校验部分
circ = ansatz_circuit()
print(circ)
