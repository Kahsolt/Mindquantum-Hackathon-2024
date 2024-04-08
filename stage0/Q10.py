#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/08

from mindquantum import *

# 请补充如下函数
def exp_of_zz(res):
  # hams: Z0 Z1
  probs = {k: (freq / res.shots) for k, freq in res.data.items()}
  for key in ['00', '01', '10', '11']:
    probs.setdefault(key, 0.0)
  exp = probs['00'] - probs['01'] - probs['10'] + probs['11']
  #exp = (probs['00'] - probs['11']) * 2
  return exp

# 以下为校验部分
shots = 10000
circ = Circuit().rx(1.0, 0).ry(2.0, 1, 0).measure_all()
sim = Simulator('mqvector', circ.n_qubits)
res = sim.sampling(circ, shots=shots, seed=42)
expectation = exp_of_zz(res)
print(expectation)



print('=' * 72)

# ref: https://hiq.huaweicloud.com/tutorial/quantum_measurement - Pauli测量
# Z基测量基本等于计算基测量，只是值域 pmesaure [0, 1] => z-pauli exp [1, -1] => [|0>, |1>]
# 对于多个Qubit的联合Z基测量，虽然教程里说是加和形式，但其实应该是乘积形式 :)
# Z = [
#   [1,  0],
#   [0, -1],
# ]
# 乘积/交: <ψ|Z⊗Z|ψ> = <ψ|M00|ψ> - <ψ|M01|ψ> - <ψ|M10|ψ> + <ψ|M11|ψ>
circ = Circuit().rx(1.0, 0).ry(2.0, 1, 0)
ham = Hamiltonian(QubitOperator('Z0 Z1'))
exp = sim.get_expectation(ham, circ)
print(exp)
# 加和/并: <ψ|Z⊗I+I⊗Z|ψ> = (<ψ|M00|ψ> - <ψ|M11|ψ>) * 2
circ = Circuit().rx(1.0, 0).ry(2.0, 1, 0)
ham = Hamiltonian(QubitOperator('Z0') + QubitOperator('Z1'))
exp = sim.get_expectation(ham, circ)
print(exp)
