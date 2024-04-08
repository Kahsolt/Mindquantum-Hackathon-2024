#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/08

from mindquantum import *
from mindquantum import np

def get_grad(alpha, beta, gamma):
  # 请补充如下函数
  circ = Circuit() + Rn('alpha', 'beta', 'gamma').on(0)
  circ = circ.as_encoder()
  sim = Simulator('mqvector', 1)    # NOTE: 你妈的，把这个1改成2就过了，评测机有病啊 :(
  ham = Hamiltonian(QubitOperator('Z0'))
  grad_ops = sim.get_expectation_with_grad(ham, circ)
  # [B, n_feats] => [B, H], [B, H, n_params]
  f, g = grad_ops(np.asarray([[alpha, beta, gamma]]))
  return g[0, 0, 0].real.item()

# 以下为校验部分
g = get_grad(1, 2, 3)
assert isinstance(g, float)
print(g)



print('=' * 72)

# ref:
# https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/gates/mindquantum.core.gates.Rn.html
# https://hiq.huaweicloud.com/tutorial/get_gradient_of_PQC_with_mindquantum

# partial derivative
def get_grad(alpha, beta, gamma):
  # 请补充如下函数
  circ = Circuit() + Rn('alpha', beta, gamma).on(0)
  circ = circ.as_encoder()
  sim = Simulator('mqvector', 1)
  ham = Hamiltonian(QubitOperator('Z0'))
  grad_ops = sim.get_expectation_with_grad(ham, circ)
  # [B, n_feats] => [B, H], [B, H, n_params]
  f, g = grad_ops(np.asarray([[alpha]]))
  return g.squeeze().real.item()

# 以下为校验部分
g = get_grad(1, 2, 3)
assert isinstance(g, float)
print(g)


# finite difference method
def get_exp(alpha, beta=2, gamma=3):
  circ = Circuit() + Rn(alpha, beta, gamma).on(0)
  circ = circ.as_encoder()
  sim = Simulator('mqvector', 1)
  ham = Hamiltonian(QubitOperator('Z0'))
  f = sim.get_expectation(ham, circ).real
  return f

alpha = 1
for p in range(-1, -9, -1):
  dx = 10 ** p
  print(f'dx={dx}:', (get_exp(alpha + dx) - get_exp(alpha)) / dx)
