#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/08

# 验证 Q7_verify.md 里手算的公式对不对

from mindquantum import *
from mindquantum import np


def verify_fval(alpha=1, beta=2, gamma=3):
  ''' Verify the expectation function value E(α,β=2,γ=3) '''

  # computed expectation by mindqunatum engine thouroghly
  sim = Simulator('mqvector', 1)
  ham = Hamiltonian(QubitOperator('Z0'))
  circ = Circuit() + Rn(alpha, beta, gamma).on(0)
  fval_mq = sim.get_expectation(ham, circ).real
  print('f-val mq:', fval_mq)

  # computed qstate by mindqunatum engine, then get expectation with formula
  sim = Simulator('mqvector', 1)
  sim.apply_gate(Rn(alpha, beta, gamma).on(0))
  qs = sim.get_qs()
  Z = np.asarray([
    [1, 0],
    [0, -1],
  ])
  fval_mq_math = (qs.conj().T @ Z @ qs).real
  print('f-val mq+math:', fval_mq_math)

  # computed through pure math formula
  def fval_formula(alpha):    # E_\alpha(\alpha)
    t = np.sqrt(alpha**2 + 13)
    return 1 - (2 - 18 / t**2) * np.sin(t/2)**2

  fval_math = fval_formula(alpha)
  print('f-val math:', fval_math)

  assert np.allclose(fval_mq, fval_mq_math, atol=1e-5), 'bad fval_mq_math'
  assert np.allclose(fval_mq, fval_math, atol=1e-5), 'bad fval_math'


def verify_gval(alpha=1, beta=2, gamma=3):
  ''' Verify the expectation function gradient value dE(α,β=2,γ=3)/dα '''

  # computed gradient by mindqunatum engine thouroghly
  sim = Simulator('mqvector', 1)
  ham = Hamiltonian(QubitOperator('Z0'))
  circ = Circuit() + Rn('alpha', beta, gamma).on(0)
  grad_ops = sim.get_expectation_with_grad(ham, circ)
  _, g = grad_ops([alpha])
  gval_mq = g.squeeze().real.item()
  print('g-val mq:', gval_mq)

  # computed through pure math formula
  def gval_formula(alpha):    # dE_\alpha(\alpha) / d\alpha
    t = np.sqrt(alpha**2 + 13)
    a = -36 * alpha / t**4 * np.sin(t/2)**2
    b = (18 / t**2 - 2) * np.sin(t/2) * np.cos(t/2) * alpha / t
    return a + b

  gval_math = gval_formula(alpha)
  print('g-val math:', gval_math)

  assert np.allclose(gval_mq, gval_math, atol=1e-5), 'bad gval_math'


if __name__ == '__main__':
  for alpha in np.linspace(-2*np.pi, 2*np.pi, 360):
    verify_fval(alpha)
    verify_gval(alpha)

  print('>> test passed')

  verify_fval(1)
  verify_gval(1)
