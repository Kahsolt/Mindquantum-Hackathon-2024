#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/08

from mindquantum import *

# 请补充如下函数
def build_noise_model():
  # NOTE: this seem not really wise though..
  noise_model = SequentialAdder([
    MixerAdder([
      GateSelector('X'),
      DepolarizingChannelAdder(p=1/20, n_qubits=1),
    ]),
    MixerAdder([
      GateSelector('Y'),
      DepolarizingChannelAdder(p=1/20, n_qubits=1),
    ]),
    MixerAdder([
      GateSelector('CX'),
      DepolarizingChannelAdder(p=1/100, n_qubits=2),
    ])
  ])
  return noise_model

# 以下为校验部分
circ = Circuit().x(0).y(1).x(1, 0).measure_all()
noise_model = build_noise_model()
noise_circ = noise_model(circ)
print(noise_circ)
