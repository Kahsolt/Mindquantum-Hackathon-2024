# 含噪条件下的变分量子算法求H4分子基态

### problem

- 作答约束
  - 实现 [solution.py](solution.py) 脚本，入口函数签名 `solution(molecule, Simulator: HKSSimulator) -> float`
  - 限制使用门集合 {X,CNOT,Y,Z,H,CZ,RX,RY,RZ,Measure,Barrier}
  - 限制测量方式为求 pauli 串的哈密顿量期望
  - 已知噪声模型
- 评测条件
  - 提交: 仅 solution.py 文件
  - 用例: 各种键长的 `H4` 分子
  - 评分: L1_err↓, n_shots↓
  - 限时: 2h
- 考点
  - pauli grouping/trimming (测量尽可能少的pauli串，平衡coeff数值精度)
  - light-weight ansatz design (用尽可能少的门)
  - error mitigate (去除空线路噪声/ZNE/多测几次取最低)

### solution

⚠ only run on Linux, due to dependecies of `openfermion` and `openfermionpyscf`

- run `python solver.py`

⚪ baselines

> truth $ E_{fci} $: -2.1663874486347625 for the default H4 1-2-3-4

| method | score↑ | energy↓ | shots↓ | time | comment |
| :-: | :-: | :-: | :-: | :-: | :-: |
| UCC |  0.290 | -0.29369 | 18400 | 324.23 | baseline |
| UCC |  0.300 | -0.31272 | 18000 | 314.37 | trim coeff < 1e-3 |
| UCC |  1.035 | -0.37762 |  5400 | 284.07 | trim coeff < 1e-3, shot=30 (随机性很大) |
| UCC |  3.888 | -0.59817 |  1640 | 251.29 | trim coeff < 1e-2, shot=10 (随机性很大) |
| UCC | 22.612 | -0.15622 |   220 |  35.40 | trim coeff < 1e-1, shot=10 (随机性很大) |

⚪ submits

| datetime | local score↑ | submit score↑ | comment |
| :-: | :-: | :-: | :-: |
| 2024-05-01 20:38:16 | 1.035 | 0.5226 | baseline, trim coeff < 1e-3, shot=30 |
| 2024-05-02 23:20:24 | 1.334 | 0.574  | ry_HEA, trim coeff < 1e-3, shots=100 |
| 2024-05-02 23:28:58 | 2.360 | 0.7994 | HF, trim coeff < 1e-3, shots=100 |
| 2024-05-05 23:15:30 | 2.360 | 1.0077 | HF, trim coeff < 1e-3, shots=100 |

### reference

- solution from Tencent Quantum Lab: [https://github.com/liwt31/QC-Contest-Demo](https://github.com/liwt31/QC-Contest-Demo)

----
by Armit
2024/04/30
