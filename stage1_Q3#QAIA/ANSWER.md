# 基于QAIA的信号解码

### problem

- 作答约束
  - 实现 [main.py](main.py) 脚本，入口函数签名
    - 构造 Ising 问题: `ising_generator(H:ndarray, y:ndarray, num_bits_per_symbol:int, snr:float) -> Tuple[ndarray, ndarray]`
    - 求解 Ising 问题: `qaia_mld_solver(J:ndarray, h:ndarray) -> ndarray`
- 评测条件
  - 评分: 比特错误率 BER↓
  - 限时: 1h
- 考点
  - MLD 问题转换为 Ising 问题
  - 改进 QAIA 算法，例如基于 SB 系列发展，或混合各类算法

### solution

- run `python judger.py`

⚪ baselines

| method | BER↓ | time | comment |
| :-: | :-: | :-: | :-: |
| ZF     | 0.41120 | -      | reference |
| NMFA   | 0.38561 | 230.66 |  |
| SimCIM | 0.23271 | 232.51 |  |
| CAC    | 0.31591 | 279.79 |  |
| CFC    | 0.23801 | 279.84 |  |
| SFC    | 0.23796 | 278.40 |  |
| ASB    | 0.34054 | 253.43 | dt=0.1 (default dt=1 doesn't run) |
| DSB    | 0.28741 | 230.11 |  |
| BSB[1] | 0.21584 | 135.38 | baseline (B=100, n_iter=100) |
| LQA[2] | 0.20627 | 229.35 | best, but too classical |
| LQA[2] | 0.21652 | 134.27 | B=100, n_iter=50  |
| LQA[2] | 0.21379 | 719.15 | B=100, n_iter=300 |
| LQA[2] | 0.20624 | 146.68 | B=50,  n_iter=100 |
| LQA[2] | 0.20400 | 710.05 | B=300, n_iter=100 |

⚪ submits

| datetime | local BER↓ | submit score↑ | comment |
| :-: | :-: | :-: | :-: |
| 2024-04-30 21:12:31 | 0.20400 | 0.7969 | LQA (B=300) |

### reference

- [1] High-performance combinatorial optimization based on classical mechanics (2021): [https://www.researchgate.net/publication/349022706_High-performance_combinatorial_optimization_based_on_classical_mechanics](https://www.researchgate.net/publication/349022706_High-performance_combinatorial_optimization_based_on_classical_mechanics)
- [2] Quadratic Unconstrained Binary Optimization via Quantum-Inspired Annealing (2022): [https://www.researchgate.net/publication/363382279_Quadratic_Unconstrained_Binary_Optimization_via_Quantum-Inspired_Annealing](https://www.researchgate.net/publication/363382279_Quadratic_Unconstrained_Binary_Optimization_via_Quantum-Inspired_Annealing)
- [3] Ising Machines' Dynamics and Regularization for Near-Optimal Large and Massive MIMO Detection (2021): [https://arxiv.org/abs/2105.10535](https://arxiv.org/abs/2105.10535)
- [4] Deep Unfolded Simulated Bifurcation for Massive MIMO Signal Detection (2023): [https://arxiv.org/abs/2306.16264](https://arxiv.org/abs/2306.16264)
- [5] Uplink MIMO Detection using Ising Machines: A Multi-Stage Ising Approach (2023): [https://arxiv.org/abs/2304.12830](https://arxiv.org/abs/2304.12830)

----
by Armit
2024/04/30
