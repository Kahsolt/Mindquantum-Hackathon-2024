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
  - 基线代码已经实现 `arXiv:2105.10535 [5]`，应该是希望选手进一步实现 `arXiv:2306.16264 [6]`

### solution

- run `python judger.py`

⚪ baselines (classical)

| method | BER↓ | comment |
| :-: | :-: | :-: |
| linear-zf-maxlog    | 0.41121 | match with the reference :) |
| linear-zf-app       | 0.40605 |  |
| linear-mf-maxlog    | 0.34104 |  |
| linear-mf-app       | 0.33401 |  |
| linear-lmmse-maxlog | 0.20779 |  |
| linear-lmmse-app    | 0.20721 | very fast; app > maxlog, lmmse > mf > zf |
| kbest-k=16          | 0.27594 | very slow |
| kbest-k=32          | 0.26785 |  |
| kbest-k=64          | 0.26206 |  |
| ep-iter=5           | 0.16968 | fast and nice! |
| ep-iter=10          | 0.16872 |  |
| ep-iter=20          | 0.16887 |  |
| ep-iter=40          | 0.16889 |  |
| mmse-iter=1         | 0.19903 | mmse is all cheaty, only for seeking BER lower bound!! |
| mmse-iter=2         | 0.16875 |  |
| mmse-iter=4         | 0.15921 |  |
| mmse-iter=8         | 0.15738 |  |
| mmse-iter=16        | 0.15768 |  |

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
| BSB[1] | 0.21584 | 135.38 | baseline[5] (B=100, n_iter=100) |
| LQA[2] | 0.20627 | 229.35 | best, but too classical |
| LQA[2] | 0.21652 | 134.27 | B=100, n_iter=50  |
| LQA[2] | 0.21379 | 719.15 | B=100, n_iter=300 |
| LQA[2] | 0.20624 | 146.68 | B=50,  n_iter=100 |
| LQA[2] | 0.20400 | 710.05 | B=300, n_iter=100 |

⚪ submits

| datetime | local BER↓ | submit score↑ | comment |
| :-: | :-: | :-: | :-: |
| 2024-04-30 21:12:31 | 0.20400 | 0.7969 | LQA (B=300) |
| 2024-05-02 21:25:46 | 0.21442 | 0.7834 | baseline (B=300) |

### dataset

```
[H] {(64, 64): 75, (128, 128): 75}
[y] {(64, 1): 75, (128, 1): 75}
[bits] {(64, 4): 30, (128, 6): 30, (64, 8): 15, (128, 8): 15, (128, 4): 30, (64, 6): 30}
[num_bits_per_symbol] {4: 60, 6: 60, 8: 30}
[SNR] {10: 50, 15: 50, 20: 50}

BER wrt. each param groups under baseline setting:
>> avgber_per_Nt:
  64: 0.21421875
  128: 0.21439670138888892
>> avgber_per_snr:
  10: 0.2542643229166667
  15: 0.20962239583333336
  20: 0.17903645833333337
>> avgber_per_nbps:
  4: 0.13942057291666668
  6: 0.2408637152777778
  8: 0.31097005208333334
>> time cost: 134.46
>> avg. BER = 0.21431
```

### reference

- [1] High-performance combinatorial optimization based on classical mechanics (2021): [https://www.researchgate.net/publication/349022706_High-performance_combinatorial_optimization_based_on_classical_mechanics](https://www.researchgate.net/publication/349022706_High-performance_combinatorial_optimization_based_on_classical_mechanics)
- [2] Quadratic Unconstrained Binary Optimization via Quantum-Inspired Annealing (2022): [https://www.researchgate.net/publication/363382279_Quadratic_Unconstrained_Binary_Optimization_via_Quantum-Inspired_Annealing](https://www.researchgate.net/publication/363382279_Quadratic_Unconstrained_Binary_Optimization_via_Quantum-Inspired_Annealing)
- [3] Leveraging Quantum Annealing for Large MIMO Processing in Centralized Radio Access Networks (2020): [https://arxiv.org/abs/2001.04014](https://arxiv.org/abs/2001.04014)
- [4] Physics-Inspired Heuristics for Soft MIMO Detection in 5G New Radio and Beyond (2021): [https://arxiv.org/abs/2103.10561](https://arxiv.org/abs/2103.10561)
- [5] Ising Machines' Dynamics and Regularization for Near-Optimal Large and Massive MIMO Detection (2021): [https://arxiv.org/abs/2105.10535](https://arxiv.org/abs/2105.10535)
- [6] Deep Unfolded Simulated Bifurcation for Massive MIMO Signal Detection (2023): [https://arxiv.org/abs/2306.16264](https://arxiv.org/abs/2306.16264)
- [7] Uplink MIMO Detection using Ising Machines: A Multi-Stage Ising Approach (2023): [https://arxiv.org/abs/2304.12830](https://arxiv.org/abs/2304.12830)
- [8] Simulated Bifurcation Algorithm for MIMO Detection (2022): [https://arxiv.org/abs/2210.14660](https://arxiv.org/abs/2210.14660)
- [9] Sionna: library for simulating the physical layer of wireless and optical communication systems
  - repo: [https://github.com/NVlabs/sionna](https://github.com/NVlabs/sionna)
  - doc: [https://nvlabs.github.io/sionna/index.html](https://nvlabs.github.io/sionna/index.html)
- [10] REM DUIDD: Deep-Unfolded Interleaved Detection and Decoding for MIMO Wireless Systems (2022): [https://arxiv.org/abs/2212.07816](https://arxiv.org/abs/2212.07816)
  - repo: [https://github.com/IIP-Group/DUIDD](https://github.com/IIP-Group/DUIDD)

----
by Armit
2024/04/30
