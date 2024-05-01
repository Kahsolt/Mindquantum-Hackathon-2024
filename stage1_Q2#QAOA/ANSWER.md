# 标准QAOA线路的最优初始化参数

### problem

- 作答约束
  - 实现 [main.py](main.py) 脚本，入口函数签名 `main(Jc_dict:Dict[Tuple[int], float], p:int, Nq:int=12) -> Tuple[ndarray, ndarray]`
  - 禁止用目标函数对参数进行迭代优化 (i.e. non-adaptive)
- 评测条件
  - 用例: 随机 2~4 阶 Ising 模型，深度为 p=4/8 的标准 QAOA 线路
  - 评分: 线路末态哈密顿量的期望 C_d↓
  - 限时: 30min
- 考点
  - 量子逻辑线路初参选取，优化问题初始解选取
  - 递归/动态规划思想，大规模问题局部复用小规模问题的初始化参数
  - 预训练知识，总体规律
  - 打一套精度更高的表 / 表生成算法

### solution

- run `python score.py`

⚪ baselines

| method | score↑ | time | comment |
| :-: | :-: | :-: | :-: |
| PT-WMC[1] | 11730.14583 | 101.60 | reference, use $ γ^{inf} $ in stead of $ γ^{median} $ |
| PS-WP[2]  | 16526.79871 | 101.72 | baseline |

⚪ submits

| datetime | local score↑ | submit score↑ | comment |
| :-: | :-: | :-: | :-: |
| 2024-05-01 17:02:50 | 11730.14583 | 16627.496  | reference |
| 2024-05-01 16:44:12 | 16526.79871 | 24999.9025 | baseline  |

### reference

- [1] Parameter Transfer for Quantum Approximate Optimization of Weighted MaxCut (2023): [https://arxiv.org/abs/2201.11785](https://arxiv.org/abs/2201.11785)
- [2] Parameter Setting in Quantum Approximate Optimization of Weighted Problems (2024): [https://arxiv.org/abs/2305.15201](https://arxiv.org/abs/2305.15201)
- [3] The Quantum Approximate Optimization Algorithm at High Depth for MaxCut on Large-Girth Regular Graphs and the Sherrington-Kirkpatrick Model (2022): [https://arxiv.org/abs/2110.14206](https://arxiv.org/abs/2110.14206)

----
by Armit
2024/04/30
