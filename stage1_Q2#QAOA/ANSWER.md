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
    - beta ↘: H0/H_B related
    - gamma ↗: H1/H_C related
  - 打一套精度更高的表 / 表生成算法
    - beta: 看似已经最优，不用做任何处理
    - gamma: 讨论放缩系数 (逐层不同，condition到问题配置)

### solution

- run `python score.py`

⚪ submits

ℹ Possible max local score: 23654.99239

| datetime | local score↑ | submit score↑ | comment |
| :-: | :-: | :-: | :-: |
| 2024-05-01 17:02:50 | 11730.14583 | 16627.496  | PT-WMC[1], reference, use $ γ^{inf} $ in stead of $ γ^{median} $ |
| 2024-05-01 16:44:12 | 16526.79871 | 24999.9025 | PS-WP[2], baseline  |
| 2024-05-06 23:09:02 | 17816.62534 | 27217.3001 | baseline, rescaler=1.275 |
|                     | 18516.17254 |            | opt-avg, iter=10000, rescaler=1.0 |
|                     | 16150.62844 |            | opt-avg, iter=10000, test with rescaler=1.275 |
|                     | 18089.52401 |            | ft; iter=100, rescaler=1.275 |
| 2024-05-07 18:28:59 | 18302.26568 | 27756.869  | ft; iter=200, rescaler=1.275 |
| 2024-05-07 18:42:49 | 18183.73663 | 27645.5604 | ft; iter=300, rescaler=1.275 |
|                     | 18181.98281 |            | ft; iter=400, rescaler=1.275 |
| 2024-05-07 19:35:12 | 18535.15152 | 27998.4487 | ft; iter=500, rescaler=1.275 |
|                     | 18452.12593 |            | ft; iter=500 |
|                     | 17510.16457 |            | ft; iter=1000, rescaler=1.275 |
|                     | 18911.62258 |            | ft-ada; iter=100, rescaler=1.275 |
|                     | 19611.28956 |            | ft-ada; iter=300, rescaler=1.275 |
|                     | 20027.51415 |            | ft-ada; iter=400, rescaler=1.275 |
|                     | 18695.67729 |            | ft-ada; iter=500, rescaler=1.275 |
|                     | 18203.21215 |            | ft-ada; iter=600, rescaler=1.275 |
|                     | 18746.89745 |            | ft-ada; iter=800, rescaler=1.275 |
|                     | 18739.11156 |            | ft-ada; iter=900, rescaler=1.275 |
|                     | 19269.59404 |            | ft-ada; iter=1000, rescaler=1.275 |
|                     | 18414.86418 |            | ft-ada; iter=1300, rescaler=1.275 |
| 2024-05-31 15:45:22 | 20489.27983 | 29920.0329 | ft-ada; iter=1400, rescaler=1.275 |
| 2024-05-31 15:11:41 | 20369.67411 | 29777.4787 | ft-ada; iter=1500, rescaler=1.275 |
|                     | 20330.73329 |            | ft-ada; iter=1900, rescaler=1.275 |
|                     | 20116.10485 |            | ft-ada; iter=2000, rescaler=1.275 |
|                     | 19226.65152 |            | ft-ada-decay; iter=2000, rescaler=1.275 |
| 2024-06-07 08:58:31 | 20368.31217 | 29747.6604 | ft-ada-decay; iter=3000, rescaler=1.275 |
| 2024-06-07 08:59:38 | 20175.90654 | 29554.6544 | ft-ada-decay; iter=4000, rescaler=1.275 |
| 2024-06-03 13:15:20 | 20696.82719 | 30072.2855 | ft-ada-decay; iter=5000, rescaler=1.275 |
| 2024-06-09 14:49:42 | 20726.71913 | 30100.7901 | ft-ada-decay; iter=5500, rescaler=1.275 |
| 2024-06-03 12:40:58 | 20729.63455 | 30105.3792 | ft-ada-decay; iter=5600, rescaler=1.275 |
|                     | 20802.38676 |            | ft-ada-decay; iter=6000, rescaler=1.275 |
|                     | 20775.81853 |            | ft-ada-decay; iter=6100, rescaler=1.275 |
|                     | 20703.96950 |            | ft-ada-decay; iter=6400, rescaler=1.275 |
|                     | 20711.00992 |            | ft-ada-decay; iter=6500, rescaler=1.275 |
|                     | 20797.86140 |            | ft-ada-decay; iter=6700, rescaler=1.275 |
|                     | 20783.31786 |            | ft-ada-decay; iter=6900, rescaler=1.275 |
|                     | 20814.45155 |            | ft-ada-decay; iter=7000, rescaler=1.275 |
|                     | 20856.19912 |            | ft-ada-decay; iter=7100, rescaler=1.275 |
|                     | 20778.53995 |            | ft-ada-decay; iter=7200, rescaler=1.275 |
|                     | 20792.42218 |            | ft-ada-decay; iter=8000, rescaler=1.275 |
|                     | 20824.90335 |            | ft-ada-decay; iter=9000, rescaler=1.275 |
|                     | 20855.28777 |            | ft-ada-decay; iter=9300, rescaler=1.275 |
| 2024-06-11 11:10:03 | 20948.34952 | 30332.7544 | ft-ada-decay; iter=9400, rescaler=1.275 |
|                     | 20907.75227 |            | ft-ada-decay; iter=9500, rescaler=1.275 |
|                     | 20863.27086 |            | ft-ada-decay; iter=9600, rescaler=1.275 |
|                     | 20809.78942 |            | ft-ada-decay; iter=9700, rescaler=1.275 |
|                     | 20827.82030 |            | ft-ada-decay; iter=10000, rescaler=1.275 |
|                     | 18027.36405 |            | opt-avg; iter=1000, rescaler=1.275 |

### reference

- [1] Parameter Transfer for Quantum Approximate Optimization of Weighted MaxCut (2023): [https://arxiv.org/abs/2201.11785](https://arxiv.org/abs/2201.11785)
- [2] Parameter Setting in Quantum Approximate Optimization of Weighted Problems (2024): [https://arxiv.org/abs/2305.15201](https://arxiv.org/abs/2305.15201)
- [3] The Quantum Approximate Optimization Algorithm at High Depth for MaxCut on Large-Girth Regular Graphs and the Sherrington-Kirkpatrick Model (2022): [https://arxiv.org/abs/2110.14206](https://arxiv.org/abs/2110.14206)

----
by Armit
2024/04/30
