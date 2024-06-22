#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/06/22 

# 查看训练数据中边的权值分布的方差

import json
import numpy as np
import matplotlib.pyplot as plt

distro_std = {
  'std': [],
  'uni': [],
  'bimodal': [],
}
for propotion in [0.3, 0.6, 0.9]:
  for k in range(2, 6):
    for coef in ['std', 'uni', 'bimodal']:
      for r in range(10):
        with open(f"data/k{k}/{coef}_p{propotion}_{r}.json") as fh:
          vals = json.load(fh)['c']
        distro_std[coef].append(np.asarray(vals).std())

plt.subplot(131) ; plt.hist(distro_std['bimodal'],                     bins=50, label='bimodal') ; plt.legend()
plt.subplot(132) ; plt.hist(distro_std['uni'],                         bins=50, label='uniform') ; plt.legend()
plt.subplot(133) ; plt.hist(distro_std['uni'] + distro_std['bimodal'], bins=50, label='mixed') ; plt.legend()
plt.tight_layout()
plt.show()
