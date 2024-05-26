#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/25

from pathlib import Path
from argparse import ArgumentParser

import torch
import matplotlib .pyplot as plt
import seaborn as sns


def vis(args):
  ckpt = torch.load(args.load, map_location='cpu')
  model = ckpt['model']
  U_64 = model['U_λ_res_64']
  U_128 = model['U_λ_res_128']
  if Path(args.load).stem.startswith('pReg'):
    U_64 = U_64 @ U_64.T
    U_128 = U_128 @ U_128.T
  
  plt.subplot(121) ; sns.heatmap(U_64)  ; plt.title('U_64')
  plt.subplot(122) ; sns.heatmap(U_128) ; plt.title('U_128')
  plt.show()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--load', required=True, help='path to *.pth')
  args = parser.parse_args()

  vis(args)
