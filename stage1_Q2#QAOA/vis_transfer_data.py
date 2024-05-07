#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/07 

import csv
from pathlib import Path
from typing import Dict

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

BASE_PATH = Path(__file__).parent
IMG_PATH = BASE_PATH / 'img' ; IMG_PATH.mkdir(exist_ok=True)
LOOKUP_FILE = 'utils/transfer_data.csv'

LookupTable = Dict[int, Dict[int, ndarray]]   # p (depth) -> k (order) -> params


def load_lookup_table() -> LookupTable:
  lookup_table = {}
  with open(LOOKUP_FILE, 'r') as csv_file:
    reader = csv.reader(csv_file)
    rows = [row for row in reader][2:]   # ignore header
    for row in rows:
      q, p, opt_nu, *params = [e for e in row if e]
      k = int(q)    # order
      p = int(p)    # depth
      opt_nu = float(opt_nu)
      if p not in lookup_table:
        lookup_table[p] = {}
      lookup_table[p][k] = np.asarray(params, dtype=np.float32)
  return lookup_table


def plot_lookup_table(lookup_table:LookupTable, subfolder='original'):
  save_dp = IMG_PATH / subfolder
  save_dp.mkdir(exist_ok=True)

  for p in sorted(lookup_table):
    plt.clf()
    for k in sorted(lookup_table[p]):
      plt.plot(lookup_table[p][k], label=f'{k}')
    plt.legend()
    plt.suptitle(f'p={p}')
    plt.tight_layout()
    plt.savefig(save_dp / f'p={p}.png', dpi=400)
    plt.close()


if __name__ == '__main__':
  lookup_table = load_lookup_table()
  plot_lookup_table(lookup_table)
