#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/01

from pathlib import Path
from dataclasses import dataclass
import pickle as pkl
from typing import *

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / 'MLD_data'
IMG_PATH = BASE_PATH / 'img' ; IMG_PATH.mkdir(exist_ok=True)


@dataclass
class Sample:
  Nr: int                    # 64/128
  Nt: int                    # 64/128
  num_bits_per_symbol: int   # 4/6/8; aka. M
  H: ndarray                 # [Nr, Nt], complex64
  y: ndarray                 # [Nr, 1], complex64
  bits: ndarray              # [Nr, M], vset {0, 1}, float32
  SNR: int                   # 10/15/20
  ZF_ber: float

  @property
  def x(self) -> ndarray:
    # x ~= H' * y (?)
    breakpoint()
    return np.linalg.inv(self.H) @ self.y


def get_dataset() -> List[Sample]:
  fps = list(DATA_PATH.iterdir())
  fps.sort(key=lambda e: int(e.stem))
  return [Sample(**pkl.load(open(fp, 'rb'))) for fp in fps]


def plot_hist(features:List[ndarray], title:str='feat'):
  real  = np.concatenate([it.real      for it in features]).flatten()
  imag  = np.concatenate([it.imag      for it in features]).flatten()
  mag   = np.concatenate([np.abs  (it) for it in features]).flatten()
  phase = np.concatenate([np.angle(it) for it in features]).flatten()

  plt.subplot(221) ; plt.hist(real,  bins=300) ; plt.title('real')
  plt.subplot(222) ; plt.hist(imag,  bins=300) ; plt.title('imag')
  plt.subplot(223) ; plt.hist(mag,   bins=300) ; plt.title('mag')
  plt.subplot(224) ; plt.hist(phase, bins=300) ; plt.title('phase')
  plt.suptitle(title)
  fp = IMG_PATH / f'{title}.png'
  print(f'>> save to {fp}')
  plt.savefig(fp, dpi=600)
  plt.close()


if __name__ == '__main__':
  dataset = get_dataset()

  y = [it.y for it in dataset]
  plot_hist(y, title='y')

  for Nr in [64, 128]:
    y = [it.y for it in dataset if it.Nr == Nr]
    plot_hist(y, title=f'y-Nr={Nr}')
  for SNR in [10, 15, 20]:
    y = [it.y for it in dataset if it.SNR == SNR]
    plot_hist(y, title=f'y-SNR={SNR}')

  x = [it.x for it in dataset]
  plot_hist(x, title='x')

  for M in [4, 6, 8]:
    x = [it.x for it in dataset if it.num_bits_per_symbol == M]
    plot_hist(x, title=f'x-M={M}')
  for SNR in [10, 15, 20]:
    y = [it.x for it in dataset if it.SNR == SNR]
    plot_hist(y, title=f'x-SNR={SNR}')
