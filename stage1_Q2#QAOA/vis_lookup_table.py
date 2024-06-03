#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/06/03 

from pathlib import Path

from utils.path import LOG_PATH, IMG_PATH
from utils.lookup_table import load_lookup_table, plot_lookup_table


def walk(dp:Path):
  for fp in dp.iterdir():
    if fp.is_dir(): walk(fp)
    if fp.suffix == '.json':
      print(f'>> processing: {fp}')
      lookup_table = load_lookup_table(fp)
      rfp = fp.relative_to(LOG_PATH)
      save_dp = IMG_PATH / str(rfp)[:-len(rfp.suffix)]
      save_dp.mkdir(exist_ok=True, parents=True)
      plot_lookup_table(lookup_table, save_dp)

walk(LOG_PATH)
