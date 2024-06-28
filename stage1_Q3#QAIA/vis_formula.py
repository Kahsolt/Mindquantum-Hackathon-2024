#!/usr/bin/env python3
# Author: Armit
# Create Time: 周六 2024/06/29 

# 当提交次数足够多的时候，我们应该可以推测出评分公式中的常数 :)

import numpy as np
import matplotlib.pyplot as plt

data = [
  # (local BER, local time, local score, submit score)
  #(0.39115, 2.87,  49.7056,   0.6092),
  #(0.22977, 5.09,  35.4083,   0.7706), 
  (0.19238, 8.82,  21.4311,  20.3597),
  (0.21313, 5.27,  34.9513,  33.8763),
  (0.21345, 2.72,  67.5861,  92.4872),
  (0.21805, 1.37, 133.5814, 105.2462),
  (0.20907, 1.54, 120.5792, 101.1801),
  (0.19932, 2.73,  68.5260,  89.3023),
  (0.20085, 3.32,  56.4051,  87.0405),
  (0.20049, 2.89,  64.8713,  73.7797),
  (0.19940, 2.77,  67.6817,  92.9875),
  (0.15602, 3.60,  54.9289,  95.1233),
  (0.21701, 2.95,  62.0686,  89.8824),
  (0.15490, 1.26, 156.9497, 137.3652),
]
local_BER    = [it[0] for it in data]
local_time   = [it[1] for it in data]
local_score  = [it[2] for it in data]
submit_score = [it[3] for it in data]
nlen = len(local_BER)

BER_score    = sorted(zip(local_BER,   submit_score))
time_score   = sorted(zip(local_time,  submit_score))
score_submit = sorted(zip(local_score, submit_score))

plt.clf()
plt.subplot(311) ; plt.plot([it[0] for it in BER_score],    'b') ; ax = plt.twinx() ; ax.plot([it[1] for it in BER_score],    'r') ; plt.title('BER')
plt.subplot(312) ; plt.plot([it[0] for it in time_score],   'b') ; ax = plt.twinx() ; ax.plot([it[1] for it in time_score],   'r') ; plt.title('time')
plt.subplot(313) ; plt.plot([it[0] for it in score_submit], 'b') ; ax = plt.twinx() ; ax.plot([it[1] for it in score_submit], 'r') ; plt.title('local score')
plt.tight_layout()
plt.show()
