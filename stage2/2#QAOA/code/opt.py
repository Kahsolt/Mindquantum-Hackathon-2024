for iter in range(n_samples):
  # 优化前的参数 & 能量
  params, E_before = ..., rc()
  # 优化线路!
  optimize(circ)
  # 优化后的参数 & 能量
  tuned_params, E_after = ..., circ()
  # 损失值自适应 + 学习率衰减
  ΔE = E_before - E_after
  lr = dx * (dx_decay ** (iter / dx_decay_every)) * log1p(ΔE)
  # 更新动量状态 & 预制表参数
  mu = (1 - m) * mu + m * tuned_params
  theta = (1 - lr) * theta + lr * mu
