def ber_loss(spins:Tensor, bits:Tensor, loss_fn:str='mse'):
  # sionna => constellation style
  bits_constellation = 1 - concat([bits[..., 0::2], bits[..., 1::2]], dim=-1)
  # Fig. 2 from arXiv:2001.04014
  nbps = bits_constellation.shape[1]
  rb = nbps // 2
  spins = reshape(spins, (rb, 2, -1))  # [rb, c=2, Nt]
  spins = permute(spins, (2, 1, 0))    # [Nt, c=2, rb]
  spins = reshape(spins, (-1, 2*rb))   # [Nt, 2*rb]
  bits_hat = (spins + 1) / 2  # Ising {-1, +1} to QUBO {0, 1}
  # QuAMax => intermediate code
  bits_final = bits_hat.clone()
  index = nonzero(bits_hat[:, rb-1] > 0.5)[:, -1]
  bits_hat[index, rb:] = 1 - bits_hat[index, rb:]
  # Differential bit encoding
  for i in range(1, nbps):             # b[i] = b[i] ^ b[i-1]
    x = bits_hat[:, i] + bits_hat[:, i - 1]
    x_dual = 2 - x
    bits_final[:, i] = where(x <= x_dual, x, x_dual)
  # calc BER
  if loss_fn in ['l2', 'mse']:
    return mse_loss(bits_final, bits_constellation)
  elif loss_fn in ['l1', 'mae']:
    return l1_loss(bits_final, bits_constellation)
  elif loss_fn == 'bce':
    pseudo_logits = bits_final * 2 - 1
    return binary_cross_entropy_with_logits(pseudo_logits, bits_constellation)
