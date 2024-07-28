if lmbd_res is None:              # DU-LM-SB
  if lmbd_mode == 'inv':
    U_λ = inv(H_tilde @ H_tilde.T + lmbd * I)
  elif lmbd_mode == 'approx':
    # Neumann series : T^-1 = Σk (I - T)^k
    # inv(λI + A) = inv(λ(I + A/λ))
    #             ~ inv(I + A/λ) / λ
    #             ~ inv(I + A|A|)
    #             ~ Σk (A/|A|)^k
    A = H_tilde @ H_tilde.T
    A /= norm(A)
    if not 'the theoretical way': # 华罗庚公式
      it = I - A
      U_λ = it
      for _ in range(24):
        U_λ = it @ (I + U_λ)
    else:                         # wtf, it just works?!
      U_λ = I - A
      for _ in range(5):
        U_λ = matmul(U_λ, I + U_λ, out=U_λ)
else:
  if lmbd_res_mode == 'res':      # pReg-LM-SB
    U_λ = inv(H_tilde @ H_tilde.T + lmbd_res) / lmbd
  elif lmbd_res_mode == 'proj':   # ppReg-LM-SB
    U_λ = lmbd_res

H_tilde_T = H_tilde @ T
J = H_tilde_T.T @ (U_λ * (-2 / qam_var)) @ H_tilde_T
for j in range(J.shape[0]): J[j, j] = 0
z = y_tilde - H_tilde @ T.sum(axis=-1) + (sqrt(M) - 1) * H_tilde.sum(axis=-1)
h = H_tilde_T.T @ (U_λ @ (2 / sqrt(qam_var) * z))
