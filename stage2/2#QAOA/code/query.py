# 计算图的度和阶数
D = ave_D(Jc_dict, Nq)
k = min(order(Jc_dict), 6)
# 以权值的方差判断无权-带权图
theta = theta_ex[SIM_EQ if std(Jc_dict) < THRESHOLD else NON_EQ]
# 训练时见过的配置，直接使用
if p in [4, 8] and k in [2, 3, 4, 5]:
  params = theta[p][k]
else:   # 否则使用外插扩展
  params = interp_expand(p, theta[4][k], theta[8][k])
gammas, betas = np.split(params, 2)
gammas = trans_gamma(gammas, D)
# 处理 rescaler 因子, 返回初参 gamma 和 beta
factor = rescale_factor(Jc_dict) * rescaler
return gammas * factor, betas
