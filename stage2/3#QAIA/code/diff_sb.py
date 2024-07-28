σ = lambda x: 1 / (1 + exp(-x))
sw = lambda x: x * σ(x)
φ_s = lambda x, Λ=10: (1/Λ) * (Λ * sw(x+1) - Λ * sw(x-1)) - 1
ψ_s = lambda x, A=100, B=1.01: σ(A * (abs(x) - B))

for k, Δ_k in enumerate(Δ):
  y = y + Δ_k * (-(1 - a[k]) * x + η * c_0 * (J @ x + h))
  x = x + Δ_k * y
  x = φ_s(x)
  y = y * (1 - ψ_s(x))
