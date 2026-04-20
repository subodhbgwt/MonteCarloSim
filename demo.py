"""
Runs the full comparison:
    1. Black-Scholes closed form.
    2. Plain Monte Carlo, antithetic, control variate, and both combined.
    3. Asian (arithmetic-average) option.
    4. Convergence plot at multiple path counts.

Usage:
    python demo.py
"""
import numpy as np
import matplotlib.pyplot as plt

from black_scholes import bs_call
from monte_carlo import mc_european, mc_asian

# ---- parameters (Hull Chapter 14 test case) ----
S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
N = 100_000
SEED = 42

bs_price = bs_call(S0, K, r, sigma, T)
print(f"Black-Scholes closed form call:  {bs_price:.4f}\n")

# ---- main comparison ----
configs = [
    ("Plain MC",                   dict(antithetic=False, control_variate=False)),
    ("MC + antithetic",            dict(antithetic=True,  control_variate=False)),
    ("MC + control variate",       dict(antithetic=False, control_variate=True)),
    ("MC + antithetic + control",  dict(antithetic=True,  control_variate=True)),
]

print(f"Monte Carlo European call, N = {N:,}")
print(f"{'Method':<30}{'Price':>10}{'Std Err':>12}{'Rel Err':>10}")
print("-" * 62)

rows = []
for label, opts in configs:
    rng = np.random.default_rng(SEED)
    price, se = mc_european(S0, K, r, sigma, T, n_paths=N, rng=rng, **opts)
    rel_err = abs(price - bs_price) / bs_price * 100
    print(f"{label:<30}{price:>10.4f}{se:>12.5f}{rel_err:>9.2f}%")
    rows.append((label, price, se, rel_err))

plain_se = rows[0][2]
print("\nVariance reduction (std-error relative to plain MC):")
for label, _price, se, _rel in rows[1:]:
    reduction = (1 - se / plain_se) * 100
    print(f"  {label:<30} {reduction:>6.1f}% reduction")

# ---- Asian option (no closed form, just show price + SE) ----
rng = np.random.default_rng(SEED)
a_price, a_se = mc_asian(S0, K, r, sigma, T, n_paths=50_000, n_steps=252,
                         antithetic=False, rng=rng)
rng = np.random.default_rng(SEED)
a_price_anti, a_se_anti = mc_asian(S0, K, r, sigma, T, n_paths=50_000, n_steps=252,
                                   antithetic=True, rng=rng)
print("\nAsian (arithmetic-avg) call, N = 50,000, 252 steps:")
print(f"  Plain MC:        {a_price:.4f}  (SE {a_se:.5f})")
print(f"  With antithetic: {a_price_anti:.4f}  (SE {a_se_anti:.5f})  "
      f"[{(1 - a_se_anti / a_se) * 100:+.1f}% SE reduction]")

# ---- convergence plot ----
print("\nGenerating convergence plot...")
path_counts = [1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000]
plain_p, plain_e = [], []
combo_p, combo_e = [], []
for M in path_counts:
    rng1 = np.random.default_rng(SEED)
    rng2 = np.random.default_rng(SEED)
    p1, e1 = mc_european(S0, K, r, sigma, T, n_paths=M, rng=rng1)
    p2, e2 = mc_european(S0, K, r, sigma, T, n_paths=M, rng=rng2,
                         antithetic=True, control_variate=True)
    plain_p.append(p1); plain_e.append(e1)
    combo_p.append(p2); combo_e.append(e2)

fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(path_counts, plain_p, yerr=[1.96 * e for e in plain_e],
            label="Plain MC", marker="o", capsize=4)
ax.errorbar(path_counts, combo_p, yerr=[1.96 * e for e in combo_e],
            label="MC + antithetic + control", marker="s", capsize=4)
ax.axhline(bs_price, color="k", linestyle="--", alpha=0.6,
           label=f"Black-Scholes = {bs_price:.3f}")
ax.set_xscale("log")
ax.set_xlabel("Monte Carlo paths")
ax.set_ylabel("Price estimate (95% CI)")
ax.set_title(f"European Call  S0={S0:.0f}, K={K:.0f}, r={r}, σ={sigma}, T={T}")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("convergence.png", dpi=130)
print("Saved convergence.png")
