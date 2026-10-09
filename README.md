# Options Pricer

Monte Carlo pricer for European and Asian options under geometric Brownian motion, with antithetic-variate and control-variate variance reduction. Validated against the closed-form Black–Scholes solution.

Built this after the Applied Maths unit on SDEs to get hands-on with simulating diffusions and measuring estimator variance.

## Model

Under the risk-neutral measure the underlying follows geometric Brownian motion:

```
dS_t = r S_t dt + σ S_t dW_t
```

Terminal price used for European options:

```
S_T = S_0 · exp((r − ½σ²)T + σ√T · Z),   Z ~ N(0, 1)
```

Asian (arithmetic-average) options are priced from fully simulated paths on `n_steps` steps.

## What's here

| File | Purpose |
|---|---|
| `black_scholes.py` | Closed-form Black–Scholes pricer (normal CDF via `math.erf`, no scipy dep). |
| `monte_carlo.py` | MC pricers for European and Asian options, with antithetic variates and a control variate on `S_T`. |
| `demo.py` | Runs the full comparison and saves a convergence plot. |

## Quick start

```bash
pip install -r requirements.txt
python demo.py
```

## Example output

```
Black-Scholes closed form call:  10.4506

Monte Carlo European call, N = 100,000
Method                             Price     Std Err   Rel Err
--------------------------------------------------------------
Plain MC                         10.4205     0.04677     0.29%
MC + antithetic                  10.4673     0.04672     0.16%
MC + control variate             10.4668     0.01788     0.16%
MC + antithetic + control        10.4602     0.01783     0.09%

Variance reduction (std-error relative to plain MC):
  MC + antithetic                   0.1% reduction
  MC + control variate             61.8% reduction
  MC + antithetic + control        61.9% reduction

Asian (arithmetic-avg) call, N = 50,000, 252 steps:
  Plain MC:        5.7740  (SE 0.03575)
  With antithetic: 5.8141  (SE 0.03599)  [-0.7% SE reduction]
```

See `convergence.png` for 95% CI bands of plain vs. reduced-variance estimates across path counts.

## Notes on the variance-reduction results

- Control variates do most of the work. `e^{-rT} S_T` is a martingale with known mean `S_0` and moves closely with the call payoff, so regressing it out removes most of the noise. SE drops by about 62%.
- Antithetic variates barely help on an at-the-money European call. The payoff `max(S_T − K, 0)` is approximately linear in `Z` near the strike, so pairing `Z` with `−Z` doesn't cancel much. Antithetic would help more for deep in-the-money options or for payoffs with stronger symmetry in the underlying shock.
- Asian antithetic SE is essentially unchanged for the same reason (the arithmetic average is a near-linear functional of the path when `σ` is modest).

## Limitations

- Constant volatility. No stochastic vol (Heston, SABR) and no local vol.
- European and Asian only, so no early exercise (American or Bermudan via Longstaff–Schwartz).
- No dividends or funding costs.
- `demo.py` fixes the RNG seed so the output is reproducible. A proper study would run many seeds and report the mean and variance of the estimator.

## Next

- Longstaff–Schwartz for American options.
- Heston stochastic-vol model.
- Quasi-Monte Carlo (Sobol sequences) benchmark.
