"""
Closed-form Black-Scholes pricer.

Used purely as a reference for validating the Monte Carlo implementation
in monte_carlo.py. No dividends, constant vol.

Normal CDF via math.erf so the file has no scipy dependency.
"""
import math
import numpy as np


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def d1(S, K, r, sigma, T):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2(S, K, r, sigma, T):
    return d1(S, K, r, sigma, T) - sigma * np.sqrt(T)


def bs_call(S, K, r, sigma, T):
    D1 = d1(S, K, r, sigma, T)
    D2 = d2(S, K, r, sigma, T)
    return S * norm_cdf(D1) - K * np.exp(-r * T) * norm_cdf(D2)


def bs_put(S, K, r, sigma, T):
    D1 = d1(S, K, r, sigma, T)
    D2 = d2(S, K, r, sigma, T)
    return K * np.exp(-r * T) * norm_cdf(-D2) - S * norm_cdf(-D1)


if __name__ == "__main__":
    # sanity check (Hull, Chapter 14 worked example)
    S, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    print(f"Call: {bs_call(S, K, r, sigma, T):.4f}")
    print(f"Put:  {bs_put(S, K, r, sigma, T):.4f}")
