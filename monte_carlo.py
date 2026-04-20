"""
Monte Carlo pricer for European and Asian options under geometric
Brownian motion (risk-neutral measure).

Supports antithetic variates and a control variate using S_T (for
European options). Asian options are priced from full simulated paths.
"""
import numpy as np


# --------------------------------------------------------------------
# GBM simulation
# --------------------------------------------------------------------
def gbm_terminal(S0, r, sigma, T, n_paths, antithetic=False, rng=None):
    """
    Simulate terminal prices S_T under risk-neutral GBM.
    Returns an array of shape (n_paths,).
    """
    rng = rng if rng is not None else np.random.default_rng()
    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("antithetic requires even n_paths")
        half = n_paths // 2
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])
    else:
        z = rng.standard_normal(n_paths)
    drift = (r - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T) * z
    return S0 * np.exp(drift + diffusion)


def gbm_paths(S0, r, sigma, T, n_paths, n_steps, antithetic=False, rng=None):
    """
    Simulate full discretised paths.
    Returns array of shape (n_paths, n_steps + 1).
    """
    rng = rng if rng is not None else np.random.default_rng()
    dt = T / n_steps
    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("antithetic requires even n_paths")
        half = n_paths // 2
        z = rng.standard_normal((half, n_steps))
        z = np.concatenate([z, -z], axis=0)
    else:
        z = rng.standard_normal((n_paths, n_steps))
    log_incr = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_incr, axis=1)], axis=1
    )
    return S0 * np.exp(log_paths)


# --------------------------------------------------------------------
# European
# --------------------------------------------------------------------
def mc_european(
    S0, K, r, sigma, T,
    n_paths=100_000, option="call",
    antithetic=False, control_variate=False,
    rng=None,
):
    """
    Monte Carlo price for a European option.
    Returns (price, standard_error).
    """
    rng = rng if rng is not None else np.random.default_rng()
    ST = gbm_terminal(S0, r, sigma, T, n_paths, antithetic=antithetic, rng=rng)

    if option == "call":
        payoff = np.maximum(ST - K, 0.0)
    elif option == "put":
        payoff = np.maximum(K - ST, 0.0)
    else:
        raise ValueError(f"unknown option: {option}")

    discounted = np.exp(-r * T) * payoff

    if control_variate:
        # Use discounted S_T as control: E[e^{-rT} S_T] = S0 (martingale)
        Y = np.exp(-r * T) * ST
        Ybar = S0
        cov = np.cov(discounted, Y, ddof=1)
        c = -cov[0, 1] / cov[1, 1]
        adjusted = discounted + c * (Y - Ybar)
        return adjusted.mean(), adjusted.std(ddof=1) / np.sqrt(len(adjusted))

    return discounted.mean(), discounted.std(ddof=1) / np.sqrt(len(discounted))


# --------------------------------------------------------------------
# Asian (arithmetic-average)
# --------------------------------------------------------------------
def mc_asian(
    S0, K, r, sigma, T,
    n_paths=50_000, n_steps=252, option="call",
    antithetic=False, rng=None,
):
    """
    Arithmetic-average Asian option priced by Monte Carlo.
    Returns (price, standard_error).
    """
    rng = rng if rng is not None else np.random.default_rng()
    paths = gbm_paths(S0, r, sigma, T, n_paths, n_steps,
                      antithetic=antithetic, rng=rng)
    # average over the observed points (exclude S0 at t=0)
    avg = paths[:, 1:].mean(axis=1)
    if option == "call":
        payoff = np.maximum(avg - K, 0.0)
    else:
        payoff = np.maximum(K - avg, 0.0)
    discounted = np.exp(-r * T) * payoff
    return discounted.mean(), discounted.std(ddof=1) / np.sqrt(len(discounted))


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    price, se = mc_european(100, 100, 0.05, 0.2, 1.0,
                            n_paths=100_000, rng=rng,
                            antithetic=True, control_variate=True)
    print(f"European call: {price:.4f}  (SE {se:.5f})")
