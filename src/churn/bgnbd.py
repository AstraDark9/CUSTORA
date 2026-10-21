"""
BG/NBD ("Buy 'Til You Die") model of repeat purchasing -- Fader, Hardie & Lee (2005).

Why it is here: when customers can leave at any time without telling you (a
shop, not a phone contract) the hard question is "has this person quietly
stopped buying, or are they just between orders?". A plain "days since last
order" cannot answer that, because a customer who normally buys weekly and has
been silent for 6 weeks is a very different case from one who buys twice a
year and has been silent for 6 weeks.

BG/NBD answers it from three numbers per customer:

    x    repeat purchases so far (distinct purchase days minus the first one)
    t_x  how long after their first purchase the last one happened
    T    how long ago the first purchase was (the customer's "age")

and two ideas: while "alive" a customer buys at a steady personal rate (rates
differ between customers, Gamma distributed), and after each purchase there is
a chance they silently drop out (also differing between customers, Beta
distributed). Fitting four shared parameters on the whole customer base then
gives, for every individual:

    p_alive(...)        probability they have NOT dropped out yet
    expected_orders(..) expected number of purchases in the next N days

Both are used as *features* for the churn model. They are computed with
nothing but information available on the snapshot date, so they cannot leak.

Implemented directly on scipy (no extra dependency). Time is measured in weeks.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1

DEFAULT_PARAMS = (0.5, 8.0, 0.8, 2.5)  # r, alpha, a, b -- only used if fitting fails
_BOUNDS = [(-6.0, 4.0)] * 4  # on the log scale


def _neg_log_likelihood(log_params, x, t_x, T, penalty):
    r, alpha, a, b = np.exp(log_params)
    safe_x = np.maximum(x, 1.0)
    ln_a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    ln_a2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
    ln_a3 = -(r + x) * np.log(alpha + T)
    ln_a4 = np.log(a) - np.log(b + safe_x - 1.0) - (r + x) * np.log(alpha + t_x)
    ln_tail = np.where(x > 0, np.logaddexp(ln_a3, ln_a4), ln_a3)
    value = -np.mean(ln_a1 + ln_a2 + ln_tail)
    return value + penalty * float(np.sum(np.square(log_params)))


def fit(x, t_x, T, penalty=1e-4, max_customers=6000, seed=0):
    """
    Fit (r, alpha, a, b) by maximum likelihood. `x`, `t_x`, `T` are arrays with
    one entry per customer (T and t_x in weeks). Returns a tuple, or
    DEFAULT_PARAMS when there is too little data / the optimiser fails.
    """
    x = np.asarray(x, dtype=float)
    t_x = np.asarray(t_x, dtype=float)
    T = np.asarray(T, dtype=float)
    ok = np.isfinite(x) & np.isfinite(t_x) & np.isfinite(T) & (T > 0) & (t_x <= T + 1e-9)
    x, t_x, T = x[ok], t_x[ok], T[ok]
    if len(x) < 50 or x.sum() < 20:
        return DEFAULT_PARAMS
    if len(x) > max_customers:  # likelihood is a mean, so a sample is plenty
        keep = np.random.default_rng(seed).choice(len(x), max_customers, replace=False)
        x, t_x, T = x[keep], t_x[keep], T[keep]
    start = np.log(np.array(DEFAULT_PARAMS))
    try:
        result = minimize(
            _neg_log_likelihood, start, args=(x, t_x, T, penalty),
            method="L-BFGS-B", bounds=_BOUNDS,
        )
        params = np.exp(result.x)
        if not np.all(np.isfinite(params)):
            return DEFAULT_PARAMS
        return tuple(float(p) for p in params)
    except (ValueError, FloatingPointError):  # pragma: no cover - defensive
        return DEFAULT_PARAMS


def p_alive(params, x, t_x, T):
    """Probability each customer has not (silently) churned yet, in [0, 1]."""
    r, alpha, a, b = params
    x = np.asarray(x, dtype=float)
    t_x = np.asarray(t_x, dtype=float)
    T = np.asarray(T, dtype=float)
    safe_x = np.maximum(x, 1.0)
    log_odds_dead = (
        np.log(a) - np.log(b + safe_x - 1.0)
        + (r + x) * (np.log(alpha + T) - np.log(alpha + t_x))
    )
    log_odds_dead = np.clip(log_odds_dead, -50, 50)
    alive = 1.0 / (1.0 + np.exp(log_odds_dead))
    return np.where(x > 0, alive, 1.0)  # no repeat purchase yet: no evidence of dropout


def expected_orders(params, x, t_x, T, horizon_weeks):
    """Expected number of purchases in the next `horizon_weeks` weeks."""
    r, alpha, a, b = params
    x = np.asarray(x, dtype=float)
    t_x = np.asarray(t_x, dtype=float)
    T = np.asarray(T, dtype=float)
    alive = p_alive(params, x, t_x, T)
    fallback = alive * (r + x) / (alpha + T) * horizon_weeks  # simple rate-based estimate
    if a <= 1.05:  # the closed form needs a > 1
        return fallback
    z = horizon_weeks / (alpha + T + horizon_weeks)
    with np.errstate(all="ignore"):
        hyp = hyp2f1(r + x, b + x, a + b + x - 1.0, z)
        first = (a + b + x - 1.0) / (a - 1.0)
        second = 1.0 - hyp * ((alpha + T) / (alpha + T + horizon_weeks)) ** (r + x)
        out = first * second * alive
    out = np.where(np.isfinite(out) & (out >= 0), out, fallback)
    return out


def customer_inputs(first_dates, last_dates, order_days, as_of):
    """(x, t_x, T) in weeks from per-customer first/last purchase dates and #purchase days."""
    first = np.asarray(first_dates, dtype="datetime64[D]")
    last = np.asarray(last_dates, dtype="datetime64[D]")
    now = np.datetime64(as_of, "D")
    x = np.maximum(np.asarray(order_days, dtype=float) - 1.0, 0.0)
    t_x = (last - first).astype(float) / 7.0
    T = (now - first).astype(float) / 7.0
    return x, t_x, np.maximum(T, t_x)
