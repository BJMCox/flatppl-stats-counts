"""Independent SciPy count-mixture oracle and analytic parameter derivatives."""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import digamma, gammaln, logsumexp
from scipy.stats import poisson

POINTS = np.array(
    [
        [0, 5, 2, 1, 2, 1],
        [0, 0.2, 0.3, 2, 2, 1],
        [1e-10, 5, 2, 1, 2, 1],
        [0.2, 5, 2, 1, 2, 1],
        [1 / 7, 5, 2, 1, 2, 1],
        [1 / 7 + 1e-8, 5, 2, 1, 2, 1],
        [0, 5, 1, 110, 2, 1],
        [3, 5, 2, 1, 2, 1],
        [100, 0.2, 0.3, 2, 2, 1],
    ]
)


def background_mle(n, m, s, rho):
    """Solve the monotone likelihood score, independent of the quadratic root."""
    if s == 0:
        return (n + m) / (1 + rho)
    if m == 0:
        return max(n / (1 + rho) - s, 0)
    score = lambda b: n / (s + b) + m / b - (1 + rho)
    return brentq(score, 1e-200, (n + m) / (1 + rho) + 1, xtol=1e-14)


def gamma_score(n, m, s, rho, a, rate):
    """Convolve independent Poisson signal and negative-multinomial background."""
    t = 1 + rho + rate
    j = np.arange(n + 1)  # source events, unlike the library's background index
    logs = (
        gammaln(n + m - j + a)
        - gammaln(n - j + 1)
        - gammaln(m + 1)
        - gammaln(a)
        + m * np.log(rho)
        + a * np.log(rate)
        - (n + m - j + a) * np.log(t)
    )
    if s == 0:
        value = logs[0]
        weights = np.eye(1, n + 1)[0]
        ds = n * t / (n + m + a - 1) - 1 if n else -1.0
    else:
        logs += poisson.logpmf(j, s)
        value = logsumexp(logs)
        weights = np.exp(logs - value)
        ds = weights @ j / s - 1
    q = n + m - j + a
    gradients = np.array(
        [
            ds,
            m / rho - weights @ q / t,
            np.log(rate) - digamma(a) + weights @ digamma(q) - np.log(t),
            a / rate - weights @ q / t,
        ]
    )
    return value, gradients


def gamma_quadrature(n, m, s, rho, a, rate):
    def integrand(b):
        lp = (
            poisson.logpmf(n, s + b)
            + poisson.logpmf(m, rho * b)
            + a * np.log(rate)
            - gammaln(a)
            + (a - 1) * np.log(b)
            - rate * b
        )
        return np.exp(lp)

    probability, error = quad(integrand, 0, np.inf, epsabs=1e-12, epsrel=1e-10)
    assert error < max(1e-11, probability * 1e-8)
    return np.log(probability)


def runtime_oracle(x):
    s, rho, a, rate, b, control = x
    values, rows = [], []
    for n, m in [(7, 12), (0, 0), (7, 0)]:
        bhat = background_mle(n, m, s, rho)
        # Analytic derivatives of the likelihood's stationary equation.
        if m:
            d = 2 * (1 + rho) * bhat + (1 + rho) * s - n - m
            db = [(m - (1 + rho) * bhat) / d, -bhat * (bhat + s) / d]
        else:
            db = [-1, -n / (1 + rho) ** 2] if n / (1 + rho) > s else [0, 0]
        values.extend(
            [bhat, poisson.logpmf(n, s + bhat) + poisson.logpmf(m, rho * bhat)]
        )
        rows.extend(
            [
                list(db) + [0] * 4,
                [n / (s + bhat) - 1 if n else -1, m / rho - bhat] + [0] * 4,
            ]
        )
        g, dg = gamma_score(n, m, s, rho, a, rate)
        values.append(g)
        rows.append(list(dg) + [0, 0])
    for n in [1, 150]:
        g, dg = gamma_score(n, 0, s, rho, a, rate)
        values.append(g)
        rows.append(list(dg) + [0, 0])
    for leakage in [False, True]:
        off_mean = rho * b + (control if leakage else 0)
        values.append(poisson.logpmf(7, s + b) + poisson.logpmf(12, off_mean))
        on_score, off_score = 7 / (s + b) - 1, 12 / off_mean - 1
        rows.append(
            [
                on_score,
                b * off_score,
                0,
                0,
                on_score + rho * off_score,
                off_score if leakage else 0,
            ]
        )
    return np.array(values), np.array(rows)


def gamma_zero_signal_curvature(n, m, rho, a, rate):
    """Second signal derivative of the exact finite polynomial at zero."""
    if n == 0:
        return 0.0
    t, q = 1 + rho + rate, n + m + a
    first = n * t / (q - 1)
    second = n * (n - 1) * t * t / ((q - 1) * (q - 2)) if n > 1 else 0
    return second - first * first
