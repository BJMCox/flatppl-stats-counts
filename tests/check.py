"""Check public numeric helpers and actual observation laws against SciPy."""

import json
import subprocess
from pathlib import Path

import numpy as np
from oracles import POINTS, gamma_quadrature, gamma_score, runtime_oracle
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]


def evaluate(expressions):
    result = subprocess.run(
        ["node", str(ROOT / "tests/evaluate.cjs")],
        input=json.dumps(expressions),
        text=True,
        capture_output=True,
        check=True,
    )
    return np.asarray(json.loads(result.stdout), dtype=float)


expressions, expected = [], []
for x in POINTS:
    s, rho, shape, rate, b, control = x
    for n, m in [(7, 12), (0, 0), (7, 0)]:
        expressions.extend(
            [
                f"a.onoff_background_mle({n},{m},{s},{rho})",
                f"a.onoff_profile_loglikelihood({n},{m},{s},{rho})",
                f"a.onoff_gamma_logdensity({n},{m},{s},{rho},{shape},{rate})",
            ]
        )
    expressions.extend(
        [
            f"a.onoff_gamma_logdensity(1,0,{s},{rho},{shape},{rate})",
            f"a.onoff_gamma_logdensity(150,0,{s},{rho},{shape},{rate})",
            f"logdensityof(a.onoff({s},{b},{rho}),record(on=7,off=12))",
            f"logdensityof(a.onoff_with_leakage({s},{control},{b},{rho}),record(on=7,off=12))",
        ]
    )
    expected.extend(runtime_oracle(x)[0])
actual = evaluate(expressions)
np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)

# Explicit zero means retain the Poisson point mass, including leakage.
np.testing.assert_allclose(
    evaluate(
        [
            "logdensityof(a.onoff(0,0,5),record(on=0,off=0))",
            "logdensityof(a.onoff_with_leakage(0,1,0,5),record(on=0,off=2))",
            "logdensityof(a.onoff(0,0,5),record(on=1,off=0))",
            "logdensityof(a.onoff(0,0,5),record(on=0,off=1))",
        ]
    ),
    [0, poisson.logpmf(2, 1), -np.inf, -np.inf],
    atol=1e-13,
)

# Cancellation-prone root: its score equation and finite positive limit matter.
large = evaluate(["a.onoff_background_mle(7,12,1e12,5)"])[0]
assert abs(large - 2) < 1e-10
assert abs(7 / (1e12 + large) + 12 / large - 6) < 1e-13

for args in [
    (0, 0, 0, 0.2, 0.3, 2),
    (1, 0, 0, 5, 0.3, 2),
    (7, 12, 0, 5, 2, 1),
    (7, 12, 0.2, 5, 2, 1),
    (7, 12, 3, 5, 2, 1),
]:
    np.testing.assert_allclose(
        gamma_score(*args)[0], gamma_quadrature(*args), atol=2e-10
    )

# Sum actual library probabilities; small means make the omitted tails negligible.
mass_values = evaluate(
    [
        f"a.onoff_gamma_logdensity({n},{m},0.2,1.3,2,8)"
        for n in range(16)
        for m in range(20)
    ]
)
mass = np.exp(mass_values).sum()
assert abs(mass - 1) < 2e-11
# Sampling exercises record-field composition through the actual constructors.
request = {
    "source": 'flatppl_compat="0.1"\na=load_module("../main.flatppl")\n'
    "x ~ a.onoff(3,2,5)\ny ~ a.onoff_with_leakage(3,1,2,5)\n"
    "z ~ a.onoff(0,0,5)\n",
    "bindings": ["x", "y", "z"],
    "sampleCount": 8192,
}
sampled = subprocess.run(
    ["node", str(ROOT / "tests/evaluate.cjs")],
    input=json.dumps(request),
    text=True,
    capture_output=True,
    check=True,
)
means = []
for measure, expected_means in zip(
    json.loads(sampled.stdout), [(5, 10), (5, 11), (0, 0)]
):
    for name, mean in zip(["on", "off"], expected_means):
        draws = np.asarray(list(measure["fields"][name]["samples"].values()))
        se = np.sqrt(mean / len(draws))
        assert abs(draws.mean() - mean) <= 6 * se
        means.append({"mean": float(draws.mean()), "standard_error": float(se)})
print(
    json.dumps(
        {
            "public_values": len(actual),
            "quadrature_cases": 5,
            "normalization_mass": mass,
            "max_value_error": float(np.max(abs(actual - expected))),
            "sampling_seed": 47710,
            "sample_count": 8192,
            "sampling_means": means,
        }
    )
)
