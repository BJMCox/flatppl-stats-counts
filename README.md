# flatppl-stats-counts

Scalar ON/OFF count statistics for FlatPPL 0.1: independent Poisson laws,
conditional background profiling, and an exact Gamma-background count marginal.

Run the [posterior example](https://live.flatppl.org/#model=https%3A%2F%2Fraw.githubusercontent.com%2Fflatppl-community%2Fflatppl-stats-counts%2Fmain%2Fexamples%2Fplayground.flatppl&target=Pi_post) in the live playground. It loads its library modules from GitHub.

```flatppl
flatppl_compat = "0.1"
counts = load_module("main.flatppl")
observation = counts.onoff(3.0, 2.0, 5.0)
score = logdensityof(observation, record(on = 7, off = 12))
```

The background argument is the expected **ON** background count. The ratio is
**OFF/ON** background exposure, the reciprocal of the usual ON/OFF `alpha`.
ON and OFF event samples must be disjoint and conditionally independent.
Both laws impose no prior. Only `onoff_gamma_logdensity` integrates a stated
Gamma(shape, rate) background prior.

[Reference](docs/reference.md) lists signatures, domains, and formulas.
The [synthetic lab](examples/lab.flatppl) provides explicit inputs and these outputs:

| Output | Value |
|---|---:|
| `joint_logdensity` | -4.615289353760048 |
| `background_mle` | 2.534240194975457 |
| `profile_loglikelihood` | -4.269155694681689 |
| `gamma_logdensity` | -5.376379699724651 |

For example, compile its deterministic outputs with
`flatppl stablehlo examples/lab.flatppl --dtype f64 -o lab.mlir`.
The lab is synthetic; it does not reproduce an instrument dataset.

## Validation

With existing NumPy/SciPy and the JavaScript engine checkout:

```sh
FLATPPL_JS_DIR=/path/to/flatppl-js python tests/check.py
```

This uses the real module loader and observation-law materialiser, independent
Poisson score roots, a Poisson/negative-multinomial convolution, numerical
Gamma-background integration, and joint-count normalization.

Prepare portable CPU/GPU f32/f64 modules with the Rust CLI:

```sh
FLATPPL_CLI=/path/to/flatppl python tests/run_compiled.py --prepare --modules /tmp/counts-modules
```

Then, in an existing JAX/Enzyme-JAX environment, run batched values, full
Jacobians, and zero-signal Gamma curvatures on the selected JAX device:

```sh
python tests/run_compiled.py --modules /tmp/counts-modules --output /tmp/counts-results.json
```

No environment or dependencies are installed by these commands. `main.flatppl`
loads the two source modules; either module also loads independently.
