# flatppl-stats-counts

[Function reference](docs/reference.md) · [Live example](https://live.flatppl.org/#model=https%3A%2F%2Fraw.githubusercontent.com%2Fflatppl-community%2Fflatppl-stats-counts%2Fmain%2Fexamples%2Fplayground.flatppl&target=Pi_post)

Scalar ON/OFF count laws, conditional background profiling, and an exact Gamma-background count marginal for FlatPPL. Library version **0.1**.

```flatppl
flatppl_compat = "0.1"
counts = load_module("main.flatppl")
observation = counts.onoff(3.0, 2.0, 5.0)
score = logdensityof(observation, record(on = 7, off = 12))
```

Background denotes the expected ON count. The exposure ratio is OFF/ON, the reciprocal of the usual `alpha`. Event samples must be disjoint and conditionally independent. Only the Gamma marginal supplies a background prior.

The [synthetic laboratory example](examples/lab.flatppl) exposes count densities, a background estimate, and a profile likelihood.

See the reference for definitions, domains, numerical limits, and validation commands.
