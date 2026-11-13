"""Prepare portable StableHLO modules or run batched values and AD on JAX/Enzyme."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from oracles import POINTS, gamma_zero_signal_curvature, runtime_oracle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.prepare:
        args.modules.mkdir(parents=True, exist_ok=True)
        values, jacobians = zip(*(runtime_oracle(x) for x in POINTS))
        (args.modules / "expected.json").write_text(
            json.dumps(
                {
                    "points": POINTS.tolist(),
                    "values": np.asarray(values).tolist(),
                    "jacobians": np.asarray(jacobians).tolist(),
                }
            )
        )
        for target in ("cpu", "gpu"):
            for dtype in ("f32", "f64"):
                subprocess.run(
                    [
                        os.environ.get("FLATPPL_CLI", "flatppl"),
                        "stablehlo",
                        str(root / "tests/runtime.flatppl"),
                        "--dtype",
                        dtype,
                        "--target",
                        target,
                        "-o",
                        str(args.modules / f"{target}-{dtype}.mlir"),
                    ],
                    check=True,
                )
        if not args.output:
            print(args.modules)
            return
    import jax
    import jax.numpy as jnp
    from enzyme_ad.jax import hlo_call

    jax.config.update("jax_enable_x64", True)
    expected = json.loads((args.modules / "expected.json").read_text())
    target = "cpu" if jax.devices()[0].platform == "cpu" else "gpu"
    results = []
    for dtype in ("f32", "f64"):
        source = (
            (args.modules / f"{target}-{dtype}.mlir")
            .read_text()
            .replace("@logdensity(", "@main(")
        )

        def forward(x, source=source):
            return hlo_call(x, source=source)[0]

        points = jnp.asarray(
            expected["points"], dtype=jnp.float32 if dtype == "f32" else jnp.float64
        )
        values = np.asarray(jax.jit(jax.vmap(forward))(points))
        jacobians = np.asarray(jax.jit(jax.vmap(jax.jacrev(forward)))(points))
        tolerance = 3e-4 if dtype == "f32" else 2e-10
        gradient_tolerance = 1e-3 if dtype == "f32" else 2e-8
        np.testing.assert_allclose(
            values, expected["values"], rtol=tolerance, atol=tolerance
        )
        np.testing.assert_allclose(
            jacobians,
            expected["jacobians"],
            rtol=gradient_tolerance,
            atol=gradient_tolerance,
        )
        # Exact boundary curvature checks the polynomial rather than only a tangent.
        curvatures, expected_curvatures, curvature_scales = [], [], []
        for point in points[np.asarray(expected["points"])[:, 0] == 0]:
            for index, (n, m) in zip(
                [2, 5, 8, 9, 10], [(7, 12), (0, 0), (7, 0), (1, 0), (150, 0)]
            ):

                def scalar(signal, point=point, index=index):
                    return forward(point.at[0].set(signal))[index]

                curvatures.append(float(jax.jit(jax.grad(jax.grad(scalar)))(point[0])))
                rho, shape, rate = np.asarray(point)[[1, 2, 3]]
                expected_curvatures.append(
                    gamma_zero_signal_curvature(n, m, rho, shape, rate)
                )
                first = n * (1 + rho + rate) / (n + m + shape - 1) if n else 0
                curvature_scales.append(max(1, first * first))
        # Curvature subtracts two terms of order first**2 even when its answer
        # is zero. Allow float roundoff on those terms, rather than on zero.
        curvature_error = abs(np.asarray(curvatures) - expected_curvatures)
        curvature_tolerance = gradient_tolerance + 8 * np.finfo(
            np.asarray(points).dtype
        ).eps * np.asarray(curvature_scales)
        np.testing.assert_array_less(curvature_error, curvature_tolerance)
        results.append(
            {
                "dtype": dtype,
                "max_boundary_curvature_error": float(
                    np.max(abs(np.asarray(curvatures) - expected_curvatures))
                ),
                "max_value_error": float(np.max(abs(values - expected["values"]))),
                "max_jacobian_error": float(
                    np.max(abs(jacobians - expected["jacobians"]))
                ),
                "max_jacobian_error_index": list(
                    map(
                        int,
                        np.unravel_index(
                            np.argmax(abs(jacobians - expected["jacobians"])),
                            jacobians.shape,
                        ),
                    )
                ),
            }
        )
    report = {
        "device": str(jax.devices()[0]),
        "points": len(points),
        "results": results,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
