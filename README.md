# causa

Causal inference for Go. Pure standard library. Zero dependencies.

> **Status: early development — pre-`v0.1.0`.** The API is not stable yet, and nothing below
> is claimed as shipped until it is implemented, tested against ground-truth datasets, and
> benchmarked. This README is kept honest by policy: capabilities are labeled exactly as they
> are.

## What

`causa` is a causal inference and causal discovery library for time series, written in pure
Go (stdlib only, CGO-free). It is being built to power deterministic root-cause analysis in
[AIOpsFlow](https://github.com/jousudo/aiopsflow) — an open-source AIOps platform — but it is
designed as a general-purpose, standalone library with no ties to any host application.

## Why

Correlation is not causation, and nowhere does that bite harder than in production telemetry:
a network saturation event raises database CPU *and* API latency at the same time, and naive
correlation blames the database. Distinguishing real causal structure from coincident movement
requires actual causal mathematics — conditional independence, structural asymmetries,
intervention modeling — not just pattern matching.

Today that mathematics lives almost entirely in Python (DoWhy, causal-learn, gCastle,
Tigramite). A Go service that needs causal reasoning must ship a Python sidecar with a
scientific stack — hundreds of megabytes of runtime and a supply chain to audit. `causa`
exists to remove that hop: causal inference as a plain Go import, small enough to embed
anywhere Go runs.

## Roadmap

| Capability | Method | Status |
|---|---|---|
| Granger causality | VAR fitting + F-test on time series | **In development** — first release target |
| Constraint-based discovery | PC algorithm (conditional-independence tests) | Planned |
| Directional discovery | LiNGAM (ICA-based, non-Gaussian noise) | Planned |
| Interventions / counterfactuals | SEM + do-calculus | Research |

Granger tells you that series *A* helps predict series *B* — necessary but not sufficient for
causation (confounders fool it). The PC algorithm and LiNGAM are what upgrade "predictive
precedence" into defensible causal structure. They ship in that order, each validated before
the next starts.

## Design principles

- **Standard library only.** No third-party imports, ever. CGO-free. The whole supply chain
  is the Go toolchain.
- **Deterministic and reproducible.** Same input, same output — causal claims must be
  auditable.
- **Documented like it matters.** Every exported symbol carries godoc; every feature ships
  with testable `Example` functions and benchmarks.
- **Validated, not asserted.** Algorithms are tested against synthetic ground-truth suites
  and cross-checked against reference implementations before any release claims them.
- **Semantic versioning** from `v0.1.0`. Pre-1.0, minor versions may break the API; this is
  stated here instead of discovered by users.

## Installation

Once `v0.1.0` is tagged:

```bash
go get github.com/jousudo/causa
```

Until then, the module is public for visibility and review, not yet for consumption.

## Contributing

Issues and discussion are welcome from day one. Code contributions will be easiest to accept
after `v0.1.0` stabilizes the core API — a CONTRIBUTING guide will land with it.

## License

[Apache License 2.0](LICENSE)
