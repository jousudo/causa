# causa

Causal inference for Go. Pure standard library. Zero dependencies.

> **Status: early development — `v0.1.0` released.** Granger causality shipped in `v0.1.0`;
> PC-stable constraint-based discovery is implemented on `main` and ships in `v0.2.0`. Pre-1.0,
> minor versions may break the API. Nothing below is claimed as shipped until it is implemented,
> tested against ground-truth datasets, and benchmarked. This README is kept honest by policy:
> capabilities are labeled exactly as they are.

## What

`causa` is a causal inference and causal discovery library for time series, written in pure
Go (stdlib only, CGO-free). It is being built to power deterministic root-cause analysis in
AIOpsFlow — an AIOps platform by the same author, currently in private development ahead of
its own open-source release — but `causa` is designed as a general-purpose, standalone
library with no ties to any host application.

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
| Granger causality | Pairwise OLS autoregressions (QR-fitted) + F-test | **Released in `v0.1.0`** — ground-truth-validated and benchmarked; flags confounders by design (see below) |
| Constraint-based discovery | PC-stable algorithm (conditional-independence tests) → CPDAG | **Implemented on `main`** (unreleased — ships in `v0.2.0`) — ground-truth-validated and benchmarked; recovers a Markov equivalence class, not a unique DAG (see below) |
| Directional discovery | LiNGAM (ICA-based, non-Gaussian noise) | Planned |
| Interventions / counterfactuals | SEM + do-calculus | Research |

Granger tells you that series *A* helps predict series *B* — necessary but not sufficient for
causation (confounders fool it). The PC algorithm and LiNGAM are what upgrade "predictive
precedence" into defensible causal structure. LiNGAM ships next, validated before it is claimed.

### Constraint-based discovery (PC-stable)

`PCStable(data, names, opts)` recovers causal structure from a panel of continuous variables
(one slice per variable). It is the order-independent PC-stable algorithm of Colombo & Maathuis
(2014): it thins a complete graph down to a skeleton with conditional-independence tests, orients
unshielded colliders, and closes under Meek's rules R1–R4. The default conditional-independence
test (`FisherZTest`) is the linear-Gaussian partial correlation — computed by QR-residualization
that reuses the same Householder solver as the Granger path — transformed by Fisher's *z*; the
`CITest` extension point lets you supply another test for non-Gaussian or discrete data.

**Output is a CPDAG, not a DAG.** Constraint-based discovery identifies structure only up to
*Markov equivalence*. A **directed** edge `A → C` is compelled (every DAG consistent with the data
agrees on it); an **undirected** edge `A — B` is reversible (the equivalence class contains DAGs
orienting it either way). A chain `A → B → C` and a fork `A ← B → C` are indistinguishable from
observational data, so their edges come back undirected; a collider `A → C ← B` is oriented,
because it alone implies `A ⫫ B`. Read an undirected edge as "direction not identified", never as
"no causal link".

**Assumptions** (standard for constraint-based discovery, and stated because they are easy to
violate in practice): *causal sufficiency* (no unobserved common cause of two measured variables),
*faithfulness* (every conditional independence in the data is entailed by the graph), and a
correct CI test (the default assumes linear-Gaussian data). On small samples there is an honest
cap: conditioning sets stop growing once `n − |S| − 3 < 1`, so independencies that need larger
conditioning sets than the sample supports cannot be tested and some edges a larger sample would
remove may remain.

`GrangerTest(cause, effect, lags)` is available since `v0.1.0` (import path
`github.com/jousudo/causa`). It fits a restricted autoregression of `effect` on its own lags and an unrestricted one
that adds `cause`'s lags, both via a Householder-QR least-squares solver, and reports the
F-statistic and its p-value. **Known limitation — confounding:** if a hidden common cause drives
both series, Granger reports causality even when no direct edge exists. This is inherent to the
method, not a defect, and it is exactly why the PC algorithm and LiNGAM are on the roadmap; the
behavior is pinned by a dedicated test (`TestGrangerFlagsConfounder`) and documented on the
function so it is never mistaken for a true causal claim.

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

```bash
go get github.com/jousudo/causa
```

## Contributing

Issues, discussion and code contributions are welcome; note that pre-1.0, minor versions may
still adjust the API. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
stdlib-only constraint, godoc/Example/benchmark expectations, and PR etiquette.

## License

[Apache License 2.0](LICENSE)
