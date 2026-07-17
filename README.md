# causa

Causal inference for Go. Pure standard library. Zero dependencies.

> **Status: early development — `v0.3.0` released.** Granger causality shipped in `v0.1.0`,
> PC-stable constraint-based discovery in `v0.2.0`, DirectLiNGAM directional discovery in
> `v0.3.0`. Pre-1.0, minor versions may break the API. Nothing
> below is claimed as shipped until it is implemented, tested against ground-truth datasets, and
> benchmarked. This README is kept honest by policy: capabilities are labeled exactly as they are.

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
| Constraint-based discovery | PC-stable algorithm (conditional-independence tests) → CPDAG | **Released in `v0.2.0`** — ground-truth-validated and benchmarked; recovers a Markov equivalence class, not a unique DAG (see below) |
| Directional discovery | DirectLiNGAM (deterministic, non-Gaussian noise) → causal order + weighted DAG | **Released in `v0.3.0`** — ground-truth-validated and benchmarked; identifies a fully directed model when the noise is non-Gaussian (see below) |
| Interventions / counterfactuals | SEM + do-calculus | Research |

Granger tells you that series *A* helps predict series *B* — necessary but not sufficient for
causation (confounders fool it). The PC algorithm and LiNGAM are what upgrade "predictive
precedence" into defensible causal structure: PC recovers a Markov equivalence class, and
DirectLiNGAM — where the non-Gaussian-noise assumption holds — pins down the full direction that
PC must leave reversible.

### Constraint-based discovery (PC-stable)

`PCStable(data, names, opts)` recovers causal structure from a panel of continuous variables
(one slice per variable). It is the order-independent PC-stable algorithm of Colombo & Maathuis
(2014): it thins a complete graph down to a skeleton with conditional-independence tests, orients
unshielded colliders, and closes under Meek's rules R1–R4. The default conditional-independence
test (`FisherZTest`) is the linear-Gaussian partial correlation — computed by QR-residualization
that reuses the same Householder solver as the Granger path — transformed by Fisher's *z*; the
`CITest` extension point lets you supply another test for non-Gaussian or discrete data.

The test of $x_i \perp\!\!\!\perp x_j \mid S$: residualize both variables on $[1, S]$, correlate
the residuals, and refer the variance-stabilized statistic to the standard normal,

```math
r_{ij\cdot S} = \operatorname{corr}\!\big(x_i - \hat{x}_i^{(S)},\; x_j - \hat{x}_j^{(S)}\big),
\qquad
z = \tfrac{1}{2}\,\ln\frac{1+r_{ij\cdot S}}{1-r_{ij\cdot S}},
```

```math
T = \sqrt{n - |S| - 3}\;\lvert z\rvert \;\overset{H_0}{\sim}\; \mathcal{N}(0,1),
\qquad
p = 2\big(1 - \Phi(T)\big) = \operatorname{erfc}\!\big(T/\sqrt{2}\big);
```

the edge is deleted (independence accepted) when $p > \alpha$ (default $\alpha = 0.05$), and the
levels stop growing once $n - |S| - 3 < 1$ (the honest small-sample cap described below).

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

### Directional discovery (DirectLiNGAM)

`DirectLiNGAM(data, names, opts)` estimates a Linear Non-Gaussian Acyclic Model (LiNGAM) from a
panel of continuous variables and returns a **full causal order plus the weighted coefficient
matrix `B`** of the structural model `x = B·x + e`. It is the deterministic DirectLiNGAM method of
Shimizu et al. (JMLR 2011) — **not** ICA-LiNGAM: there is no random ICA initialization, so, like
the rest of `causa`, the same input always yields the same output. It iteratively finds the most
exogenous variable by the paper's entropy-based independence measure (a maximum-entropy negentropy
approximation built from `E[log cosh]` and `E[x·exp(−x²/2)]` moments), regresses it out of the
rest, and recurses; the connection strengths are then least-squares estimates on the original data.
Reused throughout is the same Householder-QR OLS solver as the Granger and PC paths.

The model and the mathematics behind the direction choice: for standardized data the structural
model is

```math
x = B\,x + e,
```

with $B$ strictly lower-triangular in the causal order and $e$ mutually independent, non-Gaussian
disturbances. The differential entropy of a standardized variable $u$ is estimated by the
maximum-entropy approximation (Hyvärinen 1998),

```math
\hat{H}(u) \;\approx\; \tfrac{1}{2}\big(1+\ln 2\pi\big)
\;-\; k_1\big(\mathbb{E}[\ln\cosh u] - \gamma\big)^2
\;-\; k_2\big(\mathbb{E}[\,u\,e^{-u^2/2}\,]\big)^2,
```

with $k_1 = 79.047$, $k_2 = 7.4129$, $\gamma = 0.37457$. For a candidate cause $x_i$ against
$x_j$, the directional statistic is the log-likelihood ratio of the two directions,

```math
T \;=\; \big(H(x_j) + H(r_i^{(j)})\big) \;-\; \big(H(x_i) + H(r_j^{(i)})\big),
```

where $r_i^{(j)}$ is the standardized OLS residual of $x_i$ on $x_j$ (and vice versa): a positive
$T$ favors $x_i \to x_j$. Each candidate is scored by $\sum_j \min(0, T)^2$ — penalizing only the
pairs that testify *against* its exogeneity — and the minimizer (a true root scores ≈ 0) is peeled
off; ties break to the lowest index, keeping the whole procedure deterministic.

**Where PC leaves an edge undirected, LiNGAM directs it.** A chain `A → B → C` and a fork
`A ← B → C` are one Markov equivalence class — indistinguishable to a constraint-based method — but
DirectLiNGAM separates them, because non-Gaussian noise breaks the symmetry that made them
equivalent. The output is a single fully directed DAG, not an equivalence class.

**Assumptions** (stated bluntly because violating them silently returns a plausible but wrong
model): *linearity*, *acyclicity*, *causal sufficiency*, and — the load-bearing one — *non-Gaussian,
mutually independent noise* (at most one disturbance may be Gaussian). **On Gaussian noise the model
is fundamentally unidentifiable**: a linear-Gaussian SEM and its reverse fit the data equally well,
so the recovered order is arbitrary and meaningless. DirectLiNGAM cannot detect this and returns a
fully oriented model regardless; a LiNGAM result is only trustworthy when the non-Gaussianity
assumption genuinely holds. This failure mode is pinned by an honest-failure test. One conscious
divergence from the reference implementations: coefficient pruning is a simple absolute-magnitude
threshold (`LiNGAMOptions.PruneThreshold`), not adaptive-lasso, which would require an L1 optimizer
this stdlib-only library does not carry.

### Granger causality

`GrangerTest(cause, effect, lags)` is available since `v0.1.0` (import path
`github.com/jousudo/causa`). It fits a restricted autoregression of `effect` on its own lags and an unrestricted one
that adds `cause`'s lags, both via a Householder-QR least-squares solver, and reports the
F-statistic and its p-value:

```math
\text{restricted:}\quad y_t = c + \sum_{i=1}^{p} a_i\,y_{t-i} + \varepsilon_t
\qquad\quad
\text{unrestricted:}\quad y_t = c + \sum_{i=1}^{p} a_i\,y_{t-i} + \sum_{i=1}^{p} b_i\,x_{t-i} + \varepsilon_t
```

```math
F = \frac{(\mathrm{RSS}_r - \mathrm{RSS}_u)/p}{\mathrm{RSS}_u/(n - 2p - 1)}
\;\overset{H_0}{\sim}\; F(p,\; n-2p-1),
\qquad H_0:\ b_1 = \dots = b_p = 0,
```

with the p-value evaluated through the regularized incomplete beta function (continued-fraction
form). "x Granger-causes y" is the rejection of $H_0$ — x's past improves the prediction of y
beyond y's own past. **Known limitation — confounding:** if a hidden common cause drives
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
