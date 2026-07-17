// Package causa provides causal inference and causal discovery for time
// series in pure Go (standard library only, CGO-free).
//
// Status: early development — v0.2.0 released; pre-v1.0, minor versions may
// still change the API.
//
// Implemented:
//   - Granger causality — pairwise VAR fitting (QR-fitted OLS) + F-test
//     (GrangerTest). Released in v0.1.0.
//   - Constraint-based causal discovery — the order-independent PC-stable
//     algorithm returning a CPDAG (PCStable), with a pluggable
//     conditional-independence test (CITest) whose linear-Gaussian
//     partial-correlation / Fisher-z default is shipped (FisherZTest,
//     PartialCorrelation). Released in v0.2.0.
//   - Directional causal discovery — the deterministic DirectLiNGAM algorithm
//     (DirectLiNGAM) for linear acyclic models with non-Gaussian independent
//     noise, recovering a full causal order and weighted coefficient matrix.
//     On main, ships in v0.3.0.
//
// Research: interventions and counterfactuals via SEM + do-calculus. See the
// README for the honest roadmap and the assumptions each method rests on: no
// capability is claimed before it is implemented, validated against
// ground-truth datasets, and benchmarked.
package causa
