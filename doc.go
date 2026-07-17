// Package causa provides causal inference and causal discovery for time
// series in pure Go (standard library only, CGO-free).
//
// Status: early development, pre-v0.1.0 — the API is not stable yet.
//
// Implemented on main:
//   - Granger causality — pairwise VAR fitting (QR-fitted OLS) + F-test
//     (GrangerTest).
//   - Constraint-based causal discovery — the order-independent PC-stable
//     algorithm returning a CPDAG (PCStable), with a pluggable
//     conditional-independence test (CITest) whose linear-Gaussian
//     partial-correlation / Fisher-z default is shipped (FisherZTest,
//     PartialCorrelation).
//
// Planned: directional discovery via LiNGAM (non-Gaussian noise). See the
// README for the honest roadmap and the assumptions each method rests on: no
// capability is claimed before it is implemented, validated against
// ground-truth datasets, and benchmarked.
package causa
