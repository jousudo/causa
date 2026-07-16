// Package causa provides causal inference and causal discovery for time
// series in pure Go (standard library only, CGO-free).
//
// Status: early development, pre-v0.1.0 — the API is not stable yet.
// The first release targets Granger causality (VAR fitting + F-test);
// constraint-based discovery (PC algorithm) and LiNGAM follow. See the
// README for the honest roadmap: no capability is claimed before it is
// implemented, validated against ground-truth datasets, and benchmarked.
package causa
