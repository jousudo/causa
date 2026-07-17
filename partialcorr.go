package causa

import (
	"errors"
	"math"
)

// ErrConditioningTooLarge is returned by a CITest when the conditioning set is
// too large for the available sample: the Fisher z-transform needs at least one
// residual degree of freedom, i.e. n − |S| − 3 ≥ 1, where n is the sample size
// and |S| the number of conditioning variables.
var ErrConditioningTooLarge = errors.New("causa: too few samples for the conditioning-set size (need n - |S| - 3 >= 1)")

// CITest reports the p-value of the null hypothesis that variables i and j are
// conditionally independent given the conditioning set cond ("i ⫫ j | cond").
// It is the extension point through which PCStable queries the data, so a caller
// can substitute an alternative test (for example one tailored to discrete data)
// for the linear-Gaussian FisherZTest shipped as the default.
//
// Contract that every implementation must honour:
//
//   - data holds one variable per outer slice — data[v] is variable v's sample —
//     and all variables share the same length n = len(data[0]).
//   - i and j are distinct variable indices. cond is a set of variable indices,
//     none equal to i or j; its order carries no meaning and the result must not
//     depend on it (PCStable relies on this to stay order-independent).
//   - The returned p-value lies in [0, 1]. The caller treats i and j as
//     independent iff p > alpha, so a larger p means "more independent". A test
//     of exact dependence should return a p at or near 0.
//   - On failure — a rank-deficient conditioning design, or too few samples for
//     the conditioning size — it returns a non-nil error; the p-value is then
//     unspecified and PCStable aborts with that error.
type CITest func(data [][]float64, i, j int, cond []int) (pValue float64, err error)

// PartialCorrelation returns the sample partial correlation of variables i and j
// given the conditioning set cond, computed by QR-residualization rather than by
// inverting a covariance matrix.
//
// Variable i and variable j are each regressed on an intercept plus the
// conditioning variables using the package's Householder-QR least-squares solver
// (see fitOLS); the returned value is the Pearson correlation of the two residual
// vectors. Because the intercept is always included, the residuals are mean-zero
// and the correlation is
//
//	r = Σ rᵢ·rⱼ / sqrt(Σ rᵢ² · Σ rⱼ²).
//
// With an empty conditioning set this reduces to the ordinary Pearson correlation
// of variables i and j.
//
// The data must be rectangular (every variable the same length) with i, j and
// every index in cond in range; like any Go slice access, malformed input
// panics. PCStable validates its input before calling the test — the checks are
// not repeated here.
//
// Rank deficiency is detected, not ignored. If the conditioning design is rank
// deficient (a constant conditioning variable, or collinear conditioners) the
// underlying solver returns ErrSingular, which is propagated. If either variable
// is (numerically) perfectly explained by the conditioning set — its residual
// vector collapses to zero — the correlation is undefined and ErrSingular is
// returned as well. Both are the partial-correlation analogue of a rank-deficient
// covariance matrix.
//
// The returned r lies in [−1, 1]; a value pushed fractionally outside that range
// by floating-point rounding is clamped so it never becomes NaN.
func PartialCorrelation(data [][]float64, i, j int, cond []int) (float64, error) {
	n := len(data[i])

	// Design matrix [1, cond₀, cond₁, …]; k = 1 + |cond| columns, n rows.
	k := 1 + len(cond)
	x := make([][]float64, n)
	for r := 0; r < n; r++ {
		row := make([]float64, k)
		row[0] = 1
		for c, s := range cond {
			row[c+1] = data[s][r]
		}
		x[r] = row
	}

	ri, ssi, err := residualize(x, data[i])
	if err != nil {
		return 0, err
	}
	rj, ssj, err := residualize(x, data[j])
	if err != nil {
		return 0, err
	}

	// A residual sum of squares that has collapsed relative to the variable's own
	// centred variance means the variable is (numerically) a deterministic
	// function of the conditioning set: the partial correlation is undefined.
	const relTol = 1e-12
	if ssi <= relTol*centeredSS(data[i]) || ssj <= relTol*centeredSS(data[j]) {
		return 0, ErrSingular
	}

	var sij, sii, sjj float64
	for r := 0; r < n; r++ {
		sij += ri[r] * rj[r]
		sii += ri[r] * ri[r]
		sjj += rj[r] * rj[r]
	}
	denom := math.Sqrt(sii * sjj)
	if denom == 0 || !isFinite(denom) {
		return 0, ErrSingular
	}
	r := sij / denom
	if r > 1 {
		r = 1
	} else if r < -1 {
		r = -1
	}
	return r, nil
}

// residualize regresses y on the design x (via the Householder-QR solver) and
// returns the residual vector y − X·b together with its sum of squares. It
// propagates ErrSingular when the design is rank deficient.
func residualize(x [][]float64, y []float64) ([]float64, float64, error) {
	fit, err := fitOLS(x, y)
	if err != nil {
		return nil, 0, err
	}
	res := make([]float64, len(y))
	var ss float64
	for r := range y {
		var pred float64
		for c, b := range fit.coef {
			pred += x[r][c] * b
		}
		e := y[r] - pred
		res[r] = e
		ss += e * e
	}
	return res, ss, nil
}

// centeredSS returns Σ (v − mean(v))², the total centred sum of squares of v. It
// is the reference scale for judging whether a residual sum of squares has
// numerically collapsed in PartialCorrelation.
func centeredSS(v []float64) float64 {
	var mean float64
	for _, x := range v {
		mean += x
	}
	mean /= float64(len(v))
	var ss float64
	for _, x := range v {
		d := x - mean
		ss += d * d
	}
	return ss
}

// FisherZTest is the default CITest for PCStable: a linear-Gaussian test of
// conditional independence built on the partial correlation and Fisher's
// variance-stabilizing z-transform.
//
// It computes the partial correlation r of variables i and j given cond (see
// PartialCorrelation), transforms it to
//
//	z = ½·ln((1 + r)/(1 − r)),
//
// and forms the test statistic
//
//	T = sqrt(n − |cond| − 3)·|z|,
//
// which is asymptotically standard normal under the null of conditional
// independence. The two-sided p-value is the standard-normal tail
//
//	p = 2·(1 − Φ(T)) = erfc(T/√2),      Φ(x) = ½·(1 + erf(x/√2)),
//
// evaluated through math.Erfc so precision is retained deep in the small-p tail
// that matters for edge deletion. A |r| numerically at (or beyond) 1 — exact
// dependence — is clamped just inside the unit interval so z stays finite and the
// p-value collapses to ~0 (strongly dependent), never NaN.
//
// It returns ErrConditioningTooLarge when n − |cond| − 3 < 1 (no residual degrees
// of freedom for the transform), and propagates ErrSingular from
// PartialCorrelation when the conditioning design is rank deficient.
//
// Assumptions. The test assumes the variables are jointly Gaussian and their
// dependence is linear; the partial correlation is then zero exactly when the
// variables are conditionally independent. On non-linear or non-Gaussian data a
// zero partial correlation no longer implies independence, so this test — and any
// PCStable run using it — can miss dependencies. Supply a different CITest for
// such data.
//
// Reference: Kalisch & Bühlmann, "Estimating High-Dimensional Directed Acyclic
// Graphs with the PC-Algorithm", JMLR 8 (2007) 613–636, §3.1 (Gaussian CI test).
func FisherZTest(data [][]float64, i, j int, cond []int) (float64, error) {
	n := len(data[i])
	df := n - len(cond) - 3
	if df < 1 {
		return 0, ErrConditioningTooLarge
	}
	r, err := PartialCorrelation(data, i, j, cond)
	if err != nil {
		return 0, err
	}
	// Clamp strictly inside (−1, 1) so the log stays finite for exact dependence.
	const eps = 1e-15
	if r > 1-eps {
		r = 1 - eps
	} else if r < -1+eps {
		r = -1 + eps
	}
	z := 0.5 * math.Log((1+r)/(1-r))
	stat := math.Sqrt(float64(df)) * math.Abs(z)
	// p = 2·(1 − Φ(stat)) = erfc(stat/√2); Erfc keeps the small-p tail accurate.
	p := math.Erfc(stat / math.Sqrt2)
	return p, nil
}
