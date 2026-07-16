package causa

import (
	"errors"
	"fmt"
	"math"
)

// Errors returned by GrangerTest. They are sentinel values so callers can
// distinguish the failure modes with errors.Is.
var (
	// ErrMismatchedLength is returned when cause and effect have different
	// lengths; the test is only defined for aligned, equal-length series.
	ErrMismatchedLength = errors.New("causa: cause and effect series have different lengths")

	// ErrInvalidLags is returned when lags is less than 1. At least one lag is
	// required to define a predictive-precedence relationship.
	ErrInvalidLags = errors.New("causa: lags must be a positive integer")

	// ErrTooShort is returned when the series are too short to estimate the
	// unrestricted model with a positive residual degrees of freedom. The test
	// needs len >= 3*lags + 2.
	ErrTooShort = errors.New("causa: series too short for the requested number of lags")

	// ErrNonFinite is returned when either series contains a NaN or an infinite
	// value, which would silently poison the regression.
	ErrNonFinite = errors.New("causa: series contains a NaN or infinite value")

	// ErrSingular is returned when a regression design matrix is rank deficient
	// — typically a constant series, or a series perfectly collinear with its
	// own lags — so the F-statistic cannot be formed reliably.
	ErrSingular = errors.New("causa: design matrix is rank deficient (constant or collinear series)")
)

// GrangerResult holds the outcome of a pairwise Granger causality test.
//
// The headline figures are F and PValue. The remaining fields are exposed for
// diagnostics and reproducibility: they let a caller re-derive the F-statistic
// (RSSRestricted, RSSUnrestricted, DFNumerator, DFDenominator) and understand
// how much data actually entered the regression (Observations).
type GrangerResult struct {
	// F is the F-statistic testing the joint hypothesis that all lag
	// coefficients of the cause series are zero. Larger F means the cause lags
	// explain more of the effect's variance than its own past alone.
	F float64

	// PValue is P(F_{DFNumerator, DFDenominator} > F): the probability of an
	// F-statistic at least this large under the null hypothesis of no Granger
	// causality. Small values (e.g. < 0.05) reject the null.
	PValue float64

	// Lags is the number of lags used for both series (as requested).
	Lags int

	// Observations is the number of rows in the regression, len(effect) − Lags.
	Observations int

	// RSSRestricted is the residual sum of squares of the restricted model,
	// which regresses effect on a constant and its own Lags lags.
	RSSRestricted float64

	// RSSUnrestricted is the residual sum of squares of the unrestricted model,
	// which adds the Lags lags of the cause series to the restricted model.
	RSSUnrestricted float64

	// DFNumerator is the numerator degrees of freedom of the F-test, equal to
	// Lags (the number of restrictions imposed by the null).
	DFNumerator int

	// DFDenominator is the denominator degrees of freedom, equal to
	// Observations − (2*Lags + 1), the residual df of the unrestricted model.
	DFDenominator int
}

// String renders the result in a single compact, human-readable line.
func (r *GrangerResult) String() string {
	return fmt.Sprintf(
		"Granger F(%d,%d)=%.4f, p=%.4g (lags=%d, n=%d, RSS restricted=%.6g, unrestricted=%.6g)",
		r.DFNumerator, r.DFDenominator, r.F, r.PValue,
		r.Lags, r.Observations, r.RSSRestricted, r.RSSUnrestricted,
	)
}

// GrangerTest performs a pairwise Granger causality test asking whether the
// past of cause helps predict effect beyond what effect's own past already
// explains.
//
// Direction. The test is directional and NOT symmetric: GrangerTest(cause,
// effect, lags) tests "cause → effect". Concretely it compares two ordinary
// least-squares autoregressions of effect on the observations t = lags ..
// len(effect)−1:
//
//	restricted:    effect[t] = c + Σ_{i=1..lags} a_i·effect[t−i] + e
//	unrestricted:  effect[t] = c + Σ_{i=1..lags} a_i·effect[t−i]
//	                             + Σ_{i=1..lags} b_i·cause[t−i]  + e
//
// The null hypothesis is b_1 = … = b_lags = 0 (cause adds nothing). The test
// statistic is
//
//	F = ((RSS_r − RSS_u)/lags) / (RSS_u/(n − k_u))
//
// with n = len(effect) − lags observations and k_u = 2*lags + 1 parameters in
// the unrestricted model, so the numerator df is lags and the denominator df
// is n − k_u. Under the null and Gaussian errors, F follows an
// F(lags, n − k_u) distribution; PValue is its upper-tail probability.
//
// To test the reverse direction, call GrangerTest(effect, cause, lags).
//
// Both regressions are fit with a Householder QR factorization (see the
// package's OLS solver) for numerical stability on the collinear designs that
// stacking a series with its own lags tends to produce.
//
// LIMITATION — confounding. Granger causality measures predictive precedence,
// not mechanism. If an unobserved third series Z drives both cause and effect
// with cause leading effect in time, this test will report that cause Granger-
// causes effect even though the true structure is Z → cause and Z → effect
// with no direct cause → effect link. That is a property of the method, not a
// bug: distinguishing such confounded structures requires conditional-
// independence search (the PC algorithm) or non-Gaussian structural methods
// (LiNGAM), which are on the causa roadmap. Treat a positive Granger result as
// "cause carries leading predictive information about effect", which is
// necessary but not sufficient for a causal claim.
//
// Errors: ErrInvalidLags (lags < 1), ErrMismatchedLength (unequal lengths),
// ErrTooShort (len < 3*lags + 2), ErrNonFinite (NaN/Inf in the data), and
// ErrSingular (a constant or collinear series makes a design rank deficient).
func GrangerTest(cause, effect []float64, lags int) (*GrangerResult, error) {
	if lags < 1 {
		return nil, ErrInvalidLags
	}
	if len(cause) != len(effect) {
		return nil, ErrMismatchedLength
	}
	nTotal := len(effect)
	// Unrestricted model has k_u = 2*lags + 1 parameters over n = nTotal − lags
	// rows; a positive residual df (n − k_u >= 1) needs nTotal >= 3*lags + 2.
	if nTotal < 3*lags+2 {
		return nil, ErrTooShort
	}
	for i := 0; i < nTotal; i++ {
		if !isFinite(cause[i]) || !isFinite(effect[i]) {
			return nil, ErrNonFinite
		}
	}

	n := nTotal - lags
	kRestricted := lags + 1
	kUnrestricted := 2*lags + 1

	// Response and the two design matrices. Row r corresponds to time
	// t = r + lags; column 0 is the intercept, then the effect's own lags,
	// then (unrestricted only) the cause's lags. Lag i uses value at t−i, so
	// index (r + lags) − i.
	y := make([]float64, n)
	restricted := make([][]float64, n)
	unrestricted := make([][]float64, n)
	for r := 0; r < n; r++ {
		t := r + lags
		y[r] = effect[t]

		rr := make([]float64, kRestricted)
		ur := make([]float64, kUnrestricted)
		rr[0] = 1
		ur[0] = 1
		for i := 1; i <= lags; i++ {
			rr[i] = effect[t-i]
			ur[i] = effect[t-i]
			ur[lags+i] = cause[t-i]
		}
		restricted[r] = rr
		unrestricted[r] = ur
	}

	fitR, err := fitOLS(restricted, y)
	if err != nil {
		return nil, err
	}
	fitU, err := fitOLS(unrestricted, y)
	if err != nil {
		return nil, err
	}

	df1 := lags
	df2 := n - kUnrestricted

	// Numerator: extra variance explained by the cause lags. Clamp tiny
	// negative values from floating-point rounding (RSS_u can never exceed
	// RSS_r in exact arithmetic, since the unrestricted model nests the
	// restricted one).
	diff := fitR.rss - fitU.rss
	if diff < 0 {
		diff = 0
	}

	var f, p float64
	switch {
	case fitU.rss <= 0:
		// Unrestricted model fits perfectly. If it explains strictly more than
		// the restricted model, the statistic diverges; otherwise both fit
		// perfectly and the cause adds nothing.
		if diff > 0 {
			f = math.Inf(1)
			p = 0
		} else {
			f = 0
			p = 1
		}
	default:
		f = (diff / float64(df1)) / (fitU.rss / float64(df2))
		p = fUpperTail(f, float64(df1), float64(df2))
	}

	return &GrangerResult{
		F:               f,
		PValue:          p,
		Lags:            lags,
		Observations:    n,
		RSSRestricted:   fitR.rss,
		RSSUnrestricted: fitU.rss,
		DFNumerator:     df1,
		DFDenominator:   df2,
	}, nil
}

// isFinite reports whether v is a finite float64 (neither NaN nor ±Inf).
func isFinite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}
