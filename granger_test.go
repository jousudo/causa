package causa

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

// --- Independent reference implementation -------------------------------
//
// referenceGranger recomputes the Granger F-statistic through a completely
// separate numerical path from the production code: it forms the normal
// equations XᵀX·b = Xᵀy and solves them by Gaussian elimination with partial
// pivoting, taking RSS = yᵀy − bᵀXᵀy. The production code instead uses a
// Householder QR factorization and never forms XᵀX. Agreement between the two
// (to ~1e-8) is a cross-implementation validation of the regression numerics:
// two independent linear-algebra routines, no shared code, must land on the
// same F. This stands in for a statsmodels/R oracle, which is not available in
// this offline build.

func solveNormalEquations(x [][]float64, y []float64) (coef []float64, rss float64) {
	n := len(x)
	k := len(x[0])
	// A = XᵀX (k×k), rhs = Xᵀy (k).
	a := make([][]float64, k)
	for i := range a {
		a[i] = make([]float64, k)
	}
	rhs := make([]float64, k)
	for r := 0; r < n; r++ {
		for i := 0; i < k; i++ {
			rhs[i] += x[r][i] * y[r]
			for j := 0; j < k; j++ {
				a[i][j] += x[r][i] * x[r][j]
			}
		}
	}
	// Gaussian elimination with partial pivoting on [a | rhs].
	for col := 0; col < k; col++ {
		piv := col
		for r := col + 1; r < k; r++ {
			if math.Abs(a[r][col]) > math.Abs(a[piv][col]) {
				piv = r
			}
		}
		a[col], a[piv] = a[piv], a[col]
		rhs[col], rhs[piv] = rhs[piv], rhs[col]
		for r := col + 1; r < k; r++ {
			f := a[r][col] / a[col][col]
			for c := col; c < k; c++ {
				a[r][c] -= f * a[col][c]
			}
			rhs[r] -= f * rhs[col]
		}
	}
	coef = make([]float64, k)
	for i := k - 1; i >= 0; i-- {
		s := rhs[i]
		for j := i + 1; j < k; j++ {
			s -= a[i][j] * coef[j]
		}
		coef[i] = s / a[i][i]
	}
	var yty, bxty float64
	for r := 0; r < n; r++ {
		yty += y[r] * y[r]
	}
	// bᵀXᵀy: recompute Xᵀy fresh (rhs was overwritten by elimination).
	for i := 0; i < k; i++ {
		var xtyi float64
		for r := 0; r < n; r++ {
			xtyi += x[r][i] * y[r]
		}
		bxty += coef[i] * xtyi
	}
	rss = yty - bxty
	return coef, rss
}

func referenceGranger(cause, effect []float64, lags int) (fStat, rssR, rssU float64) {
	nTotal := len(effect)
	n := nTotal - lags
	kR := lags + 1
	kU := 2*lags + 1
	y := make([]float64, n)
	xr := make([][]float64, n)
	xu := make([][]float64, n)
	for r := 0; r < n; r++ {
		t := r + lags
		y[r] = effect[t]
		rr := make([]float64, kR)
		ur := make([]float64, kU)
		rr[0], ur[0] = 1, 1
		for i := 1; i <= lags; i++ {
			rr[i] = effect[t-i]
			ur[i] = effect[t-i]
			ur[lags+i] = cause[t-i]
		}
		xr[r], xu[r] = rr, ur
	}
	_, rssR = solveNormalEquations(xr, y)
	_, rssU = solveNormalEquations(xu, y)
	df1 := float64(lags)
	df2 := float64(n - kU)
	fStat = ((rssR - rssU) / df1) / (rssU / df2)
	return fStat, rssR, rssU
}

// --- Synthetic ground-truth generators ----------------------------------

// genDriven builds effect[t] = 0.5·effect[t-1] + 0.6·cause[t-1] + noise with an
// independent AR(1) cause. cause genuinely drives effect one step ahead.
func genDriven(seed int64, n int) (cause, effect []float64) {
	rng := rand.New(rand.NewSource(seed))
	cause = make([]float64, n)
	effect = make([]float64, n)
	for t := 1; t < n; t++ {
		cause[t] = 0.4*cause[t-1] + rng.NormFloat64()
		effect[t] = 0.5*effect[t-1] + 0.6*cause[t-1] + 0.5*rng.NormFloat64()
	}
	return cause, effect
}

// genIndependent builds two independent AR(1) series that share no dynamics.
func genIndependent(seed int64, n int) (a, b []float64) {
	rng := rand.New(rand.NewSource(seed))
	a = make([]float64, n)
	b = make([]float64, n)
	for t := 1; t < n; t++ {
		a[t] = 0.5*a[t-1] + rng.NormFloat64()
		b[t] = 0.5*b[t-1] + rng.NormFloat64()
	}
	return a, b
}

// genConfounded builds a hidden common cause z (AR(1)) that drives BOTH x and y
// one step ahead, with NO direct x→y edge. Granger is expected to flag x→y
// anyway — the confounding limitation documented on GrangerTest.
func genConfounded(seed int64, n int) (x, y []float64) {
	rng := rand.New(rand.NewSource(seed))
	z := make([]float64, n)
	x = make([]float64, n)
	y = make([]float64, n)
	for t := 1; t < n; t++ {
		z[t] = 0.8*z[t-1] + rng.NormFloat64()
		x[t] = 0.9*z[t-1] + 0.3*rng.NormFloat64()
		y[t] = 0.9*z[t-1] + 0.3*rng.NormFloat64()
	}
	return x, y
}

// --- Cross-implementation validation ------------------------------------

func TestGrangerMatchesIndependentReference(t *testing.T) {
	cause, effect := genDriven(1, 400)
	for _, lags := range []int{1, 2, 4} {
		res, err := GrangerTest(cause, effect, lags)
		if err != nil {
			t.Fatalf("lags=%d: unexpected error: %v", lags, err)
		}
		wantF, wantRSSr, wantRSSu := referenceGranger(cause, effect, lags)

		if rel(res.F, wantF) > 1e-8 {
			t.Errorf("lags=%d: F=%.10f, reference=%.10f (rel %.2e)", lags, res.F, wantF, rel(res.F, wantF))
		}
		if rel(res.RSSRestricted, wantRSSr) > 1e-8 {
			t.Errorf("lags=%d: RSS_r=%.10f, reference=%.10f", lags, res.RSSRestricted, wantRSSr)
		}
		if rel(res.RSSUnrestricted, wantRSSu) > 1e-8 {
			t.Errorf("lags=%d: RSS_u=%.10f, reference=%.10f", lags, res.RSSUnrestricted, wantRSSu)
		}
	}
}

// TestGrangerReferenceConstants locks the exact statistics of a fixed dataset
// so any future numerical drift is caught. The constants below are produced by
// the independent normal-equations reference above (provenance:
// referenceGranger on genDriven(1, 400) with lags=2), and the p-value is the
// F-distribution upper tail whose accuracy is independently pinned in
// fdist_test.go against closed forms and published critical values.
func TestGrangerReferenceConstants(t *testing.T) {
	cause, effect := genDriven(1, 400)
	res, err := GrangerTest(cause, effect, 2)
	if err != nil {
		t.Fatal(err)
	}
	// Values produced by referenceGranger (independent normal-equations solver)
	// on genDriven(1, 400) with lags=2; the QR production path reproduces them
	// to 10 significant digits.
	const (
		wantF    = 304.8152743034
		wantRSSr = 253.0307579256
		wantRSSu = 99.1801895553
	)
	if rel(res.F, wantF) > 1e-6 {
		t.Errorf("F drift: got %.10f, locked %.10f", res.F, wantF)
	}
	if rel(res.RSSRestricted, wantRSSr) > 1e-6 {
		t.Errorf("RSS_r drift: got %.10f, locked %.10f", res.RSSRestricted, wantRSSr)
	}
	if rel(res.RSSUnrestricted, wantRSSu) > 1e-6 {
		t.Errorf("RSS_u drift: got %.10f, locked %.10f", res.RSSUnrestricted, wantRSSu)
	}
}

// --- Behavioral ground-truth --------------------------------------------

func TestGrangerDetectsGenuineCause(t *testing.T) {
	cause, effect := genDriven(7, 500)
	res, err := GrangerTest(cause, effect, 2)
	if err != nil {
		t.Fatal(err)
	}
	if res.PValue >= 0.01 {
		t.Errorf("expected to detect genuine cause (p<0.01), got p=%.4g (F=%.4f)", res.PValue, res.F)
	}
	// The reverse direction (effect → cause) must NOT be significant: effect
	// does not drive cause in this generator.
	rev, err := GrangerTest(effect, cause, 2)
	if err != nil {
		t.Fatal(err)
	}
	if rev.PValue < 0.05 {
		t.Errorf("reverse direction falsely significant: p=%.4g (F=%.4f)", rev.PValue, rev.F)
	}
}

func TestGrangerIgnoresIndependentSeries(t *testing.T) {
	a, b := genIndependent(20, 600)
	res, err := GrangerTest(a, b, 2)
	if err != nil {
		t.Fatal(err)
	}
	if res.PValue < 0.05 {
		t.Errorf("independent series falsely significant: p=%.4g (F=%.4f)", res.PValue, res.F)
	}
}

// TestGrangerFlagsConfounder documents the method's central limitation: with a
// hidden common cause and no direct edge, Granger still reports causality.
func TestGrangerFlagsConfounder(t *testing.T) {
	x, y := genConfounded(3, 600)
	res, err := GrangerTest(x, y, 2)
	if err != nil {
		t.Fatal(err)
	}
	if res.PValue >= 0.05 {
		t.Errorf("confounder case unexpectedly NOT flagged (p=%.4g); the documented "+
			"limitation is that Granger DOES flag it", res.PValue)
	}
	t.Logf("confounded x→y flagged as expected: F=%.4f, p=%.4g (no direct edge exists)", res.F, res.PValue)
}

// --- Error handling ------------------------------------------------------

func TestGrangerErrors(t *testing.T) {
	ok := make([]float64, 50)
	for i := range ok {
		ok[i] = float64(i%7) + 0.1*float64(i)
	}
	constSeries := make([]float64, 50) // all zero → rank deficient design

	tests := []struct {
		name       string
		cause, eff []float64
		lags       int
		want       error
	}{
		{"invalid lags zero", ok, ok, 0, ErrInvalidLags},
		{"invalid lags negative", ok, ok, -1, ErrInvalidLags},
		{"mismatched length", ok[:40], ok, 2, ErrMismatchedLength},
		{"too short", ok[:6], ok[:6], 2, ErrTooShort},
		{"nan in effect", ok, withNaN(ok), 2, ErrNonFinite},
		{"inf in cause", withInf(ok), ok, 2, ErrNonFinite},
		{"constant effect", ok, constSeries, 2, ErrSingular},
		{"constant cause", constSeries, ok, 2, ErrSingular},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := GrangerTest(tc.cause, tc.eff, tc.lags)
			if !errors.Is(err, tc.want) {
				t.Errorf("got err=%v, want %v", err, tc.want)
			}
		})
	}
}

func TestGrangerResultFieldsConsistent(t *testing.T) {
	cause, effect := genDriven(9, 300)
	res, err := GrangerTest(cause, effect, 3)
	if err != nil {
		t.Fatal(err)
	}
	if res.Lags != 3 {
		t.Errorf("Lags = %d, want 3", res.Lags)
	}
	if res.Observations != 300-3 {
		t.Errorf("Observations = %d, want %d", res.Observations, 297)
	}
	if res.DFNumerator != 3 {
		t.Errorf("DFNumerator = %d, want 3", res.DFNumerator)
	}
	if want := res.Observations - (2*3 + 1); res.DFDenominator != want {
		t.Errorf("DFDenominator = %d, want %d", res.DFDenominator, want)
	}
	if res.RSSUnrestricted > res.RSSRestricted {
		t.Errorf("RSS_u (%.4f) must not exceed RSS_r (%.4f)", res.RSSUnrestricted, res.RSSRestricted)
	}
	if res.String() == "" {
		t.Error("String() returned empty")
	}
}

// --- helpers -------------------------------------------------------------

func rel(got, want float64) float64 {
	d := math.Abs(got - want)
	den := math.Abs(want)
	if den < 1e-300 {
		return d
	}
	return d / den
}

func withNaN(src []float64) []float64 {
	out := append([]float64(nil), src...)
	out[len(out)/2] = math.NaN()
	return out
}

func withInf(src []float64) []float64 {
	out := append([]float64(nil), src...)
	out[len(out)/2] = math.Inf(1)
	return out
}
