package causa

import (
	"errors"
	"math"
	"testing"
)

// --- Numeric oracle ------------------------------------------------------
//
// The constants below are produced by an INDEPENDENT pure-Python (stdlib only,
// no numpy) reference that solves the same least-squares residualization by the
// NORMAL EQUATIONS — a different numerical path from causa's Householder-QR
// residualization — and then applies the Fisher z-transform and the standard-
// normal tail. Agreement to >=10 significant digits cross-validates the CI-test
// numerics against a second implementation.
//
// Provenance. The script below was saved as partialcorr_oracle.py and run once
// with `python3 partialcorr_oracle.py` (CPython 3.x). Its literal x/y/z arrays
// are identical to oracleX/oracleY/oracleZ here. Recorded output:
//
//	r     = 0.1684791495479350
//	z     = 0.1701009680081220
//	stat  = 0.4811181918597564
//	p     = 0.6304324993811583
//
// --- partialcorr_oracle.py ------------------------------------------------
//	import math
//	x = [0.5, 1.8, 1.1, -0.9, 2.7, 0.2, -1.5, 1.9, 2.9, 0.1, 1.2, -0.8]
//	y = [1.2, 2.1, 0.3, -0.7, 2.5, -0.4, -1.9, 1.1, 3.1, 0.6, 0.9, -1.5]
//	z = [1.0, 2.0, 0.5, -1.0, 3.0, 0.0, -2.0, 1.5, 2.5, -0.5, 0.8, -1.2]
//	n = len(x)
//	def residuals_on_z(v):
//	    sz = sum(z); szz = sum(zi*zi for zi in z); sv = sum(v)
//	    szv = sum(zi*vi for zi, vi in zip(z, v))
//	    det = n*szz - sz*sz
//	    b0 = (sv*szz - sz*szv)/det
//	    b1 = (n*szv - sz*sv)/det
//	    return [v[i]-(b0+b1*z[i]) for i in range(n)]
//	rx = residuals_on_z(x); ry = residuals_on_z(y)
//	sxy = sum(a*b for a,b in zip(rx,ry))
//	sxx = sum(a*a for a in rx); syy = sum(b*b for b in ry)
//	r = sxy/math.sqrt(sxx*syy)
//	zt = 0.5*math.log((1+r)/(1-r))
//	stat = math.sqrt(n-1-3)*abs(zt)
//	p = math.erfc(stat/math.sqrt(2.0))
//	print(r, zt, stat, p)
// --------------------------------------------------------------------------

var (
	oracleX = []float64{0.5, 1.8, 1.1, -0.9, 2.7, 0.2, -1.5, 1.9, 2.9, 0.1, 1.2, -0.8}
	oracleY = []float64{1.2, 2.1, 0.3, -0.7, 2.5, -0.4, -1.9, 1.1, 3.1, 0.6, 0.9, -1.5}
	oracleZ = []float64{1.0, 2.0, 0.5, -1.0, 3.0, 0.0, -2.0, 1.5, 2.5, -0.5, 0.8, -1.2}
)

func TestFisherZMatchesPythonOracle(t *testing.T) {
	const (
		wantR    = 0.1684791495479350
		wantZ    = 0.1701009680081220
		wantStat = 0.4811181918597564
		wantP    = 0.6304324993811583
	)
	data := [][]float64{oracleX, oracleY, oracleZ}

	r, err := PartialCorrelation(data, 0, 1, []int{2})
	if err != nil {
		t.Fatal(err)
	}
	if rel(r, wantR) > 1e-10 {
		t.Errorf("partial correlation r = %.16f, oracle %.16f (rel %.2e)", r, wantR, rel(r, wantR))
	}

	// Fisher z and statistic recomputed here from r, cross-checked against the
	// oracle's independent computation.
	z := 0.5 * math.Log((1+r)/(1-r))
	if rel(z, wantZ) > 1e-10 {
		t.Errorf("Fisher z = %.16f, oracle %.16f", z, wantZ)
	}
	n := len(oracleX)
	stat := math.Sqrt(float64(n-1-3)) * math.Abs(z)
	if rel(stat, wantStat) > 1e-10 {
		t.Errorf("statistic = %.16f, oracle %.16f", stat, wantStat)
	}

	p, err := FisherZTest(data, 0, 1, []int{2})
	if err != nil {
		t.Fatal(err)
	}
	if rel(p, wantP) > 1e-10 {
		t.Errorf("p-value = %.16f, oracle %.16f (rel %.2e)", p, wantP, rel(p, wantP))
	}
}

// TestPartialCorrelationEmptyCondIsPearson checks that with no conditioners the
// partial correlation reduces to the ordinary Pearson correlation.
func TestPartialCorrelationEmptyCondIsPearson(t *testing.T) {
	data := [][]float64{oracleX, oracleY}
	got, err := PartialCorrelation(data, 0, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := pearson(oracleX, oracleY)
	if rel(got, want) > 1e-12 {
		t.Errorf("partial correlation (empty cond) = %.12f, Pearson = %.12f", got, want)
	}
}

// TestFisherZDetectsDependence: two strongly linearly related series are flagged
// dependent (small p), and the |r| clamp keeps the statistic finite when the
// relation is exact.
func TestFisherZDependenceAndClamp(t *testing.T) {
	// y is an exact affine function of x: r = 1 after residualizing on the
	// intercept, which the clamp must handle without producing NaN.
	x := []float64{0.5, 1.8, 1.1, -0.9, 2.7, 0.2, -1.5, 1.9, 2.9, 0.1, 1.2, -0.8}
	y := make([]float64, len(x))
	for i := range x {
		y[i] = 2*x[i] + 3
	}
	data := [][]float64{x, y}
	p, err := FisherZTest(data, 0, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(p) || math.IsInf(p, 0) {
		t.Fatalf("p is not finite: %v", p)
	}
	if p >= 1e-6 {
		t.Errorf("exact linear dependence should give p ~ 0, got %.3g", p)
	}
}

func TestFisherZErrors(t *testing.T) {
	// Constant conditioner -> rank-deficient design -> ErrSingular.
	x := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	y := []float64{2, 1, 4, 3, 6, 5, 8, 7}
	constCond := make([]float64, 8)
	if _, err := PartialCorrelation([][]float64{x, y, constCond}, 0, 1, []int{2}); !errors.Is(err, ErrSingular) {
		t.Errorf("constant conditioner: got %v, want ErrSingular", err)
	}

	// A variable perfectly explained by the conditioner -> zero residual ->
	// ErrSingular.
	xdup := append([]float64(nil), x...) // variable 0 equals conditioner 2
	if _, err := PartialCorrelation([][]float64{xdup, y, x}, 0, 1, []int{2}); !errors.Is(err, ErrSingular) {
		t.Errorf("collinear-with-conditioner: got %v, want ErrSingular", err)
	}

	// Not enough residual degrees of freedom: n - |S| - 3 < 1.
	short := []float64{1, 2, 3, 4, 5}
	if _, err := FisherZTest([][]float64{short, short, short, short}, 0, 1, []int{2, 3}); !errors.Is(err, ErrConditioningTooLarge) {
		t.Errorf("oversized conditioning set: got %v, want ErrConditioningTooLarge", err)
	}
}

// pearson computes the plain Pearson correlation, an independent check for the
// empty-conditioning-set case.
func pearson(a, b []float64) float64 {
	n := float64(len(a))
	var sa, sb float64
	for i := range a {
		sa += a[i]
		sb += b[i]
	}
	ma, mb := sa/n, sb/n
	var sab, saa, sbb float64
	for i := range a {
		da, db := a[i]-ma, b[i]-mb
		sab += da * db
		saa += da * da
		sbb += db * db
	}
	return sab / math.Sqrt(saa*sbb)
}
