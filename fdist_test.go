package causa

import (
	"math"
	"testing"
)

// The regularized incomplete beta function has closed forms for special
// parameters; these are exact identities (not tabulated approximations), so we
// hold the implementation to machine precision against them.
//
//	I_x(1, 1)     = x
//	I_x(2, 1)     = x^2
//	I_x(1, 2)     = 2x − x^2
//	I_x(a, 1)     = x^a
//	I_x(1/2, 1/2) = (2/π)·arcsin(√x)
//
// Sources: Abramowitz & Stegun §6.6 (incomplete beta), §26.6; DLMF §8.17.
func TestRegIncBetaClosedForms(t *testing.T) {
	const tol = 1e-12
	for _, x := range []float64{0.05, 0.2, 0.37, 0.5, 0.63, 0.8, 0.95} {
		if got, want := regIncBeta(1, 1, x), x; math.Abs(got-want) > tol {
			t.Errorf("I_%v(1,1) = %v, want %v", x, got, want)
		}
		if got, want := regIncBeta(2, 1, x), x*x; math.Abs(got-want) > tol {
			t.Errorf("I_%v(2,1) = %v, want %v", x, got, want)
		}
		if got, want := regIncBeta(1, 2, x), 2*x-x*x; math.Abs(got-want) > tol {
			t.Errorf("I_%v(1,2) = %v, want %v", x, got, want)
		}
		if got, want := regIncBeta(3.5, 1, x), math.Pow(x, 3.5); math.Abs(got-want) > tol {
			t.Errorf("I_%v(3.5,1) = %v, want %v", x, got, want)
		}
		if got, want := regIncBeta(0.5, 0.5, x), (2/math.Pi)*math.Asin(math.Sqrt(x)); math.Abs(got-want) > tol {
			t.Errorf("I_%v(0.5,0.5) = %v, want %v", x, got, want)
		}
	}
}

func TestRegIncBetaBoundsAndSymmetry(t *testing.T) {
	if got := regIncBeta(2, 3, 0); got != 0 {
		t.Errorf("I_0(2,3) = %v, want 0", got)
	}
	if got := regIncBeta(2, 3, 1); got != 1 {
		t.Errorf("I_1(2,3) = %v, want 1", got)
	}
	// Reflection identity I_x(a,b) = 1 − I_{1−x}(b,a).
	const tol = 1e-13
	for _, tc := range []struct{ a, b, x float64 }{
		{2, 5, 0.3}, {4.5, 1.5, 0.72}, {10, 3, 0.4},
	} {
		lhs := regIncBeta(tc.a, tc.b, tc.x)
		rhs := 1 - regIncBeta(tc.b, tc.a, 1-tc.x)
		if math.Abs(lhs-rhs) > tol {
			t.Errorf("reflection failed for (a=%v,b=%v,x=%v): %v vs %v", tc.a, tc.b, tc.x, lhs, rhs)
		}
	}
}

// The F-distribution upper tail has an exact closed form when df1 = 2:
//
//	P(F_{2,d} > f) = (d/(d + 2f))^{d/2}.
//
// (From I_x(d/2, 1) = x^{d/2} with x = d/(d+2f).) Machine-precision check.
// Source: Abramowitz & Stegun §26.6.2 together with §6.6.
func TestFUpperTailClosedFormDf1Is2(t *testing.T) {
	const tol = 1e-12
	for _, d := range []float64{2, 5, 8, 20, 100} {
		for _, f := range []float64{0.25, 1, 2.5, 7} {
			want := math.Pow(d/(d+2*f), d/2)
			got := fUpperTail(f, 2, d)
			if math.Abs(got-want) > tol {
				t.Errorf("P(F_{2,%v} > %v) = %v, want %v", d, f, got, want)
			}
		}
	}
	// f = 0 and below is the whole mass.
	if got := fUpperTail(0, 3, 10); got != 1 {
		t.Errorf("P(F > 0) = %v, want 1", got)
	}
}

// Upper 5% and 1% critical points from standard F-distribution tables. At the
// tabulated critical value the survival function must return the corresponding
// alpha. Values transcribed from the NIST/SEMATECH e-Handbook of Statistical
// Methods, §1.3.6.7.3 (F-distribution tables); they agree with Rohlf & Sokal,
// "Statistical Tables". Table entries carry 4 decimals, so the tolerance is
// loosened to 1e-3 to absorb that rounding.
func TestFUpperTailAgainstPublishedCriticalValues(t *testing.T) {
	const tol = 1e-3
	cases := []struct {
		df1, df2  float64
		crit      float64
		wantAlpha float64
	}{
		{1, 10, 4.9646, 0.05},
		{2, 10, 4.1028, 0.05},
		{3, 20, 3.0984, 0.05},
		{4, 15, 3.0556, 0.05},
		{5, 30, 2.5336, 0.05},
		{10, 20, 2.3479, 0.05},
		{1, 10, 10.0443, 0.01},
		{3, 20, 4.9382, 0.01},
	}
	for _, tc := range cases {
		got := fUpperTail(tc.crit, tc.df1, tc.df2)
		if math.Abs(got-tc.wantAlpha) > tol {
			t.Errorf("P(F_{%v,%v} > %v) = %v, want ~%v", tc.df1, tc.df2, tc.crit, got, tc.wantAlpha)
		}
	}
}

func TestFUpperTailMonotoneDecreasing(t *testing.T) {
	prev := 2.0
	for f := 0.1; f < 20; f += 0.1 {
		p := fUpperTail(f, 4, 25)
		if p > prev {
			t.Fatalf("survival not monotone: P(F>%v)=%v exceeded previous %v", f, p, prev)
		}
		if p < 0 || p > 1 {
			t.Fatalf("survival out of [0,1] at f=%v: %v", f, p)
		}
		prev = p
	}
}
