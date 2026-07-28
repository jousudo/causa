package causa

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

// gaussianRows draws nobs i.i.d. rows from N(0, cov) and returns them column-major
// (out[v] is variable v's sample), using the package's own Cholesky factor.
func gaussianRows(cov [][]float64, nobs int, rng *rand.Rand) [][]float64 {
	l, ok := cholesky(cov)
	if !ok {
		panic("test cov not PD")
	}
	p := len(cov)
	out := make([][]float64, p)
	for v := range out {
		out[v] = make([]float64, nobs)
	}
	z := make([]float64, p)
	for k := 0; k < nobs; k++ {
		for i := 0; i < p; i++ {
			z[i] = rng.NormFloat64()
		}
		for i := 0; i < p; i++ { // row = L·z
			var s float64
			for j := 0; j <= i; j++ {
				s += l[i][j] * z[j]
			}
			out[i][k] = s
		}
	}
	return out
}

func TestSampleGaussian(t *testing.T) {
	cov := [][]float64{{2, 1}, {1, 1}}
	data := gaussianRows(cov, 20000, rand.New(rand.NewSource(42)))
	g, err := SampleGaussian(data)
	if err != nil {
		t.Fatal(err)
	}
	// Sample estimates converge to the truth; a loose tolerance suffices.
	for i := 0; i < 2; i++ {
		if math.Abs(g.mean[i]) > 0.1 {
			t.Errorf("mean[%d] = %g, want ~0", i, g.mean[i])
		}
		for j := 0; j < 2; j++ {
			if math.Abs(g.cov[i][j]-cov[i][j]) > 0.1 {
				t.Errorf("cov[%d][%d] = %g, want %g", i, j, g.cov[i][j], cov[i][j])
			}
		}
	}
}

func TestSampleGaussianErrors(t *testing.T) {
	tests := []struct {
		name string
		data [][]float64
		want error
	}{
		{"no vars", nil, ErrBadGaussian},
		{"too few samples", [][]float64{{1}}, ErrTooFewSamples},
		{"unequal", [][]float64{{1, 2, 3}, {1, 2}}, ErrUnequalLengths},
		{"nonfinite", [][]float64{{1, math.Inf(1), 3}}, ErrNonFinite},
		{"collinear", [][]float64{{1, 2, 3}, {2, 4, 6}}, ErrNotPositiveDefinite},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := SampleGaussian(tc.data); !errors.Is(err, tc.want) {
				t.Errorf("got %v, want %v", err, tc.want)
			}
		})
	}
}

func TestSampleDistribution(t *testing.T) {
	// Two binary variables; X=0, Y=1. Empirical frequencies of a fixed set of rows.
	data := [][]int{
		{0, 0, 1, 1, 1}, // X
		{0, 1, 0, 1, 1}, // Y
	}
	d, err := SampleDistribution(data, []int{2, 2})
	if err != nil {
		t.Fatal(err)
	}
	// Counts: (0,0)=1 (0,1)=1 (1,0)=1 (1,1)=2 over 5 → mixed-radix index X*2+Y.
	want := []float64{0.2, 0.2, 0.2, 0.4}
	for i := range want {
		if math.Abs(d.prob[i]-want[i]) > 1e-12 {
			t.Errorf("prob[%d] = %g, want %g", i, d.prob[i], want[i])
		}
	}
	if _, err := SampleDistribution([][]int{{0, 2}}, []int{2}); !errors.Is(err, ErrBadDistribution) {
		t.Errorf("out-of-range state should be ErrBadDistribution")
	}
	if _, err := SampleDistribution([][]int{{0, 1, 0}, {0, 1}}, []int{2, 2}); !errors.Is(err, ErrUnequalLengths) {
		t.Errorf("unequal lengths should be ErrUnequalLengths")
	}
}

// The bootstrap of a plain column mean must (a) be reproducible under a fixed seed,
// (b) report the full-sample mean as its point estimate, and (c) recover the
// analytic standard error σ/√n.
func TestBootstrapMean(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const n = 800
	col := make([]float64, n)
	for i := range col {
		col[i] = rng.NormFloat64()
	}
	mean := func(idx []int) (float64, error) {
		var s float64
		for _, r := range idx {
			s += col[r]
		}
		return s / float64(len(idx)), nil
	}
	opts := BootstrapOptions{Resamples: 2000, Level: 0.95, Seed: 1}
	r1, err := Bootstrap(n, mean, opts)
	if err != nil {
		t.Fatal(err)
	}
	r2, _ := Bootstrap(n, mean, opts)
	if r1.Lower != r2.Lower || r1.Upper != r2.Upper || r1.StdErr != r2.StdErr {
		t.Errorf("same seed must be reproducible: %+v vs %+v", r1, r2)
	}
	full, _ := mean(seq(n))
	if math.Abs(r1.Point-full) > 1e-12 {
		t.Errorf("point = %g, want full-sample mean %g", r1.Point, full)
	}
	// Analytic SE of the mean is sd(col)/√n; the bootstrap SE should be close.
	se := stddev(col) / math.Sqrt(n)
	if math.Abs(r1.StdErr-se) > 0.2*se {
		t.Errorf("bootstrap SE = %g, analytic ≈ %g", r1.StdErr, se)
	}
	if !(r1.Lower < r1.Point && r1.Point < r1.Upper) {
		t.Errorf("interval [%g, %g] must bracket the point %g", r1.Lower, r1.Upper, r1.Point)
	}
}

func TestBootstrapErrors(t *testing.T) {
	ok := func(idx []int) (float64, error) { return 1, nil }
	if _, err := Bootstrap(0, ok, BootstrapOptions{}); !errors.Is(err, ErrTooFewSamples) {
		t.Errorf("n<1 should be ErrTooFewSamples, got %v", err)
	}
	if _, err := Bootstrap(10, ok, BootstrapOptions{Level: 1.5}); !errors.Is(err, ErrBootstrap) {
		t.Errorf("bad level should be ErrBootstrap, got %v", err)
	}
	// A statistic that always errors: point estimate fails → propagated.
	bad := func(idx []int) (float64, error) { return 0, ErrSingular }
	if _, err := Bootstrap(10, bad, BootstrapOptions{Resamples: 10}); !errors.Is(err, ErrSingular) {
		t.Errorf("point-estimate error should propagate, got %v", err)
	}
	// Point estimate succeeds but every resample fails → ErrBootstrap.
	calls := 0
	flaky := func(idx []int) (float64, error) {
		calls++
		if calls == 1 { // the identity (point) call
			return 1, nil
		}
		return 0, ErrSingular
	}
	if _, err := Bootstrap(10, flaky, BootstrapOptions{Resamples: 50, Seed: 3}); !errors.Is(err, ErrBootstrap) {
		t.Errorf("all-resamples-fail should be ErrBootstrap, got %v", err)
	}
}

// BootstrapGaussianEffect's point estimate must equal the effect evaluated on the
// full-sample Gaussian, and the interval must bracket it.
func TestBootstrapGaussianEffectPoint(t *testing.T) {
	// Back-door X→Y confounded by observed Z. X=0, Y=1, Z=2.
	g, _ := NewDiagram([]string{"X", "Y", "Z"}, [][2]int{{0, 1}, {2, 0}, {2, 1}}, nil)
	res, _ := Identify(g, []int{1}, []int{0})
	// The Σ implied by Z→X, Z→Y, X→Y (effect 2), from ExampleExpr_EvaluateGaussian.
	cov := [][]float64{{2, 5, 1}, {5, 14, 3}, {1, 3, 1}}
	data := gaussianRows(cov, 4000, rand.New(rand.NewSource(11)))

	ci, err := res.Estimand.BootstrapGaussianEffect(data, 0, 1, BootstrapOptions{Resamples: 400, Seed: 2})
	if err != nil {
		t.Fatal(err)
	}
	// Full-sample point estimate.
	gd, _ := SampleGaussian(data)
	f, _ := res.Estimand.EvaluateGaussian(gd)
	want, _ := gaussianEffect(f, 0, 1)
	if math.Abs(ci.Point-want) > 1e-12 {
		t.Errorf("point = %g, want full-sample effect %g", ci.Point, want)
	}
	if !(ci.Lower < ci.Point && ci.Point < ci.Upper) {
		t.Errorf("interval [%g, %g] must bracket the point %g", ci.Lower, ci.Upper, ci.Point)
	}
	// With n=4000 the estimate should be near the structural truth of 2.
	if math.Abs(ci.Point-2) > 0.25 {
		t.Errorf("effect point %g not near structural truth 2", ci.Point)
	}
}

func TestBootstrapGaussianEffectErrors(t *testing.T) {
	e := jointExpr([]int{0, 1})
	if _, err := e.BootstrapGaussianEffect([][]float64{{1, 2, 3}}, 0, 1, BootstrapOptions{}); !errors.Is(err, ErrTooFewVariables) {
		t.Errorf("single column should be ErrTooFewVariables, got %v", err)
	}
	data := [][]float64{{1, 2, 3}, {4, 5, 6}}
	if _, err := e.BootstrapGaussianEffect(data, 0, 5, BootstrapOptions{}); !errors.Is(err, ErrBadGaussian) {
		t.Errorf("out-of-range y should be ErrBadGaussian, got %v", err)
	}
	if _, err := e.BootstrapGaussianEffect(data, 1, 1, BootstrapOptions{}); !errors.Is(err, ErrBadGaussian) {
		t.Errorf("x==y should be ErrBadGaussian, got %v", err)
	}
}

func seq(n int) []int {
	s := make([]int, n)
	for i := range s {
		s[i] = i
	}
	return s
}
