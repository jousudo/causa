package causa

import (
	"errors"
	"math"
	"testing"
)

// hand joint N(0, Σ) over X=0, Y=1 with Σ = [[2,1],[1,1]].
func handGaussXY() *GaussianDistribution {
	d, err := NewGaussian([]float64{0, 0}, [][]float64{{2, 1}, {1, 1}})
	if err != nil {
		panic(err)
	}
	return d
}

func approxGauss(t *testing.T, got *GaussianDistribution, wantVars []int, wantMean []float64, wantCov [][]float64) {
	t.Helper()
	if !sortedEqual(got.Vars(), wantVars) {
		t.Fatalf("vars = %v, want %v", got.Vars(), wantVars)
	}
	for i := range wantMean {
		if math.Abs(got.mean[i]-wantMean[i]) > 1e-9 {
			t.Errorf("mean[%d] = %g, want %g", i, got.mean[i], wantMean[i])
		}
		for j := range wantMean {
			if math.Abs(got.cov[i][j]-wantCov[i][j]) > 1e-9 {
				t.Errorf("cov[%d][%d] = %g, want %g", i, j, got.cov[i][j], wantCov[i][j])
			}
		}
	}
}

func TestNewGaussianErrors(t *testing.T) {
	tests := []struct {
		name string
		mean []float64
		cov  [][]float64
		want error
	}{
		{"empty", nil, nil, ErrBadGaussian},
		{"cov wrong rows", []float64{0, 0}, [][]float64{{1, 0}}, ErrBadGaussian},
		{"cov wrong cols", []float64{0, 0}, [][]float64{{1, 0}, {0}}, ErrBadGaussian},
		{"nonfinite mean", []float64{math.Inf(1), 0}, [][]float64{{1, 0}, {0, 1}}, ErrBadGaussian},
		{"nonfinite cov", []float64{0, 0}, [][]float64{{math.NaN(), 0}, {0, 1}}, ErrBadGaussian},
		{"asymmetric", []float64{0, 0}, [][]float64{{1, 0.5}, {0.4, 1}}, ErrBadGaussian},
		{"not PD", []float64{0, 0}, [][]float64{{1, 2}, {2, 1}}, ErrNotPositiveDefinite},
		{"zero variance", []float64{0, 0}, [][]float64{{0, 0}, {0, 1}}, ErrNotPositiveDefinite},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NewGaussian(tc.mean, tc.cov); !errors.Is(err, tc.want) {
				t.Errorf("got %v, want %v", err, tc.want)
			}
		})
	}
}

// The observational joint itself is the estimand P(V) — EvaluateGaussian on the
// bare joint, conditioned on nothing, must reproduce the input mean and covariance.
func TestGaussIdentityRoundTrip(t *testing.T) {
	d := handGaussXY()
	f, err := jointExpr([]int{0, 1}).EvaluateGaussian(d)
	if err != nil {
		t.Fatal(err)
	}
	got, err := f.Condition(nil)
	if err != nil {
		t.Fatal(err)
	}
	approxGauss(t, got, []int{0, 1}, []float64{0, 0}, [][]float64{{2, 1}, {1, 1}})
}

// Σ_X P(X,Y) = P(Y): marginalizing X out of the joint yields N(0, Σ_YY=1).
func TestGaussMarginal(t *testing.T) {
	d := handGaussXY()
	e := marginalExpr([]int{0}, jointExpr([]int{0, 1}))
	f, err := e.EvaluateGaussian(d)
	if err != nil {
		t.Fatal(err)
	}
	got, err := f.Condition(nil)
	if err != nil {
		t.Fatal(err)
	}
	approxGauss(t, got, []int{1}, []float64{0}, [][]float64{{1}})
}

// P(Y|X): the Gaussian conditional. For Σ=[[2,1],[1,1]], E[Y|X=x] = 0.5·x and
// Var(Y|X) = 1 − 1²/2 = 0.5, independent of x.
func TestGaussConditional(t *testing.T) {
	d := handGaussXY()
	e := condition(jointExpr([]int{0, 1}), []int{1}, []int{0}, []int{0, 1})
	f, err := e.EvaluateGaussian(d)
	if err != nil {
		t.Fatal(err)
	}
	for _, x := range []float64{-3, 0, 2.5} {
		got, err := f.Condition(map[int]float64{0: x})
		if err != nil {
			t.Fatal(err)
		}
		approxGauss(t, got, []int{1}, []float64{0.5 * x}, [][]float64{{0.5}})
	}
}

// Chain rule: Σ_X P(X)·P(Y|X) must reproduce the marginal P(Y) = N(0, 1),
// exercising product, conditional and marginal together (the TestEvalProductChain
// analogue).
func TestGaussProductChain(t *testing.T) {
	d := handGaussXY()
	univ := []int{0, 1}
	px := condition(jointExpr([]int{0, 1}), []int{0}, nil, univ)            // P(X)
	pyGivenX := condition(jointExpr([]int{0, 1}), []int{1}, []int{0}, univ) // P(Y|X)
	e := marginalExpr([]int{0}, productExpr([]*Expr{px, pyGivenX}))
	f, err := e.EvaluateGaussian(d)
	if err != nil {
		t.Fatal(err)
	}
	got, err := f.Condition(nil)
	if err != nil {
		t.Fatal(err)
	}
	approxGauss(t, got, []int{1}, []float64{0}, [][]float64{{1}})
}

func TestGaussConditionErrors(t *testing.T) {
	f, err := jointExpr([]int{0, 1}).EvaluateGaussian(handGaussXY())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Condition(map[int]float64{9: 1}); !errors.Is(err, ErrBadGaussian) {
		t.Errorf("unknown variable: got %v, want ErrBadGaussian", err)
	}
	if _, err := f.Condition(map[int]float64{0: math.Inf(1)}); !errors.Is(err, ErrBadGaussian) {
		t.Errorf("non-finite value: got %v, want ErrBadGaussian", err)
	}
}

func TestEvaluateGaussianNotFullJoint(t *testing.T) {
	// A GaussianDistribution over {1,2} (not 0..n-1) is rejected — the same guard as
	// the discrete Evaluate.
	d := &GaussianDistribution{vars: []int{1, 2}, mean: []float64{0, 0}, cov: [][]float64{{1, 0}, {0, 1}}}
	if _, err := jointExpr([]int{1, 2}).EvaluateGaussian(d); !errors.Is(err, ErrBadGaussian) {
		t.Errorf("got %v, want ErrBadGaussian", err)
	}
}
