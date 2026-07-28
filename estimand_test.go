package causa

import (
	"errors"
	"math"
	"testing"
)

// hand joint over X=0, Y=1 (both binary), index = X*2 + Y.
func handJointXY() *Distribution {
	d, err := NewDistribution([]int{2, 2}, []float64{0.1, 0.2, 0.3, 0.4})
	if err != nil {
		panic(err)
	}
	return d
}

func approxTable(t *testing.T, got *Distribution, wantVars []int, want []float64) {
	t.Helper()
	if !sortedEqual(got.Vars(), wantVars) {
		t.Fatalf("vars = %v, want %v", got.Vars(), wantVars)
	}
	if len(got.prob) != len(want) {
		t.Fatalf("len prob = %d, want %d", len(got.prob), len(want))
	}
	for i := range want {
		if math.Abs(got.prob[i]-want[i]) > 1e-9 {
			t.Errorf("prob[%d] = %g, want %g (%v)", i, got.prob[i], want[i], got.prob)
			return
		}
	}
}

func TestNewDistributionErrors(t *testing.T) {
	tests := []struct {
		name string
		card []int
		prob []float64
	}{
		{"wrong length", []int{2, 2}, []float64{1, 0, 0}},
		{"negative", []int{2}, []float64{-0.1, 1.1}},
		{"zero card", []int{0}, []float64{}},
		{"nonfinite", []int{2}, []float64{math.Inf(1), 0}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NewDistribution(tc.card, tc.prob); !errors.Is(err, ErrBadDistribution) {
				t.Errorf("got %v, want ErrBadDistribution", err)
			}
		})
	}
}

func TestProbAt(t *testing.T) {
	d := handJointXY()
	p, err := d.ProbAt(map[int]int{0: 1, 1: 0})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(p-0.3) > 1e-12 {
		t.Errorf("P(X=1,Y=0) = %g, want 0.3", p)
	}
	if _, err := d.ProbAt(map[int]int{0: 1}); !errors.Is(err, ErrAssignment) {
		t.Errorf("partial assignment should error")
	}
}

// TestEvalMarginal: Σ_X P(X,Y) = P(Y) = [0.4, 0.6].
func TestEvalMarginal(t *testing.T) {
	d := handJointXY()
	e := marginalExpr([]int{0}, jointExpr([]int{0, 1}))
	got, err := e.Evaluate(d)
	if err != nil {
		t.Fatal(err)
	}
	approxTable(t, got, []int{1}, []float64{0.4, 0.6})
}

// TestEvalConditional: P(Y|X) over {X,Y} = [1/3, 2/3, 3/7, 4/7].
func TestEvalConditional(t *testing.T) {
	d := handJointXY()
	e := condition(jointExpr([]int{0, 1}), []int{1}, []int{0}, []int{0, 1})
	got, err := e.Evaluate(d)
	if err != nil {
		t.Fatal(err)
	}
	approxTable(t, got, []int{0, 1}, []float64{1.0 / 3, 2.0 / 3, 3.0 / 7, 4.0 / 7})
}

// TestEvalProductChain: Σ_X P(X)·P(Y|X) must reproduce P(Y) — the chain rule,
// exercising product, conditional and marginal together.
func TestEvalProductChain(t *testing.T) {
	d := handJointXY()
	univ := []int{0, 1}
	px := condition(jointExpr([]int{0, 1}), []int{0}, nil, univ)            // P(X)
	pyGivenX := condition(jointExpr([]int{0, 1}), []int{1}, []int{0}, univ) // P(Y|X)
	e := marginalExpr([]int{0}, productExpr([]*Expr{px, pyGivenX}))
	got, err := e.Evaluate(d)
	if err != nil {
		t.Fatal(err)
	}
	approxTable(t, got, []int{1}, []float64{0.4, 0.6})
}
