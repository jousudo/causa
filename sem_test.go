package causa

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

// chainSEM builds the ground-truth model X -> Y -> Z with Y = 2X, Z = 3Y and the
// given intercepts (nil = zeros). Indices: X=0, Y=1, Z=2.
func chainSEM(t *testing.T, intercept []float64) *SEM {
	t.Helper()
	coef := [][]float64{
		{0, 0, 0}, // X: root
		{2, 0, 0}, // Y = 2·X
		{0, 3, 0}, // Z = 3·Y
	}
	s, err := NewSEM([]string{"X", "Y", "Z"}, coef, intercept)
	if err != nil {
		t.Fatalf("NewSEM: %v", err)
	}
	return s
}

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestNewSEMRecoversOrder(t *testing.T) {
	s := chainSEM(t, nil)
	if got := s.OrderedNodes(); !(got[0] == "X" && got[1] == "Y" && got[2] == "Z") {
		t.Fatalf("causal order = %v, want [X Y Z]", got)
	}
}

func TestInterveneChain(t *testing.T) {
	s := chainSEM(t, nil)

	// do(X=1): Y = 2, Z = 6.
	got, err := s.Intervene(map[string]float64{"X": 1})
	if err != nil {
		t.Fatalf("Intervene: %v", err)
	}
	if !near(got["X"], 1) || !near(got["Y"], 2) || !near(got["Z"], 6) {
		t.Fatalf("do(X=1) = %v, want X=1 Y=2 Z=6", got)
	}

	// do(Y=5): Z = 15, and X is NOT a descendant of Y so it stays at its mean (0).
	got, err = s.Intervene(map[string]float64{"Y": 5})
	if err != nil {
		t.Fatalf("Intervene: %v", err)
	}
	if !near(got["X"], 0) || !near(got["Y"], 5) || !near(got["Z"], 15) {
		t.Fatalf("do(Y=5) = %v, want X=0 Y=5 Z=15", got)
	}
}

func TestInterveneWithIntercepts(t *testing.T) {
	// X = 1 + e, Y = 10 + 2X + e, Z = 100 + 3Y + e.
	s := chainSEM(t, []float64{1, 10, 100})
	got, err := s.Intervene(map[string]float64{"X": 0})
	if err != nil {
		t.Fatalf("Intervene: %v", err)
	}
	// do(X=0): Y = 10 + 0 = 10; Z = 100 + 30 = 130.
	if !near(got["X"], 0) || !near(got["Y"], 10) || !near(got["Z"], 130) {
		t.Fatalf("do(X=0) = %v, want X=0 Y=10 Z=130", got)
	}
	// No intervention: every variable takes its structural mean.
	got, _ = s.Intervene(nil)
	// X=1, Y=10+2=12, Z=100+36=136.
	if !near(got["X"], 1) || !near(got["Y"], 12) || !near(got["Z"], 136) {
		t.Fatalf("no-do means = %v, want X=1 Y=12 Z=136", got)
	}
}

func TestTotalEffect(t *testing.T) {
	s := chainSEM(t, nil)
	cases := []struct {
		from, to string
		want     float64
	}{
		{"X", "Y", 2},
		{"Y", "Z", 3},
		{"X", "Z", 6}, // 2·3 along the chain
		{"Z", "X", 0}, // X is not a descendant of Z
		{"Y", "X", 0},
	}
	for _, c := range cases {
		got, err := s.TotalEffect(c.from, c.to)
		if err != nil {
			t.Fatalf("TotalEffect(%s,%s): %v", c.from, c.to, err)
		}
		if !near(got, c.want) {
			t.Errorf("TotalEffect(%s,%s) = %v, want %v", c.from, c.to, got, c.want)
		}
	}
}

func TestCounterfactualNoiseFree(t *testing.T) {
	s := chainSEM(t, nil)
	// Observed row is exactly on the model (zero noise).
	obs := map[string]float64{"X": 1, "Y": 2, "Z": 6}

	// Had we set Y=10: X is unchanged (not a descendant), Z = 30.
	cf, err := s.Counterfactual(obs, map[string]float64{"Y": 10})
	if err != nil {
		t.Fatalf("Counterfactual: %v", err)
	}
	if !near(cf["X"], 1) || !near(cf["Y"], 10) || !near(cf["Z"], 30) {
		t.Fatalf("cf do(Y=10) = %v, want X=1 Y=10 Z=30", cf)
	}
}

func TestCounterfactualWithNoise(t *testing.T) {
	s := chainSEM(t, nil)
	// Observed row carries disturbances: e_Y = 3 − 2·1 = 1, e_Z = 8 − 3·3 = −1.
	obs := map[string]float64{"X": 1, "Y": 3, "Z": 8}

	// Empty do must reproduce the observation (the abducted noise regenerates it).
	cf, err := s.Counterfactual(obs, nil)
	if err != nil {
		t.Fatalf("Counterfactual: %v", err)
	}
	if !near(cf["X"], 1) || !near(cf["Y"], 3) || !near(cf["Z"], 8) {
		t.Fatalf("cf empty-do = %v, want the observation back", cf)
	}

	// Had we set X=5, holding the SAME disturbances: Y = 2·5 + 1 = 11, Z = 3·11 − 1 = 32.
	cf, err = s.Counterfactual(obs, map[string]float64{"X": 5})
	if err != nil {
		t.Fatalf("Counterfactual: %v", err)
	}
	if !near(cf["X"], 5) || !near(cf["Y"], 11) || !near(cf["Z"], 32) {
		t.Fatalf("cf do(X=5) = %v, want X=5 Y=11 Z=32", cf)
	}
}

func TestFitSEMRecoversChain(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	const n = 4000
	x := make([]float64, n)
	y := make([]float64, n)
	z := make([]float64, n)
	for t := 0; t < n; t++ {
		x[t] = rng.NormFloat64()
		y[t] = 2*x[t] + rng.NormFloat64()
		z[t] = 3*y[t] + rng.NormFloat64()
	}
	s, err := FitSEM([][]float64{x, y, z}, []string{"X", "Y", "Z"}, []int{0, 1, 2})
	if err != nil {
		t.Fatalf("FitSEM: %v", err)
	}
	w := s.Weights()
	if math.Abs(w[1][0]-2) > 0.1 {
		t.Errorf("fitted X->Y = %.3f, want ~2", w[1][0])
	}
	if math.Abs(w[2][1]-3) > 0.1 {
		t.Errorf("fitted Y->Z = %.3f, want ~3", w[2][1])
	}
	// Z has no DIRECT edge from X (it depends on X only through Y): the fitted
	// direct coefficient should be ~0.
	if math.Abs(w[2][0]) > 0.1 {
		t.Errorf("fitted DIRECT X->Z = %.3f, want ~0", w[2][0])
	}
	// The total effect of X on Z is ~6 despite no direct edge.
	te, _ := s.TotalEffect("X", "Z")
	if math.Abs(te-6) > 0.3 {
		t.Errorf("total effect X->Z = %.3f, want ~6", te)
	}
}

func TestFitSEMComposesWithLiNGAM(t *testing.T) {
	// DirectLiNGAM discovers the order; FitSEM estimates the structural model on it.
	rng := rand.New(rand.NewSource(5))
	const n = 4000
	a := make([]float64, n)
	b := make([]float64, n)
	for t := 0; t < n; t++ {
		a[t] = rng.Float64()*2 - 1              // uniform (non-Gaussian) root
		b[t] = 1.5*a[t] + (rng.Float64()*2 - 1) // B = 1.5·A + noise
	}
	res, err := DirectLiNGAM([][]float64{a, b}, []string{"A", "B"}, nil)
	if err != nil {
		t.Fatalf("DirectLiNGAM: %v", err)
	}
	s, err := FitSEM([][]float64{a, b}, res.Nodes(), res.CausalOrder())
	if err != nil {
		t.Fatalf("FitSEM: %v", err)
	}
	te, _ := s.TotalEffect("A", "B")
	if math.Abs(te-1.5) > 0.15 {
		t.Errorf("total effect A->B = %.3f, want ~1.5", te)
	}
}

func TestNewSEMRejectsCycle(t *testing.T) {
	coef := [][]float64{
		{0, 1}, // x0 = x1
		{1, 0}, // x1 = x0  → cycle
	}
	if _, err := NewSEM(nil, coef, nil); !errors.Is(err, ErrCyclic) {
		t.Fatalf("cyclic coef: err = %v, want ErrCyclic", err)
	}
}

func TestNewSEMRejectsNonSquare(t *testing.T) {
	coef := [][]float64{{0, 0, 0}, {1, 0, 0}}
	if _, err := NewSEM(nil, coef, nil); !errors.Is(err, ErrNotSquare) {
		t.Fatalf("non-square coef: err = %v, want ErrNotSquare", err)
	}
}

func TestSEMRejectsUnknownAndIncomplete(t *testing.T) {
	s := chainSEM(t, nil)
	if _, err := s.Intervene(map[string]float64{"Q": 1}); !errors.Is(err, ErrUnknownVariable) {
		t.Errorf("unknown do var: err = %v, want ErrUnknownVariable", err)
	}
	if _, err := s.Counterfactual(map[string]float64{"X": 1, "Y": 2}, nil); !errors.Is(err, ErrIncompleteObservation) {
		t.Errorf("incomplete observation: err = %v, want ErrIncompleteObservation", err)
	}
}
