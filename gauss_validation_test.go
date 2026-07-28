package causa

import (
	"math"
	"math/rand"
	"strconv"
	"testing"
)

// This file validates Expr.EvaluateGaussian against closed-form truth on random
// LINEAR-GAUSSIAN latent SCMs — the continuous analogue of the discrete
// brute-force SCM harness in id_test.go. For each diagram we (1) draw a random
// linear-Gaussian structural model consistent with it (one latent root per
// bidirected edge), (2) compute the exact observational covariance of the observed
// variables in closed form, Σ = A·Ω·Aᵀ with A = (I−B)⁻¹, and (3) require the
// interventional slope dE[Y | do(X)]/dX read off the evaluated estimand to match
// the structural total effect the SEM path computes directly. If the estimand is
// right it matches for every random parameterization.

// gaussJordanInverse inverts a general square matrix by Gauss–Jordan elimination
// with partial pivoting (test-only; the library's own solves are all SPD). It
// panics on a singular matrix — the test models are acyclic, so I−B is always
// invertible.
func gaussJordanInverse(a [][]float64) [][]float64 {
	n := len(a)
	m := make([][]float64, n)
	for i := range m {
		m[i] = make([]float64, 2*n)
		copy(m[i], a[i])
		m[i][n+i] = 1
	}
	for col := 0; col < n; col++ {
		piv := col
		for r := col + 1; r < n; r++ {
			if math.Abs(m[r][col]) > math.Abs(m[piv][col]) {
				piv = r
			}
		}
		if math.Abs(m[piv][col]) < 1e-15 {
			panic("singular matrix in test inverse")
		}
		m[col], m[piv] = m[piv], m[col]
		d := m[col][col]
		for c := 0; c < 2*n; c++ {
			m[col][c] /= d
		}
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := m[r][col]
			for c := 0; c < 2*n; c++ {
				m[r][c] -= f * m[col][c]
			}
		}
	}
	inv := make([][]float64, n)
	for i := range inv {
		inv[i] = append([]float64(nil), m[i][n:]...)
	}
	return inv
}

// linGaussSCM draws a random linear-Gaussian model for the diagram g: observed
// directed edges get random coefficients, each bidirected edge gets a latent root
// pointing at both endpoints, and every node gets a random positive noise variance.
// It returns the full coefficient matrix B (b[i][j] = effect of j on i) and noise
// variances over total = observed + latent nodes.
func linGaussSCM(g *Diagram, seed int64) (b [][]float64, omega []float64, total, n int) {
	rng := rand.New(rand.NewSource(seed))
	n = g.Order()
	var latEdges [][2]int
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if g.HasBidirected(i, j) {
				latEdges = append(latEdges, [2]int{i, j})
			}
		}
	}
	total = n + len(latEdges)
	b = make([][]float64, total)
	for i := range b {
		b[i] = make([]float64, total)
	}
	omega = make([]float64, total)
	coef := func() float64 {
		c := 0.5 + rng.Float64() // magnitude in [0.5, 1.5], away from zero
		if rng.Intn(2) == 0 {
			c = -c
		}
		return c
	}
	for v := 0; v < n; v++ {
		for _, p := range g.Parents(v) {
			b[v][p] = coef()
		}
	}
	for e, pr := range latEdges {
		l := n + e
		b[pr[0]][l] = coef()
		b[pr[1]][l] = coef()
	}
	for v := 0; v < total; v++ {
		omega[v] = 0.5 + rng.Float64() // noise variance in [0.5, 1.5]
	}
	return b, omega, total, n
}

// observedCov returns the exact covariance of the observed variables 0..n-1 under
// the linear-Gaussian model: Σ = A·Ω·Aᵀ with A = (I−B)⁻¹ and Ω = diag(noise).
func observedCov(b [][]float64, omega []float64, total, n int) [][]float64 {
	im := make([][]float64, total)
	for i := range im {
		im[i] = make([]float64, total)
		im[i][i] = 1
		for j := range im[i] {
			im[i][j] -= b[i][j]
		}
	}
	a := gaussJordanInverse(im) // (I−B)⁻¹
	// Σ_full = A·Ω·Aᵀ
	full := make([][]float64, total)
	for i := range full {
		full[i] = make([]float64, total)
		for j := range full[i] {
			var s float64
			for k := 0; k < total; k++ {
				s += a[i][k] * omega[k] * a[j][k]
			}
			full[i][j] = s
		}
	}
	obs := make([][]float64, n)
	for i := 0; i < n; i++ {
		obs[i] = append([]float64(nil), full[i][:n]...)
	}
	return obs
}

// trueSlope is the structural total effect dE[Y | do(X)]/dX, computed directly from
// the full SEM (the do-operator path), independent of the identification machinery.
func trueSlope(b [][]float64, total, x, y int) float64 {
	names := make([]string, total)
	for i := range names {
		names[i] = strconv.Itoa(i)
	}
	sem, err := NewSEM(names, b, nil)
	if err != nil {
		panic(err)
	}
	eff, err := sem.TotalEffect(strconv.Itoa(x), strconv.Itoa(y))
	if err != nil {
		panic(err)
	}
	return eff
}

func TestEvaluateGaussianAgainstSCM(t *testing.T) {
	cases := []struct {
		name       string
		names      []string
		directed   [][2]int
		bidirected [][2]int
		x, y       int
	}{
		// Back-door: observed confounder Z. X=0, Y=1, Z=2.
		{"backdoor", []string{"X", "Y", "Z"}, [][2]int{{0, 1}, {2, 0}, {2, 1}}, nil, 0, 1},
		// Direct effect, no confounding. X=0, Y=1.
		{"direct", []string{"X", "Y"}, [][2]int{{0, 1}}, nil, 0, 1},
		// Front door: X→M→Y with a latent common cause X↔Y. X=0, M=1, Y=2.
		{"frontdoor", []string{"X", "M", "Y"}, [][2]int{{0, 1}, {1, 2}}, [][2]int{{0, 2}}, 0, 2},
		// Generalized front door: X→M1→M2→Y, latent X↔Y. X=0, M1=1, M2=2, Y=3.
		{"frontdoor_chain", []string{"X", "M1", "M2", "Y"}, [][2]int{{0, 1}, {1, 2}, {2, 3}}, [][2]int{{0, 3}}, 0, 3},
		// Mediated back-door: Z→X→M→Y and Z→Y, all observed. X=1, Y=3.
		{"mediated_backdoor", []string{"Z", "X", "M", "Y"}, [][2]int{{0, 1}, {1, 2}, {2, 3}, {0, 3}}, nil, 1, 3},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g, err := NewDiagram(tc.names, tc.directed, tc.bidirected)
			if err != nil {
				t.Fatal(err)
			}
			res, err := Identify(g, []int{tc.y}, []int{tc.x})
			if err != nil {
				t.Fatal(err)
			}
			if !res.Identifiable {
				t.Fatalf("%s: expected identifiable, got hedge: %s", tc.name, res)
			}
			t.Logf("estimand: %s", res)
			for seed := int64(1); seed <= 6; seed++ {
				b, omega, total, n := linGaussSCM(g, seed)
				joint, err := NewGaussian(make([]float64, n), observedCov(b, omega, total, n))
				if err != nil {
					t.Fatalf("seed %d: NewGaussian: %v", seed, err)
				}
				f, err := res.Estimand.EvaluateGaussian(joint)
				if err != nil {
					t.Fatalf("seed %d: EvaluateGaussian: %v", seed, err)
				}
				// Slope of E[Y | do(X)] in X: linear, so a unit difference isolates it.
				m1, err := condMean(f, tc.x, 1, tc.y)
				if err != nil {
					t.Fatalf("seed %d: %v", seed, err)
				}
				m0, err := condMean(f, tc.x, 0, tc.y)
				if err != nil {
					t.Fatalf("seed %d: %v", seed, err)
				}
				got := m1 - m0
				want := trueSlope(b, total, tc.x, tc.y)
				if math.Abs(got-want) > 1e-7 {
					t.Errorf("seed %d: dE[Y|do(X)]/dX = %.9f, want %.9f (Δ=%.2e)\nestimand: %s",
						seed, got, want, math.Abs(got-want), res)
				}
				// Linearity check: E[Y|do(X=2)] must be the same slope extrapolated.
				m2, err := condMean(f, tc.x, 2, tc.y)
				if err != nil {
					t.Fatalf("seed %d: %v", seed, err)
				}
				if math.Abs((m2-m0)-2*got) > 1e-7 {
					t.Errorf("seed %d: non-linear interventional mean: E[Y|do(X=2)]-E[Y|do(X=0)]=%.9f, want %.9f",
						seed, m2-m0, 2*got)
				}
			}
		})
	}
}

// condMean fixes X=xval in the estimand factor and reads the interventional mean of
// the outcome variable y.
func condMean(f *GaussianFactor, x int, xval float64, y int) (float64, error) {
	d, err := f.Condition(map[int]float64{x: xval})
	if err != nil {
		return 0, err
	}
	return d.MeanAt(y)
}
