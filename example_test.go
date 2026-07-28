package causa_test

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/jousudo/causa"
)

// ExampleGrangerTest demonstrates a one-directional Granger causality test on
// synthetic data where a "cause" series genuinely drives an "effect" series one
// step ahead. The random draw is seeded so the output is deterministic.
func ExampleGrangerTest() {
	rng := rand.New(rand.NewSource(42))
	const n = 400
	cause := make([]float64, n)
	effect := make([]float64, n)
	for t := 1; t < n; t++ {
		// cause is its own AR(1) process; effect depends on cause's last value.
		cause[t] = 0.4*cause[t-1] + rng.NormFloat64()
		effect[t] = 0.5*effect[t-1] + 0.7*cause[t-1] + 0.5*rng.NormFloat64()
	}

	// Does the past of cause help predict effect? (cause -> effect)
	fwd, err := causa.GrangerTest(cause, effect, 2)
	if err != nil {
		panic(err)
	}
	// And the reverse: does effect help predict cause? It should not.
	rev, err := causa.GrangerTest(effect, cause, 2)
	if err != nil {
		panic(err)
	}

	fmt.Printf("cause -> effect: significant=%v\n", fwd.PValue < 0.01)
	fmt.Printf("effect -> cause: significant=%v\n", rev.PValue < 0.01)

	// Output:
	// cause -> effect: significant=true
	// effect -> cause: significant=false
}

// ExamplePCStable recovers the structure of a collider A -> C <- B from
// observational data. A and B are independent causes of C; because they are
// marginally independent, the PC algorithm orients the v-structure, printing
// both compelled arrowheads into C. The random draw is seeded for a deterministic
// result.
func ExamplePCStable() {
	rng := rand.New(rand.NewSource(3))
	const n = 5000
	a := make([]float64, n)
	b := make([]float64, n)
	c := make([]float64, n)
	for t := 0; t < n; t++ {
		a[t] = rng.NormFloat64()
		b[t] = rng.NormFloat64()
		c[t] = 0.8*a[t] + 0.8*b[t] + rng.NormFloat64()
	}

	g, err := causa.PCStable([][]float64{a, b, c}, []string{"A", "B", "C"}, nil)
	if err != nil {
		panic(err)
	}

	for _, e := range g.DirectedEdges() {
		nodes := g.Nodes()
		fmt.Printf("%s -> %s\n", nodes[e.From], nodes[e.To])
	}
	fmt.Printf("undirected edges: %d\n", len(g.UndirectedEdges()))

	// Output:
	// A -> C
	// B -> C
	// undirected edges: 0
}

// ExampleDirectLiNGAM recovers the direction of a two-variable cause→effect model
// from observational data. Because the noise is non-Gaussian (uniform), LiNGAM
// identifies the direction that a constraint-based method could not: it prints the
// causal order (cause first) and the estimated connection strength. The random
// draw is seeded for a deterministic result.
func ExampleDirectLiNGAM() {
	rng := rand.New(rand.NewSource(1))
	const n = 3000
	cause := make([]float64, n)
	effect := make([]float64, n)
	for t := 0; t < n; t++ {
		cause[t] = rng.Float64()*2 - 1                   // uniform, non-Gaussian
		effect[t] = 1.5*cause[t] + (rng.Float64()*2 - 1) // effect = 1.5·cause + noise
	}

	res, err := causa.DirectLiNGAM([][]float64{cause, effect}, []string{"cause", "effect"}, nil)
	if err != nil {
		panic(err)
	}

	fmt.Printf("causal order: %v\n", res.OrderedNodes())
	fmt.Printf("cause -> effect: %.1f\n", res.Coefficient(0, 1))

	// Output:
	// causal order: [cause effect]
	// cause -> effect: 1.5
}

// ExamplePartialCorrelation shows how conditioning removes a spurious
// correlation. A latent common cause Z drives both X and Y, so X and Y are
// correlated on their own; but given Z the partial correlation collapses to ~0 —
// the numeric heart of constraint-based discovery. The draw is seeded so the
// output is deterministic.
func ExamplePartialCorrelation() {
	rng := rand.New(rand.NewSource(7))
	const n = 2000
	x := make([]float64, n)
	y := make([]float64, n)
	z := make([]float64, n)
	for t := 0; t < n; t++ {
		z[t] = rng.NormFloat64()
		x[t] = z[t] + rng.NormFloat64() // X = Z + noise
		y[t] = z[t] + rng.NormFloat64() // Y = Z + noise
	}
	data := [][]float64{x, y, z} // indices: X=0, Y=1, Z=2

	// Marginal correlation of X and Y (empty conditioning set).
	rMarg, err := causa.PartialCorrelation(data, 0, 1, nil)
	if err != nil {
		panic(err)
	}
	// Partial correlation of X and Y given Z.
	rCond, err := causa.PartialCorrelation(data, 0, 1, []int{2})
	if err != nil {
		panic(err)
	}

	fmt.Printf("X,Y correlated marginally: %v\n", math.Abs(rMarg) > 0.3)
	fmt.Printf("X,Y uncorrelated given Z:  %v\n", math.Abs(rCond) < 0.1)

	// Output:
	// X,Y correlated marginally: true
	// X,Y uncorrelated given Z:  true
}

// ExampleFisherZTest shows the default conditional-independence test PCStable
// uses. On the same latent-common-cause data (Z drives X and Y), the test reports
// X and Y as dependent on their own, but conditionally INDEPENDENT given Z — the
// edge-deletion decision the PC algorithm is built on. The draw is seeded so the
// output is deterministic.
func ExampleFisherZTest() {
	rng := rand.New(rand.NewSource(7))
	const n = 2000
	x := make([]float64, n)
	y := make([]float64, n)
	z := make([]float64, n)
	for t := 0; t < n; t++ {
		z[t] = rng.NormFloat64()
		x[t] = z[t] + rng.NormFloat64()
		y[t] = z[t] + rng.NormFloat64()
	}
	data := [][]float64{x, y, z} // indices: X=0, Y=1, Z=2

	pMarg, err := causa.FisherZTest(data, 0, 1, nil)
	if err != nil {
		panic(err)
	}
	pCond, err := causa.FisherZTest(data, 0, 1, []int{2})
	if err != nil {
		panic(err)
	}

	// Marginally X and Y are strongly dependent; conditioning on the common cause Z
	// collapses that dependence by many orders of magnitude (the edge-deletion the
	// PC algorithm makes). A fixed p>0.05 threshold would be brittle at this sample
	// size — a correctly-sized α=0.05 test still rejects ~5% of the time under true
	// independence — so the robust, deterministic claim is the contrast itself.
	fmt.Printf("marginal dependence significant (p < 1e-6):    %v\n", pMarg < 1e-6)
	fmt.Printf("given Z the signal collapses (pCond >> pMarg): %v\n", pCond > 1e6*pMarg)

	// Output:
	// marginal dependence significant (p < 1e-6):    true
	// given Z the signal collapses (pCond >> pMarg): true
}

// ExampleSEM shows the do-operator and a counterfactual on a fully specified
// linear structural model X -> Y -> Z (Y = 2X, Z = 3Y). Interventions and
// counterfactuals reduce to forward substitution through the structural
// equations, so the answers are exact and deterministic.
func ExampleSEM() {
	coef := [][]float64{
		{0, 0, 0}, // X is a root
		{2, 0, 0}, // Y = 2·X
		{0, 3, 0}, // Z = 3·Y
	}
	s, err := causa.NewSEM([]string{"X", "Y", "Z"}, coef, nil)
	if err != nil {
		panic(err)
	}

	// Intervention: force X = 1 and read the interventional expectation.
	got, _ := s.Intervene(map[string]float64{"X": 1})
	fmt.Printf("do(X=1): Y=%.0f Z=%.0f\n", got["Y"], got["Z"])

	// Total effect of X on Z along the chain (2 × 3).
	te, _ := s.TotalEffect("X", "Z")
	fmt.Printf("total effect X->Z: %.0f\n", te)

	// Counterfactual: we OBSERVED (X=1, Y=3, Z=8) — Y and Z carry disturbances.
	// Had X been 5 instead, holding the same disturbances, what would follow?
	cf, _ := s.Counterfactual(
		map[string]float64{"X": 1, "Y": 3, "Z": 8},
		map[string]float64{"X": 5},
	)
	fmt.Printf("counterfactual do(X=5): Y=%.0f Z=%.0f\n", cf["Y"], cf["Z"])

	// Output:
	// do(X=1): Y=2 Z=6
	// total effect X->Z: 6
	// counterfactual do(X=5): Y=11 Z=32
}
