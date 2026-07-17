package causa_test

import (
	"fmt"
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
