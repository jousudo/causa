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
