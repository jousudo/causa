package causa_test

import (
	"fmt"
	"math/rand"

	"github.com/jousudo/causa"
)

// ExampleVARGrangerTest shows why the multivariate test matters. An observed
// process Z leads both X and Y, so the pairwise test mistakes X's leading
// information for a direct relation. Conditioning on all three series removes
// that omitted-variable signal.
func ExampleVARGrangerTest() {
	rng := rand.New(rand.NewSource(19))
	const n, burn = 2500, 300
	x := make([]float64, n+burn)
	y := make([]float64, n+burn)
	z := make([]float64, n+burn)
	for t := 1; t < len(x); t++ {
		z[t] = 0.82*z[t-1] + rng.NormFloat64()
		x[t] = 0.90*z[t-1] + 0.35*rng.NormFloat64()
		y[t] = 0.45*y[t-1] + 0.90*z[t-1] + 0.35*rng.NormFloat64()
	}
	x, y, z = x[burn:], y[burn:], z[burn:]

	pairwise, _ := causa.GrangerTest(x, y, 2)
	conditional, _ := causa.VARGrangerTest(
		[][]float64{x, y, z}, []string{"X", "Y", "Z"}, 0, 1, 2,
	)
	fmt.Println("pairwise X -> Y significant:", pairwise.PValue < 0.05)
	fmt.Println("given Z, X -> Y significant:", conditional.PValue < 0.05)
	fmt.Println("conditioned variable indices:", conditional.ConditionedOn)

	// Output:
	// pairwise X -> Y significant: true
	// given Z, X -> Y significant: false
	// conditioned variable indices: [2]
}

// ExampleStationaryBootstrap compares independent-row resampling with a
// dependence-preserving bootstrap on a serially correlated sample.
func ExampleStationaryBootstrap() {
	rng := rand.New(rand.NewSource(27))
	const n, burn = 800, 300
	series := make([]float64, n+burn)
	for t := 1; t < len(series); t++ {
		series[t] = 0.85*series[t-1] + rng.NormFloat64()
	}
	series = series[burn:]
	mean := func(indices []int) (float64, error) {
		var sum float64
		for _, index := range indices {
			sum += series[index]
		}
		return sum / float64(len(indices)), nil
	}
	base := causa.BootstrapOptions{Resamples: 600, Seed: 9}
	iid, _ := causa.Bootstrap(n, mean, base)
	dependent, _ := causa.StationaryBootstrap(n, mean, causa.StationaryBootstrapOptions{
		BootstrapOptions: base,
		MeanBlockLength:  24,
	})
	fmt.Println("dependent standard error is larger:", dependent.StdErr > iid.StdErr)

	// Output:
	// dependent standard error is larger: true
}
