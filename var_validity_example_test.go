package causa_test

import (
	"fmt"
	"math/rand"

	"github.com/jousudo/causa"
)

func validityExampleData() ([][]float64, []string) {
	rng := rand.New(rand.NewSource(71))
	const observations, burn = 2400, 300
	data := make([][]float64, 3)
	for variable := range data {
		data[variable] = make([]float64, observations+burn)
	}
	for t := 1; t < observations+burn; t++ {
		data[0][t] = 0.50*data[0][t-1] + rng.NormFloat64()
		data[1][t] = 0.60*data[1][t-1] + 0.70*data[0][t-1] + rng.NormFloat64()
		data[2][t] = 0.40*data[2][t-1] + 0.60*data[1][t-1] + rng.NormFloat64()
	}
	for variable := range data {
		data[variable] = data[variable][burn:]
	}
	return data, []string{"load", "queue", "latency"}
}

func ExampleVARModel_Stability() {
	data, names := validityExampleData()
	model, _ := causa.FitVAR(data, names, 1)
	stability, _ := model.Stability(nil)
	fmt.Println(stability.Status)
	fmt.Printf("positive stability margin: %t\n", stability.Margin > causa.DefaultVARStabilityTolerance)

	// Output:
	// stable
	// positive stability margin: true
}

func ExampleVARModel_WhitenessTest() {
	rng := rand.New(rand.NewSource(83))
	series := make([]float64, 3200)
	for t := 2; t < len(series); t++ {
		series[t] = 0.75*series[t-1] - 0.45*series[t-2] + rng.NormFloat64()
	}
	underfit, _ := causa.FitVAR([][]float64{series}, []string{"x"}, 1)
	adequate, _ := causa.FitVAR([][]float64{series}, []string{"x"}, 2)
	underfitTest, _ := underfit.WhitenessTest(12, true)
	adequateTest, _ := adequate.WhitenessTest(12, true)
	fmt.Println("underfit rejected:", underfitTest.PValue < 0.05)
	fmt.Println("adequate order rejected:", adequateTest.PValue < 0.05)

	// Output:
	// underfit rejected: true
	// adequate order rejected: false
}

func ExampleVARGrangerScan() {
	data, names := validityExampleData()
	scan, _ := causa.VARGrangerScan(data, names, 1, nil) // Holm is the zero-value default.
	for _, finding := range scan.Findings {
		if finding.Significant {
			fmt.Printf("%s -> %s\n", finding.Test.CauseName, finding.Test.EffectName)
		}
	}

	// Output:
	// load -> queue
	// queue -> latency
}
