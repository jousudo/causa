package causa_test

import (
	"math/rand"
	"testing"

	"github.com/jousudo/causa"
)

func benchVARData(seed int64, observations, variables int) [][]float64 {
	rng := rand.New(rand.NewSource(seed))
	data := make([][]float64, variables)
	for variable := range data {
		data[variable] = make([]float64, observations)
	}
	for t := 2; t < observations; t++ {
		for variable := 0; variable < variables; variable++ {
			value := 0.45*data[variable][t-1] - 0.15*data[variable][t-2]
			if variable > 0 {
				value += 0.25 * data[variable-1][t-1]
			}
			data[variable][t] = value + rng.NormFloat64()
		}
	}
	return data
}

func BenchmarkFitVAR_p6_n5000_lags4(b *testing.B) {
	data := benchVARData(51, 5000, 6)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := causa.FitVAR(data, nil, 4); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkVARGrangerTest_p6_n5000_lags4(b *testing.B) {
	data := benchVARData(51, 5000, 6)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := causa.VARGrangerTest(data, nil, 0, 1, 4); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSelectVARLags_p6_n5000_max8(b *testing.B) {
	data := benchVARData(51, 5000, 6)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := causa.SelectVARLags(data, nil, 8); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkStationaryBootstrapMean_n2000_b1000(b *testing.B) {
	data := benchVARData(51, 2000, 1)[0]
	stat := func(indices []int) (float64, error) {
		var sum float64
		for _, index := range indices {
			sum += data[index]
		}
		return sum / float64(len(indices)), nil
	}
	opts := causa.StationaryBootstrapOptions{
		BootstrapOptions: causa.BootstrapOptions{Resamples: 1000, Seed: 1},
		MeanBlockLength:  20,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := causa.StationaryBootstrap(len(data), stat, opts); err != nil {
			b.Fatal(err)
		}
	}
}
