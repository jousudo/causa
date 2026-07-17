package causa_test

import (
	"math/rand"
	"testing"

	"github.com/jousudo/causa"
)

// benchData builds a deterministic driven pair of the requested length so the
// benchmark measures the test, not the data generation.
func benchData(n int) (cause, effect []float64) {
	rng := rand.New(rand.NewSource(99))
	cause = make([]float64, n)
	effect = make([]float64, n)
	for t := 1; t < n; t++ {
		cause[t] = 0.4*cause[t-1] + rng.NormFloat64()
		effect[t] = 0.5*effect[t-1] + 0.6*cause[t-1] + 0.5*rng.NormFloat64()
	}
	return cause, effect
}

func BenchmarkGrangerTest_n1000_lags4(b *testing.B) {
	cause, effect := benchData(1000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := causa.GrangerTest(cause, effect, 4); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGrangerTest_n5000_lags8(b *testing.B) {
	cause, effect := benchData(5000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := causa.GrangerTest(cause, effect, 8); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGrangerTest_n1000_lags1(b *testing.B) {
	cause, effect := benchData(1000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := causa.GrangerTest(cause, effect, 1); err != nil {
			b.Fatal(err)
		}
	}
}

// benchPCData builds a deterministic n-sample linear-Gaussian panel over p
// variables from a fixed random DAG (edges from lower to higher index), so the
// benchmark measures PCStable, not data generation.
func benchPCData(seed int64, n, p int) [][]float64 {
	rng := rand.New(rand.NewSource(seed))
	w := make([][]float64, p)
	for i := range w {
		w[i] = make([]float64, p)
	}
	for i := 0; i < p; i++ {
		for j := i + 1; j < p; j++ {
			if rng.Float64() < 0.3 {
				w[i][j] = 0.8
			}
		}
	}
	data := make([][]float64, p)
	for i := range data {
		data[i] = make([]float64, n)
	}
	for t := 0; t < n; t++ {
		for node := 0; node < p; node++ {
			v := rng.NormFloat64()
			for parent := 0; parent < node; parent++ {
				if w[parent][node] != 0 {
					v += w[parent][node] * data[parent][t]
				}
			}
			data[node][t] = v
		}
	}
	return data
}

func BenchmarkPCStable_p10_n1000(b *testing.B) {
	data := benchPCData(7, 1000, 10)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := causa.PCStable(data, nil, nil); err != nil {
			b.Fatal(err)
		}
	}
}
