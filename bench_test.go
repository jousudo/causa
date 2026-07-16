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
