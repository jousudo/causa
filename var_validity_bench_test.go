package causa_test

import (
	"testing"

	"github.com/jousudo/causa"
)

func BenchmarkVARStability_p6_lags4(b *testing.B) {
	model, err := causa.FitVAR(benchVARData(61, 5000, 6), nil, 4)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := model.Stability(nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkVARWhiteness_p6_n5000_lags4_max12(b *testing.B) {
	model, err := causa.FitVAR(benchVARData(61, 5000, 6), nil, 4)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := model.WhitenessTest(12, true); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkVARGrangerScan_p6_n5000_lags4(b *testing.B) {
	data := benchVARData(61, 5000, 6)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := causa.VARGrangerScan(data, nil, 4, nil); err != nil {
			b.Fatal(err)
		}
	}
}
