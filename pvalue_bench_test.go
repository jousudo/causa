package causa_test

import (
	"testing"

	"github.com/jousudo/causa"
)

func BenchmarkAdjustPValuesHolm_m10000(b *testing.B) {
	pValues := make([]float64, 10_000)
	for index := range pValues {
		pValues[index] = float64((index*7919)%10_001) / 10_001
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := causa.AdjustPValues(pValues, causa.PValueHolm); err != nil {
			b.Fatal(err)
		}
	}
}
