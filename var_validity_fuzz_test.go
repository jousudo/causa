package causa_test

import (
	"math"
	"testing"

	"github.com/jousudo/causa"
)

func FuzzAdjustPValues(f *testing.F) {
	f.Add([]byte{1, 4, 3, 0, 80}, byte(causa.PValueHolm))
	f.Add([]byte{}, byte(causa.PValueBenjaminiHochberg))
	f.Fuzz(func(t *testing.T, encoded []byte, methodByte byte) {
		if len(encoded) > 128 {
			t.Skip()
		}
		pValues := make([]float64, len(encoded))
		for index, value := range encoded {
			pValues[index] = float64(value) / 255
		}
		method := causa.PValueAdjustment(methodByte % 3)
		adjusted, err := causa.AdjustPValues(pValues, method)
		if err != nil {
			t.Fatal(err)
		}
		if len(adjusted) != len(pValues) {
			t.Fatalf("length changed from %d to %d", len(pValues), len(adjusted))
		}
		for _, pValue := range adjusted {
			if math.IsNaN(pValue) || pValue < 0 || pValue > 1 {
				t.Fatalf("invalid adjusted p-value %g", pValue)
			}
		}
	})
}
