package causa

import (
	"errors"
	"math"
	"slices"
	"testing"
)

func TestAdjustPValuesMatchesR(t *testing.T) {
	pValues := []float64{0.01, 0.04, 0.03, 0.002, 0.8}
	tests := []struct {
		method PValueAdjustment
		want   []float64
	}{
		{PValueHolm, []float64{0.04, 0.09, 0.09, 0.01, 0.8}},
		{PValueBenjaminiHochberg, []float64{0.025, 0.05, 0.05, 0.01, 0.8}},
		{PValueNone, pValues},
	}
	for _, test := range tests {
		got, err := AdjustPValues(pValues, test.method)
		if err != nil {
			t.Fatal(err)
		}
		for i := range got {
			if math.Abs(got[i]-test.want[i]) > 1e-15 {
				t.Errorf("%s[%d]=%.17g, want %.17g", test.method, i, got[i], test.want[i])
			}
		}
		got[0] = 1
		if pValues[0] != 0.01 {
			t.Fatal("adjustment modified caller input")
		}
	}
	if PValueAdjustment(99).String() != "unknown" || PValueHolm.String() != "holm" {
		t.Fatal("unexpected adjustment String output")
	}
}

func TestAdjustPValuesErrorsAndEdges(t *testing.T) {
	if got, err := AdjustPValues(nil, PValueHolm); err != nil || got == nil {
		t.Fatalf("empty family: got=%v err=%v", got, err)
	}
	if got, err := AdjustPValues([]float64{0.2}, PValueHolm); err != nil || !slices.Equal(got, []float64{0.2}) {
		t.Fatalf("singleton: got=%v err=%v", got, err)
	}
	for _, pValues := range [][]float64{{-0.1}, {1.1}, {math.NaN()}, {math.Inf(1)}} {
		if _, err := AdjustPValues(pValues, PValueHolm); !errors.Is(err, ErrInvalidPValue) {
			t.Errorf("invalid %v: got %v", pValues, err)
		}
	}
	if _, err := AdjustPValues([]float64{0.1}, PValueAdjustment(99)); !errors.Is(err, ErrInvalidPValueAdjustment) {
		t.Errorf("unknown method: got %v", err)
	}
}
