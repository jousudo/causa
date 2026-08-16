package causa

import (
	"errors"
	"math"
	"testing"
)

func TestVARAdditionalValidationAndDiagnostics(t *testing.T) {
	valid := generateKnownVAR(83, 120)
	model, err := FitVARWithOptions(valid, nil, 2, &VAROptions{MaxDesignCells: -1})
	if err != nil {
		t.Fatal(err)
	}
	if rss := model.ResidualSumSquares(); len(rss) != 3 || rss[0] <= 0 {
		t.Fatalf("bad residual sums of squares: %v", rss)
	}
	if _, err := VARGrangerTest(valid[:1], nil, 0, 0, 1); !errors.Is(err, ErrTooFewVariables) {
		t.Errorf("one-variable Granger: got %v", err)
	}
	if _, err := VARGrangerTest(valid, nil, 0, 3, 1); !errors.Is(err, ErrVARIndex) {
		t.Errorf("out-of-range effect: got %v", err)
	}
	if _, err := VARGrangerTest(valid, nil, 0, 1, 0); !errors.Is(err, ErrInvalidLags) {
		t.Errorf("zero Granger lag: got %v", err)
	}
	if _, err := FitVAR([][]float64{{}}, nil, 1); !errors.Is(err, ErrVARTooShort) {
		t.Errorf("empty series: got %v", err)
	}
	if _, err := model.ResidualAutocorrelation(model.Observations()); !errors.Is(err, ErrInvalidResidualLag) {
		t.Errorf("long residual lag: got %v", err)
	}
}

func TestVARNumericEdgeCases(t *testing.T) {
	f, p := nestedFStatistic(1, 0, 1, 1)
	if !math.IsInf(f, 1) || p != 0 {
		t.Fatalf("perfect unrestricted improvement: F=%g p=%g", f, p)
	}
	f, p = nestedFStatistic(0, 0, 1, 1)
	if f != 0 || p != 1 {
		t.Fatalf("equal perfect fits: F=%g p=%g", f, p)
	}
	f, p = nestedFStatistic(1, 2, 1, 5)
	if f != 0 || p != 1 {
		t.Fatalf("roundoff clamp: F=%g p=%g", f, p)
	}
	if _, ok := logDetSPD([][]float64{{0}}); ok {
		t.Fatal("singular matrix accepted as SPD")
	}
	if _, ok := sampleCorrelation([]float64{1}, []float64{1}); ok {
		t.Fatal("one-row correlation accepted")
	}
	if _, ok := sampleCorrelation([]float64{1, 1}, []float64{2, 3}); ok {
		t.Fatal("constant correlation accepted")
	}
}
