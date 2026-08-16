package causa

import (
	"context"
	"errors"
	"math"
	"testing"
)

func TestVARGrangerScanMatchesIndividualTests(t *testing.T) {
	data := generateKnownVAR(117, 1800)
	names := []string{"X", "Y", "Z"}
	scan, err := VARGrangerScan(data, names, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	if scan.Tests != 6 || len(scan.Findings) != 6 || scan.Adjustment != PValueHolm || scan.Alpha != 0.05 {
		t.Fatalf("bad scan metadata: %+v", scan)
	}
	raw := make([]float64, len(scan.Findings))
	for index, finding := range scan.Findings {
		individual, err := VARGrangerTest(data, names, finding.Test.Cause, finding.Test.Effect, 2)
		if err != nil {
			t.Fatal(err)
		}
		if rel(finding.Test.F, individual.F) > 1e-11 || math.Abs(finding.Test.PValue-individual.PValue) > 1e-13 {
			t.Errorf("scan differs for %d->%d: scan=%+v individual=%+v", finding.Test.Cause, finding.Test.Effect, finding.Test, individual)
		}
		raw[index] = finding.Test.PValue
	}
	wantAdjusted, _ := AdjustPValues(raw, PValueHolm)
	for index, finding := range scan.Findings {
		if math.Abs(finding.AdjustedPValue-wantAdjusted[index]) > 1e-15 {
			t.Errorf("adjusted[%d]=%g, want %g", index, finding.AdjustedPValue, wantAdjusted[index])
		}
	}
}

func TestVARGrangerScanRecoversKnownDirections(t *testing.T) {
	data := generateKnownVAR(31, 3500)
	scan, err := VARGrangerScan(data, []string{"X", "Y", "Z"}, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	found := make(map[[2]int]VARGrangerFinding)
	for _, finding := range scan.Findings {
		found[[2]int{finding.Test.Cause, finding.Test.Effect}] = finding
	}
	for _, edge := range [][2]int{{0, 1}, {1, 2}} {
		if !found[edge].Significant || found[edge].AdjustedPValue >= 1e-8 {
			t.Errorf("known direction %v not recovered: %+v", edge, found[edge])
		}
	}
	if found[[2]int{2, 0}].Significant {
		t.Errorf("spurious reverse direction survived Holm: %+v", found[[2]int{2, 0}])
	}
}

func TestVARGrangerScanOptionsAndErrors(t *testing.T) {
	data := generateKnownVAR(23, 300)
	opts := &VARGrangerScanOptions{
		Adjustment: PValueBenjaminiHochberg,
		Alpha:      0.1,
		MaxTests:   6,
	}
	result, err := VARGrangerScan(data, nil, 2, opts)
	if err != nil {
		t.Fatal(err)
	}
	if result.Adjustment != PValueBenjaminiHochberg || result.Alpha != 0.1 {
		t.Fatalf("options not retained: %+v", result)
	}
	opts.MaxTests = 5
	if _, err := VARGrangerScan(data, nil, 2, opts); !errors.Is(err, ErrVARGrangerTestBudget) {
		t.Errorf("budget: got %v", err)
	}
	opts.MaxTests = 6
	opts.Alpha = 1
	if _, err := VARGrangerScan(data, nil, 2, opts); !errors.Is(err, ErrInvalidAlpha) {
		t.Errorf("alpha: got %v", err)
	}
	opts.Alpha = 0.05
	opts.Adjustment = PValueAdjustment(99)
	if _, err := VARGrangerScan(data, nil, 2, opts); !errors.Is(err, ErrInvalidPValueAdjustment) {
		t.Errorf("adjustment: got %v", err)
	}
	if _, err := VARGrangerScan(data[:1], nil, 2, nil); !errors.Is(err, ErrTooFewVariables) {
		t.Errorf("one variable: got %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := VARGrangerScanContext(ctx, data, nil, 2, nil); !errors.Is(err, context.Canceled) {
		t.Errorf("cancellation: got %v", err)
	}
}
