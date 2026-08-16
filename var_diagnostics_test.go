package causa

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"
)

func scalarVARModel(coefficients ...float64) *VARModel {
	coef := make([][][]float64, len(coefficients))
	for lag, coefficient := range coefficients {
		coef[lag] = [][][]float64{{{coefficient}}}[0]
	}
	return &VARModel{names: []string{"X"}, lags: len(coefficients), coef: coef}
}

func TestVARStabilityKnownRoots(t *testing.T) {
	// Characteristic polynomial roots: 0.2 and 0.5±0.4i.
	model := scalarVARModel(1.2, -0.61, 0.082)
	result, err := model.Stability(nil)
	if err != nil {
		t.Fatal(err)
	}
	wantRadius := math.Sqrt(0.41)
	if result.Status != VARStable || math.Abs(result.SpectralRadius-wantRadius) > 1e-10 {
		t.Fatalf("stability=%+v eigenvalues=%v, want radius=%g", result, result.Eigenvalues(), wantRadius)
	}
	eigenvalues := result.Eigenvalues()
	if len(eigenvalues) != 3 || cmplx.Abs(eigenvalues[0]) < cmplx.Abs(eigenvalues[2]) {
		t.Fatalf("unsorted eigenvalues: %v", eigenvalues)
	}
	eigenvalues[0] = 99
	if result.Eigenvalues()[0] == 99 {
		t.Fatal("Eigenvalues exposed internal storage")
	}

	unstable := scalarVARModel(1.05)
	unstableResult, err := unstable.Stability(nil)
	if err != nil || unstableResult.Status != VARUnstable {
		t.Fatalf("unstable result=%+v err=%v", unstableResult, err)
	}
	boundary := scalarVARModel(1 + 5e-9)
	boundaryResult, err := boundary.Stability(nil)
	if err != nil || boundaryResult.Status != VARStabilityIndeterminate {
		t.Fatalf("boundary result=%+v err=%v", boundaryResult, err)
	}
	if VARStable.String() != "stable" || VARUnstable.String() != "unstable" ||
		VARStabilityIndeterminate.String() != "indeterminate" || VARStabilityStatus(99).String() != "unknown" {
		t.Fatal("unexpected stability String output")
	}
}

func TestVARStabilityBudgetsAndErrors(t *testing.T) {
	model := scalarVARModel(1.2, -0.61, 0.082)
	if _, err := model.Stability(&VARStabilityOptions{BoundaryTolerance: math.NaN()}); !errors.Is(err, ErrInvalidVARStabilityOptions) {
		t.Errorf("NaN tolerance: got %v", err)
	}
	if _, err := model.Stability(&VARStabilityOptions{BoundaryTolerance: -1}); !errors.Is(err, ErrInvalidVARStabilityOptions) {
		t.Errorf("negative tolerance: got %v", err)
	}
	if _, err := model.Stability(&VARStabilityOptions{MaxIterations: -1}); !errors.Is(err, ErrInvalidVARStabilityOptions) {
		t.Errorf("negative iterations: got %v", err)
	}
	if _, err := model.Stability(&VARStabilityOptions{MaxCompanionCells: 1}); !errors.Is(err, ErrVARCompanionTooLarge) {
		t.Errorf("cell budget: got %v", err)
	}
	if _, err := model.Stability(&VARStabilityOptions{MaxIterations: 1}); !errors.Is(err, ErrVAREigenNoConvergence) {
		t.Errorf("iteration budget: got %v", err)
	}
}

func TestVARWhitenessDistinguishesCorrectAndUnderfitOrder(t *testing.T) {
	data := generateKnownVAR(101, 5000)
	correct, err := FitVAR(data, nil, 2)
	if err != nil {
		t.Fatal(err)
	}
	white, err := correct.WhitenessTest(12, true)
	if err != nil {
		t.Fatal(err)
	}
	if white.PValue <= 0.01 || white.DegreesOfFreedom != 90 || !white.SmallSampleAdjusted {
		t.Fatalf("correct VAR rejected as non-white: %+v", white)
	}
	reject, err := white.Reject(0.01)
	if err != nil || reject {
		t.Fatalf("unexpected rejection=%v err=%v", reject, err)
	}
	underfit, err := FitVAR(data, nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	nonwhite, err := underfit.WhitenessTest(12, true)
	if err != nil {
		t.Fatal(err)
	}
	if nonwhite.PValue >= 1e-8 {
		t.Fatalf("underfit residual dependence missed: %+v", nonwhite)
	}
}

func TestVARWhitenessErrorsAndChiSquareTail(t *testing.T) {
	model, err := FitVAR(generateKnownVAR(3, 100), nil, 2)
	if err != nil {
		t.Fatal(err)
	}
	for _, lag := range []int{2, model.Observations()} {
		if _, err := model.WhitenessTest(lag, true); !errors.Is(err, ErrInvalidWhitenessLag) {
			t.Errorf("lag %d: got %v", lag, err)
		}
	}
	if _, err := (&VARWhitenessResult{PValue: 0.5}).Reject(1); !errors.Is(err, ErrInvalidAlpha) {
		t.Errorf("bad alpha: got %v", err)
	}
	if got := chiSquareUpperTail(5.991464547107979, 2); math.Abs(got-0.05) > 2e-12 {
		t.Errorf("chi-square tail=%g, want 0.05", got)
	}
	if !math.IsNaN(chiSquareUpperTail(-1, 2)) {
		t.Fatal("negative chi-square statistic accepted")
	}
}
