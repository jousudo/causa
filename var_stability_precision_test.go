package causa

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"
)

func symmetricVAR1Model(roots []float64) *VARModel {
	dimension := len(roots)
	q := make([][]float64, dimension)
	for row := range q {
		q[row] = make([]float64, dimension)
		for column := range q[row] {
			scale := math.Sqrt(2 / float64(dimension))
			if column == 0 {
				scale = math.Sqrt(1 / float64(dimension))
			}
			q[row][column] = scale * math.Cos(math.Pi*(float64(row)+0.5)*float64(column)/float64(dimension))
		}
	}
	matrix := make([][]float64, dimension)
	for row := range matrix {
		matrix[row] = make([]float64, dimension)
		for column := range matrix[row] {
			for root := range roots {
				matrix[row][column] += q[row][root] * roots[root] * q[column][root]
			}
		}
	}
	return &VARModel{names: make([]string, dimension), lags: 1, coef: [][][]float64{matrix}}
}

func TestVARStabilityDenseKnownSpectrum(t *testing.T) {
	want := []float64{-0.98, -0.75, -0.51, -0.22, -0.08, 0.04, 0.17, 0.31, 0.49, 0.68, 0.86, 0.99999998}
	result, err := symmetricVAR1Model(want).Stability(nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Status != VARStable || math.Abs(result.SpectralRadius-0.99999998) > 2e-11 {
		t.Fatalf("result=%+v, want stable radius 0.99999998", result)
	}
	got := result.Eigenvalues()
	matched := make([]bool, len(got))
	for _, root := range want {
		closest := -1
		distance := 1.0
		for index, eigenvalue := range got {
			if candidate := cmplx.Abs(eigenvalue - complex(root, 0)); !matched[index] && candidate < distance {
				closest, distance = index, candidate
			}
		}
		if closest < 0 || distance > 2e-10 {
			t.Fatalf("no eigenvalue matches root %.17g; got %v", root, got)
		}
		matched[closest] = true
	}

	boundary := append([]float64(nil), want...)
	boundary[len(boundary)-1] = 0.999999995
	boundaryResult, err := symmetricVAR1Model(boundary).Stability(nil)
	if err != nil || boundaryResult.Status != VARStabilityIndeterminate {
		t.Fatalf("boundary result=%+v err=%v", boundaryResult, err)
	}
}

func TestVARStabilityDefaultDimensionBudget(t *testing.T) {
	oversized := &VARModel{names: make([]string, 65), lags: 1}
	if _, err := oversized.Stability(nil); !errors.Is(err, ErrVARCompanionTooLarge) {
		t.Fatalf("default dimension budget: got %v", err)
	}
	if _, err := scalarVARModel(0.5).Stability(&VARStabilityOptions{MaxCompanionCells: -1}); err != nil {
		t.Fatalf("explicitly uncapped small model: %v", err)
	}
}

func TestVARWhitenessRejectsSingularInnovations(t *testing.T) {
	model := &VARModel{
		names:        []string{"x", "y"},
		lags:         1,
		observations: 5,
		residuals: [][]float64{
			{1, -1, 1, -1, 0},
			{1, -1, 1, -1, 0},
		},
	}
	if _, err := model.WhitenessTest(2, true); !errors.Is(err, ErrSingular) {
		t.Fatalf("singular innovations: got %v", err)
	}
}

func TestEigenvalues2x2AvoidsSmallRootCancellation(t *testing.T) {
	first, second := eigenvalues2x2(1, 3, 0, 1e-18)
	if cmplx.Abs(first) < cmplx.Abs(second) {
		first, second = second, first
	}
	if cmplx.Abs(first-1) > 1e-15 {
		t.Fatalf("large root=%g, want 1", first)
	}
	if cmplx.Abs(second-1e-18) > 1e-30 {
		t.Fatalf("small root=%.17g, want 1e-18", second)
	}
}
