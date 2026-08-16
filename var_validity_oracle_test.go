package causa

import (
	"math/cmplx"
	"testing"
)

// Golden values come from scripts/var_validity_oracle.R. The oracle uses base
// R's LAPACK eigensolver, pchisq, pf, and p.adjust, independently of causa's
// stdlib-only implementations.
func TestVARValidityMatchesIndependentBaseROracle(t *testing.T) {
	data := deterministicVAROracleData()
	model, err := FitVAR(data, []string{"x", "y", "z"}, 2)
	if err != nil {
		t.Fatal(err)
	}

	stability, err := model.Stability(nil)
	if err != nil {
		t.Fatal(err)
	}
	wantEigenvalues := []complex128{
		complex(0.40795117966017108, 0.42155424501037309),
		complex(0.40795117966017108, -0.42155424501037309),
		complex(0.19019085854244652, 0.44641126187318453),
		complex(0.19019085854244652, -0.44641126187318453),
		complex(-0.077622245157010217, 0.072085001677589369),
		complex(-0.077622245157010217, -0.072085001677589369),
	}
	gotEigenvalues := stability.Eigenvalues()
	if len(gotEigenvalues) != len(wantEigenvalues) {
		t.Fatalf("eigenvalue count: got %d, want %d", len(gotEigenvalues), len(wantEigenvalues))
	}
	matched := make([]bool, len(gotEigenvalues))
	for i, want := range wantEigenvalues {
		closest := -1
		distance := 1.0
		for j, got := range gotEigenvalues {
			if candidate := cmplx.Abs(got - want); !matched[j] && candidate < distance {
				closest, distance = j, candidate
			}
		}
		if closest < 0 || distance > 2e-11 {
			t.Errorf("eigenvalue[%d]: no match within tolerance for %.17g", i, want)
			continue
		}
		matched[closest] = true
	}

	adjusted, err := model.WhitenessTest(12, true)
	if err != nil {
		t.Fatal(err)
	}
	requireOracleVector(t, "adjusted whiteness", []float64{
		adjusted.Statistic, adjusted.PValue, float64(adjusted.DegreesOfFreedom),
	}, []float64{97.486100361150648, 0.27672386781325903, 90})

	unadjusted, err := model.WhitenessTest(12, false)
	if err != nil {
		t.Fatal(err)
	}
	requireOracleVector(t, "unadjusted whiteness", []float64{
		unadjusted.Statistic, unadjusted.PValue, float64(unadjusted.DegreesOfFreedom),
	}, []float64{91.347601380397606, 0.4405391496358142, 90})

	for _, test := range []struct {
		name       string
		adjustment PValueAdjustment
		want       []float64
	}{
		{"Holm", PValueHolm, []float64{1, 0.030623213548429271, 1, 0.002616698236623553, 0.43698865114671209, 1}},
		{"BH", PValueBenjaminiHochberg, []float64{0.66421825902720477, 0.018373928129057561, 0.66421825902720477, 0.002616698236623553, 0.21849432557335605, 0.66421825902720477}},
	} {
		t.Run(test.name, func(t *testing.T) {
			scan, err := VARGrangerScan(data, []string{"x", "y", "z"}, 2, &VARGrangerScanOptions{
				Adjustment: test.adjustment,
			})
			if err != nil {
				t.Fatal(err)
			}
			got := make([]float64, len(scan.Findings))
			for i, finding := range scan.Findings {
				got[i] = finding.AdjustedPValue
			}
			requireOracleVector(t, "adjusted scan", got, test.want)
		})
	}
}
