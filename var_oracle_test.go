package causa

import (
	"math"
	"testing"
)

// deterministicVAROracleData mirrors scripts/var_oracle.R without sharing any
// fitting code. The small integer recurrence is exactly representable before
// conversion to float64, keeping the fixture portable across R and Go.
func deterministicVAROracleData() [][]float64 {
	const variables, burn, keep = 3, 40, 120
	total := burn + keep
	values := make([][]float64, variables)
	for variable := range values {
		values[variable] = make([]float64, total)
	}
	state := []int{13, 29, 47}
	innovation := func(channel int) float64 {
		state[channel] = (97*state[channel] + 37) % 9973
		return float64(state[channel])/9973 - 0.5
	}
	for t := 2; t < total; t++ {
		values[0][t] = 0.55*values[0][t-1] - 0.20*values[0][t-2] + innovation(0)
		values[1][t] = 0.15*values[0][t-1] + 0.40*values[1][t-1] +
			0.05*values[2][t-1] - 0.10*values[1][t-2] + innovation(1)
		values[2][t] = -0.25*values[0][t-1] + 0.35*values[1][t-1] +
			0.30*values[2][t-1] + 0.10*values[0][t-2] + innovation(2)
	}
	for variable := range values {
		values[variable] = append([]float64(nil), values[variable][burn:]...)
	}
	return values
}

func closeOracle(got, want float64) bool {
	return math.Abs(got-want) <= 2e-11*math.Max(1, math.Abs(want))
}

func requireOracleVector(t *testing.T, label string, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length: got %d, want %d", label, len(got), len(want))
	}
	for i := range got {
		if !closeOracle(got[i], want[i]) {
			t.Errorf("%s[%d]: got %.17g, want %.17g", label, i, got[i], want[i])
		}
	}
}

func TestVARMatchesIndependentBaseROracle(t *testing.T) {
	data := deterministicVAROracleData()
	model, err := FitVAR(data, []string{"x", "y", "z"}, 2)
	if err != nil {
		t.Fatal(err)
	}
	requireOracleVector(t, "intercept", model.Intercept(), []float64{
		0.012990963991173457, 0.031235087943117131, -0.035573830927875007,
	})
	wantCoefficients := [][][]float64{
		{
			{0.54262614549741117, 0.0039266129726465253, 0.14368952187236067},
			{0.049625182055064727, 0.31868814482200186, 0.096905684814977994},
			{-0.1877882588888076, 0.39913887723862218, 0.17972529577180063},
		},
		{
			{-0.1923684549558769, -0.11427069652444699, 0.08802411908139457},
			{0.03925059979232131, -0.25693978505286019, -0.017870852918587789},
			{-0.14544637048650977, -0.12124359571552745, 0.049699232261251058},
		},
	}
	gotCoefficients := model.CoefficientMatrices()
	for lag := range wantCoefficients {
		for equation := range wantCoefficients[lag] {
			requireOracleVector(t, "coefficient row", gotCoefficients[lag][equation], wantCoefficients[lag][equation])
		}
	}
	wantSigma := [][]float64{
		{0.066487016378488153, 0.0039798675904733722, -0.0015116943390340741},
		{0.0039798675904733722, 0.068657754949529179, 0.011366252167529513},
		{-0.0015116943390340741, 0.011366252167529513, 0.07437043878054854},
	}
	gotSigma := model.ResidualCovariance()
	for row := range wantSigma {
		requireOracleVector(t, "sigma row", gotSigma[row], wantSigma[row])
	}
	score := model.InformationCriteria()
	requireOracleVector(t, "criteria", []float64{score.AIC, score.BIC, score.HQIC}, []float64{
		-7.6622186245871386, -7.1691306829449442, -7.4620103829498481,
	})

	granger, err := VARGrangerTest(data, []string{"x", "y", "z"}, 0, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	requireOracleVector(t, "conditional Granger", []float64{
		granger.F, granger.PValue, granger.RSSRestricted, granger.RSSUnrestricted,
	}, []float64{
		0.41065629384736124, 0.66421825902720477,
		8.1615606556587057, 8.1016150840444432,
	})
	if granger.DFNumerator != 2 || granger.DFDenominator != 111 {
		t.Fatalf("degrees of freedom: got (%d,%d), want (2,111)", granger.DFNumerator, granger.DFDenominator)
	}
}
