package causa

import (
	"errors"
	"math"
	"math/rand"
	"slices"
	"testing"
)

func generateKnownVAR(seed int64, n int) [][]float64 {
	const burn = 300
	rng := rand.New(rand.NewSource(seed))
	full := make([][]float64, 3)
	for i := range full {
		full[i] = make([]float64, n+burn)
	}
	for t := 2; t < n+burn; t++ {
		e0 := 0.6 * rng.NormFloat64()
		e1 := 0.5 * rng.NormFloat64()
		e2 := 0.7 * rng.NormFloat64()
		full[0][t] = 0.55*full[0][t-1] - 0.28*full[0][t-2] + e0
		full[1][t] = 0.35*full[1][t-1] + 0.72*full[0][t-1] + e1
		full[2][t] = 0.40*full[2][t-1] - 0.48*full[1][t-2] + e2
	}
	out := make([][]float64, 3)
	for i := range out {
		out[i] = append([]float64(nil), full[i][burn:]...)
	}
	return out
}

func generateObservedConfounder(seed int64, n int, direct bool) (x, y, z []float64) {
	const burn = 300
	rng := rand.New(rand.NewSource(seed))
	xFull := make([]float64, n+burn)
	yFull := make([]float64, n+burn)
	zFull := make([]float64, n+burn)
	for t := 1; t < n+burn; t++ {
		zFull[t] = 0.82*zFull[t-1] + rng.NormFloat64()
		xFull[t] = 0.90*zFull[t-1] + 0.35*rng.NormFloat64()
		directTerm := 0.0
		if direct {
			directTerm = 0.65 * xFull[t-1]
		}
		yFull[t] = 0.45*yFull[t-1] + 0.90*zFull[t-1] + directTerm + 0.35*rng.NormFloat64()
	}
	return append([]float64(nil), xFull[burn:]...),
		append([]float64(nil), yFull[burn:]...),
		append([]float64(nil), zFull[burn:]...)
}

func TestFitVARRecoversKnownCoefficients(t *testing.T) {
	data := generateKnownVAR(11, 5000)
	model, err := FitVAR(data, []string{"X", "Y", "Z"}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if model.Variables() != 3 || model.Lags() != 2 || model.Observations() != 4998 {
		t.Fatalf("unexpected dimensions: variables=%d lags=%d observations=%d", model.Variables(), model.Lags(), model.Observations())
	}
	checks := []struct {
		lag, response, predictor int
		want                     float64
	}{
		{1, 0, 0, 0.55},
		{2, 0, 0, -0.28},
		{1, 1, 0, 0.72},
		{1, 1, 1, 0.35},
		{2, 2, 1, -0.48},
		{1, 2, 2, 0.40},
	}
	for _, check := range checks {
		got, err := model.Coefficient(check.lag, check.response, check.predictor)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(got-check.want) > 0.04 {
			t.Errorf("A%d[%d,%d]=%.4f, want %.4f", check.lag, check.response, check.predictor, got, check.want)
		}
	}
	cov := model.ResidualCovariance()
	for i := range cov {
		if cov[i][i] <= 0 {
			t.Fatalf("non-positive residual variance: %v", cov)
		}
	}
}

func TestVARGrangerReducesToPairwiseForTwoVariables(t *testing.T) {
	cause, effect := genDriven(4, 700)
	pairwise, err := GrangerTest(cause, effect, 3)
	if err != nil {
		t.Fatal(err)
	}
	conditional, err := VARGrangerTest([][]float64{cause, effect}, []string{"cause", "effect"}, 0, 1, 3)
	if err != nil {
		t.Fatal(err)
	}
	if rel(conditional.F, pairwise.F) > 1e-10 ||
		rel(conditional.RSSRestricted, pairwise.RSSRestricted) > 1e-10 ||
		rel(conditional.RSSUnrestricted, pairwise.RSSUnrestricted) > 1e-10 ||
		math.Abs(conditional.PValue-pairwise.PValue) > 1e-12 {
		t.Fatalf("conditional two-variable test differs: conditional=%+v pairwise=%+v", conditional, pairwise)
	}
}

func TestVARGrangerControlsObservedConfounder(t *testing.T) {
	x, y, z := generateObservedConfounder(19, 2500, false)
	pairwise, err := GrangerTest(x, y, 2)
	if err != nil {
		t.Fatal(err)
	}
	if pairwise.PValue >= 0.01 {
		t.Fatalf("fixture did not expose omitted-variable false positive: p=%g", pairwise.PValue)
	}
	conditional, err := VARGrangerTest([][]float64{x, y, z}, []string{"X", "Y", "Z"}, 0, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	if conditional.PValue <= 0.05 {
		t.Fatalf("observed confounder was not controlled: F=%g p=%g", conditional.F, conditional.PValue)
	}
	if !slices.Equal(conditional.ConditionedOn, []int{2}) {
		t.Fatalf("conditioned on %v, want [2]", conditional.ConditionedOn)
	}
}

func TestVARGrangerRetainsDirectPredictiveEffect(t *testing.T) {
	x, y, z := generateObservedConfounder(23, 1800, true)
	result, err := VARGrangerTest([][]float64{x, y, z}, []string{"X", "Y", "Z"}, 0, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	if result.PValue >= 1e-6 {
		t.Fatalf("direct effect missed: F=%g p=%g", result.F, result.PValue)
	}
	if result.String() == "" {
		t.Fatal("empty String result")
	}
}

func TestSelectVARLagsFindsKnownOrder(t *testing.T) {
	data := generateKnownVAR(31, 3500)
	selection, err := SelectVARLags(data, []string{"X", "Y", "Z"}, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(selection.Scores) != 5 {
		t.Fatalf("scores=%d, want 5", len(selection.Scores))
	}
	if selection.SelectedBIC != 2 || selection.SelectedHQIC != 2 {
		t.Fatalf("selected AIC/BIC/HQIC=%d/%d/%d, want BIC=HQIC=2", selection.SelectedAIC, selection.SelectedBIC, selection.SelectedHQIC)
	}
	for i, score := range selection.Scores {
		if score.Lags != i+1 || !isFinite(score.AIC) || !isFinite(score.BIC) || !isFinite(score.HQIC) {
			t.Fatalf("bad score %d: %+v", i, score)
		}
	}
}

func TestVARResidualAutocorrelationDiagnosesUnderfit(t *testing.T) {
	data := generateKnownVAR(41, 5000)
	underfit, err := FitVAR(data, nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	correct, err := FitVAR(data, nil, 2)
	if err != nil {
		t.Fatal(err)
	}
	badCorr, err := underfit.ResidualAutocorrelation(1)
	if err != nil {
		t.Fatal(err)
	}
	goodCorr, err := correct.ResidualAutocorrelation(1)
	if err != nil {
		t.Fatal(err)
	}
	if maxAbsMatrix(badCorr) <= 2*maxAbsMatrix(goodCorr) {
		t.Fatalf("underfit residual correlation not materially larger: bad=%g good=%g", maxAbsMatrix(badCorr), maxAbsMatrix(goodCorr))
	}
}

func TestVARAccessorsAreDefensive(t *testing.T) {
	model, err := FitVAR(generateKnownVAR(7, 500), []string{"X", "Y", "Z"}, 2)
	if err != nil {
		t.Fatal(err)
	}
	names := model.Nodes()
	names[0] = "changed"
	coef := model.CoefficientMatrices()
	coef[0][0][0] = 999
	residuals := model.Residuals()
	residuals[0][0] = 999
	covariance := model.ResidualCovariance()
	covariance[0][0] = 999
	if model.Nodes()[0] != "X" {
		t.Fatal("Nodes exposed internal storage")
	}
	value, _ := model.Coefficient(1, 0, 0)
	if value == 999 || model.Residuals()[0][0] == 999 || model.ResidualCovariance()[0][0] == 999 {
		t.Fatal("VAR accessor exposed internal storage")
	}
}

func TestVARErrors(t *testing.T) {
	valid := generateKnownVAR(3, 100)
	if _, err := FitVAR(nil, nil, 1); !errors.Is(err, ErrBadVAR) {
		t.Errorf("empty data: got %v", err)
	}
	if _, err := FitVAR(valid, nil, 0); !errors.Is(err, ErrInvalidLags) {
		t.Errorf("zero lag: got %v", err)
	}
	if _, err := FitVAR(valid, []string{"X"}, 1); !errors.Is(err, ErrNameCount) {
		t.Errorf("name count: got %v", err)
	}
	if _, err := FitVAR([][]float64{valid[0], valid[1][:90]}, nil, 1); !errors.Is(err, ErrUnequalLengths) {
		t.Errorf("ragged data: got %v", err)
	}
	nonfinite := cloneMatrix(valid)
	nonfinite[0][10] = math.NaN()
	if _, err := FitVAR(nonfinite, nil, 1); !errors.Is(err, ErrNonFinite) {
		t.Errorf("nonfinite data: got %v", err)
	}
	if _, err := FitVAR(valid, nil, 30); !errors.Is(err, ErrVARTooShort) {
		t.Errorf("too short: got %v", err)
	}
	if _, err := FitVARWithOptions(valid, nil, 1, &VAROptions{MaxDesignCells: 1}); !errors.Is(err, ErrVARDesignTooLarge) {
		t.Errorf("design budget: got %v", err)
	}
	if _, err := SelectVARLags(valid, nil, int(^uint(0)>>1)); !errors.Is(err, ErrVARTooShort) {
		t.Errorf("oversized max lags: got %v", err)
	}
	if _, err := SelectVARLags(valid, nil, 0); !errors.Is(err, ErrInvalidMaxLags) {
		t.Errorf("max lags: got %v", err)
	}
	if _, err := VARGrangerTest(valid, nil, 0, 0, 1); !errors.Is(err, ErrVARIndex) {
		t.Errorf("same index: got %v", err)
	}
	model, err := FitVAR(valid, nil, 2)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.Coefficient(0, 0, 0); !errors.Is(err, ErrVARIndex) {
		t.Errorf("coefficient index: got %v", err)
	}
	if _, err := model.ResidualAutocorrelation(0); !errors.Is(err, ErrInvalidResidualLag) {
		t.Errorf("residual lag: got %v", err)
	}
}

func maxAbsMatrix(matrix [][]float64) float64 {
	var maximum float64
	for _, row := range matrix {
		for _, value := range row {
			if math.Abs(value) > maximum {
				maximum = math.Abs(value)
			}
		}
	}
	return maximum
}
