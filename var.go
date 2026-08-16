package causa

import (
	"errors"
	"fmt"
	"math"
)

const (
	// DefaultMaxVARDesignCells limits one row-major VAR regression design to
	// 128 MiB of float64 cells. VAROptions can override it after a caller reviews
	// the process memory budget.
	DefaultMaxVARDesignCells = 1 << 24
)

// Errors returned by VAR fitting, lag selection, and conditional Granger tests.
var (
	ErrBadVAR             = errors.New("causa: VAR requires at least one aligned variable")
	ErrVARTooShort        = errors.New("causa: series too short for the requested VAR order")
	ErrVARIndex           = errors.New("causa: VAR variable or lag index out of range")
	ErrInvalidMaxLags     = errors.New("causa: maximum VAR lag must be positive")
	ErrVARDesignTooLarge  = errors.New("causa: VAR regression design exceeds the configured cell budget")
	ErrInvalidResidualLag = errors.New("causa: residual lag must be positive and shorter than the residual series")
)

// VAROptions bounds the memory used by VAR regression designs. The zero value
// is safe and uses DefaultMaxVARDesignCells.
type VAROptions struct {
	// MaxDesignCells is the maximum number of float64 cells in one regression
	// design. Zero selects DefaultMaxVARDesignCells; a negative value removes the
	// configured cap, while integer overflow is always rejected.
	MaxDesignCells int
}

// VARModel is an ordinary-least-squares estimate of a reduced-form VAR(p),
//
//	y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + u_t.
//
// Coefficient matrices use [lag][response][predictor] order: A_l[i][j] is the
// coefficient from variable j at lag l+1 to equation i. The model is predictive,
// not a contemporaneously identified structural VAR; innovation covariance does
// not by itself identify instantaneous causal directions.
type VARModel struct {
	names        []string
	lags         int
	observations int
	intercept    []float64
	coef         [][][]float64
	residuals    [][]float64
	sigma        [][]float64
	rss          []float64
	logDetSigma  float64
}

// Variables returns the number of endogenous series in the system.
func (m *VARModel) Variables() int { return len(m.names) }

// Nodes returns a copy of the variable names in model order.
func (m *VARModel) Nodes() []string { return append([]string(nil), m.names...) }

// Lags returns the fitted VAR order p.
func (m *VARModel) Lags() int { return m.lags }

// Observations returns the number of aligned response rows used in the fit.
func (m *VARModel) Observations() int { return m.observations }

// Intercept returns a copy of c, one value per response equation.
func (m *VARModel) Intercept() []float64 { return append([]float64(nil), m.intercept...) }

// Coefficient returns A_lag[response][predictor]. lag is one-based.
func (m *VARModel) Coefficient(lag, response, predictor int) (float64, error) {
	if lag < 1 || lag > m.lags || response < 0 || response >= m.Variables() || predictor < 0 || predictor >= m.Variables() {
		return 0, ErrVARIndex
	}
	return m.coef[lag-1][response][predictor], nil
}

// CoefficientMatrices returns a deep copy of A_1,...,A_p in
// [lag][response][predictor] order.
func (m *VARModel) CoefficientMatrices() [][][]float64 { return cloneCube(m.coef) }

// Residuals returns a variable-major copy of fitted innovations. Every row has
// Observations values and begins at the first fitted response timestamp.
func (m *VARModel) Residuals() [][]float64 { return cloneMatrix(m.residuals) }

// ResidualCovariance returns the maximum-likelihood innovation covariance
// E[uu'] (division by Observations), as used by the information criteria.
func (m *VARModel) ResidualCovariance() [][]float64 { return cloneMatrix(m.sigma) }

// ResidualSumSquares returns one residual sum of squares per response equation.
func (m *VARModel) ResidualSumSquares() []float64 { return append([]float64(nil), m.rss...) }

// VARLagScore holds the Lütkepohl-form information criteria for one VAR order.
// Lower values are preferred. The covariance is the maximum-likelihood residual
// covariance, and the penalty counts every fitted equation coefficient.
type VARLagScore struct {
	Lags                     int
	AIC                      float64
	BIC                      float64
	HQIC                     float64
	LogDetResidualCovariance float64
}

// InformationCriteria reports AIC, BIC, and Hannan-Quinn for this fitted model.
// For comparing different orders, prefer SelectVARLags, which holds the response
// sample fixed at the maximum candidate lag.
func (m *VARModel) InformationCriteria() VARLagScore {
	return varInformationCriteria(m.lags, m.Variables(), m.observations, m.logDetSigma)
}

// VARLagSelection contains every candidate score and the minimizing order for
// each criterion. Ties resolve to the smaller order.
type VARLagSelection struct {
	Scores       []VARLagScore
	SelectedAIC  int
	SelectedBIC  int
	SelectedHQIC int
}

// VARGrangerResult is a conditional, multivariate Granger F-test. It asks
// whether all lags of Cause can be removed from Effect's VAR equation while the
// lags of every other supplied variable remain in both models.
type VARGrangerResult struct {
	F               float64
	PValue          float64
	Lags            int
	Observations    int
	Variables       int
	Cause           int
	Effect          int
	CauseName       string
	EffectName      string
	ConditionedOn   []int
	RSSRestricted   float64
	RSSUnrestricted float64
	DFNumerator     int
	DFDenominator   int
}

// String renders the direction, conditioning dimension, and F-test compactly.
func (r *VARGrangerResult) String() string {
	return fmt.Sprintf(
		"conditional Granger %s -> %s: F(%d,%d)=%.4f, p=%.4g (lags=%d, variables=%d, n=%d)",
		r.CauseName, r.EffectName, r.DFNumerator, r.DFDenominator, r.F, r.PValue,
		r.Lags, r.Variables, r.Observations,
	)
}

// FitVAR estimates a reduced-form VAR with a constant and a common positive lag
// order. data is variable-major: data[v][t]. Rows must be aligned, finite, and
// sampled at a common interval. The fit uses Householder-QR OLS independently
// for each equation, which is equivalent to multivariate least squares because
// every equation has the same regressors.
//
// Statistical scope: the usual inference and information criteria assume an
// approximately covariance-stationary VAR with innovation residuals that are
// not serially correlated. FitVAR does not transform, difference, or silently
// declare nonstationary data valid. ResidualAutocorrelation helps diagnose an
// insufficient lag order, but is not a stationarity test.
func FitVAR(data [][]float64, names []string, lags int) (*VARModel, error) {
	return FitVARWithOptions(data, names, lags, nil)
}

// FitVARWithOptions is FitVAR with an explicit regression-design cell budget.
func FitVARWithOptions(data [][]float64, names []string, lags int, opts *VAROptions) (*VARModel, error) {
	resolved, _, err := validateVARData(data, names)
	if err != nil {
		return nil, err
	}
	return fitVARPrepared(data, resolved, lags, lags, varDesignLimit(opts))
}

// SelectVARLags fits orders 1..maxLags and selects minima of AIC, BIC, and HQIC.
// Every candidate uses the SAME response timestamps maxLags..n-1, so scores are
// comparable rather than rewarding a larger order for discarding more rows.
func SelectVARLags(data [][]float64, names []string, maxLags int) (*VARLagSelection, error) {
	return SelectVARLagsWithOptions(data, names, maxLags, nil)
}

// SelectVARLagsWithOptions is SelectVARLags with a design-memory budget.
func SelectVARLagsWithOptions(data [][]float64, names []string, maxLags int, opts *VAROptions) (*VARLagSelection, error) {
	if maxLags < 1 {
		return nil, ErrInvalidMaxLags
	}
	resolved, observations, err := validateVARData(data, names)
	if err != nil {
		return nil, err
	}
	if observations < 2 || maxLags > (observations-2)/(len(data)+1) {
		return nil, ErrVARTooShort
	}
	limit := varDesignLimit(opts)
	scores := make([]VARLagScore, 0, maxLags)
	for lags := 1; lags <= maxLags; lags++ {
		model, err := fitVARPrepared(data, resolved, lags, maxLags, limit)
		if err != nil {
			return nil, err
		}
		scores = append(scores, model.InformationCriteria())
	}
	result := &VARLagSelection{Scores: scores}
	result.SelectedAIC = scores[0].Lags
	result.SelectedBIC = scores[0].Lags
	result.SelectedHQIC = scores[0].Lags
	bestAIC, bestBIC, bestHQ := scores[0].AIC, scores[0].BIC, scores[0].HQIC
	for _, score := range scores[1:] {
		if score.AIC < bestAIC {
			bestAIC, result.SelectedAIC = score.AIC, score.Lags
		}
		if score.BIC < bestBIC {
			bestBIC, result.SelectedBIC = score.BIC, score.Lags
		}
		if score.HQIC < bestHQ {
			bestHQ, result.SelectedHQIC = score.HQIC, score.Lags
		}
	}
	return result, nil
}

// VARGrangerTest performs a conditional Granger test inside a full VAR. The
// unrestricted Effect equation contains lags of every supplied variable; the
// restricted equation removes only Cause's lags. It therefore controls for the
// observed histories in data and avoids the omitted-variable failure of a
// pairwise test when the relevant common driver is included.
//
// This remains predictive causality, not interventional identification. Hidden
// drivers, nonstationarity, instantaneous effects, nonlinear dynamics, bad lag
// order, and serially correlated innovations can invalidate the p-value.
func VARGrangerTest(data [][]float64, names []string, cause, effect, lags int) (*VARGrangerResult, error) {
	return VARGrangerTestWithOptions(data, names, cause, effect, lags, nil)
}

// VARGrangerTestWithOptions is VARGrangerTest with a design-memory budget.
func VARGrangerTestWithOptions(data [][]float64, names []string, cause, effect, lags int, opts *VAROptions) (*VARGrangerResult, error) {
	resolved, _, err := validateVARData(data, names)
	if err != nil {
		return nil, err
	}
	p := len(data)
	if p < 2 {
		return nil, ErrTooFewVariables
	}
	if cause < 0 || cause >= p || effect < 0 || effect >= p || cause == effect {
		return nil, ErrVARIndex
	}
	design, err := buildVARDesign(data, lags, lags, varDesignLimit(opts))
	if err != nil {
		return nil, err
	}
	y := append([]float64(nil), data[effect][lags:]...)
	fitU, err := fitOLS(design, y)
	if err != nil {
		return nil, err
	}
	restricted := make([][]float64, len(design))
	for row := range design {
		rr := make([]float64, 0, len(design[row])-lags)
		rr = append(rr, design[row][0])
		for lag := 0; lag < lags; lag++ {
			base := 1 + lag*p
			for predictor := 0; predictor < p; predictor++ {
				if predictor != cause {
					rr = append(rr, design[row][base+predictor])
				}
			}
		}
		restricted[row] = rr
	}
	fitR, err := fitOLS(restricted, y)
	if err != nil {
		return nil, err
	}
	df1 := lags
	df2 := len(y) - len(design[0])
	f, pValue := nestedFStatistic(fitR.rss, fitU.rss, df1, df2)
	conditioned := make([]int, 0, p-2)
	for v := 0; v < p; v++ {
		if v != cause && v != effect {
			conditioned = append(conditioned, v)
		}
	}
	return &VARGrangerResult{
		F: f, PValue: pValue, Lags: lags, Observations: len(y), Variables: p,
		Cause: cause, Effect: effect, CauseName: resolved[cause], EffectName: resolved[effect],
		ConditionedOn: conditioned, RSSRestricted: fitR.rss, RSSUnrestricted: fitU.rss,
		DFNumerator: df1, DFDenominator: df2,
	}, nil
}

// ResidualAutocorrelation returns Corr(u_i[t], u_j[t-lag]) for every response i
// and predictor-residual j. Large remaining values warn that the fitted lag
// order may not have whitened the innovations. They are diagnostics, not
// automatically calibrated hypothesis tests.
func (m *VARModel) ResidualAutocorrelation(lag int) ([][]float64, error) {
	if len(m.residuals) == 0 || lag < 1 || lag >= len(m.residuals[0]) {
		return nil, ErrInvalidResidualLag
	}
	p := len(m.residuals)
	n := len(m.residuals[0]) - lag
	out := make([][]float64, p)
	for i := 0; i < p; i++ {
		out[i] = make([]float64, p)
		for j := 0; j < p; j++ {
			x := m.residuals[i][lag:]
			y := m.residuals[j][:n]
			corr, ok := sampleCorrelation(x, y)
			if !ok {
				return nil, ErrSingular
			}
			out[i][j] = corr
		}
	}
	return out, nil
}

func validateVARData(data [][]float64, names []string) ([]string, int, error) {
	if len(data) == 0 {
		return nil, 0, ErrBadVAR
	}
	n := len(data[0])
	if n == 0 {
		return nil, 0, ErrVARTooShort
	}
	for _, series := range data {
		if len(series) != n {
			return nil, 0, ErrUnequalLengths
		}
		for _, value := range series {
			if !isFinite(value) {
				return nil, 0, ErrNonFinite
			}
		}
	}
	resolved := names
	switch {
	case names == nil:
		resolved = make([]string, len(data))
		for i := range resolved {
			resolved[i] = fmt.Sprintf("V%d", i)
		}
	case len(names) != len(data):
		return nil, 0, ErrNameCount
	default:
		resolved = append([]string(nil), names...)
	}
	return resolved, n, nil
}

func fitVARPrepared(data [][]float64, names []string, lags, holdback, maxCells int) (*VARModel, error) {
	design, err := buildVARDesign(data, lags, holdback, maxCells)
	if err != nil {
		return nil, err
	}
	p := len(data)
	nobs := len(design)
	intercept := make([]float64, p)
	coef := make([][][]float64, lags)
	for lag := range coef {
		coef[lag] = make([][]float64, p)
		for response := range coef[lag] {
			coef[lag][response] = make([]float64, p)
		}
	}
	residuals := make([][]float64, p)
	rss := make([]float64, p)
	for response := 0; response < p; response++ {
		y := append([]float64(nil), data[response][holdback:]...)
		fit, err := fitOLS(design, y)
		if err != nil {
			return nil, err
		}
		intercept[response] = fit.coef[0]
		for lag := 0; lag < lags; lag++ {
			for predictor := 0; predictor < p; predictor++ {
				coef[lag][response][predictor] = fit.coef[1+lag*p+predictor]
			}
		}
		residuals[response] = make([]float64, nobs)
		for row := 0; row < nobs; row++ {
			predicted := 0.0
			for col, value := range design[row] {
				predicted += value * fit.coef[col]
			}
			residuals[response][row] = y[row] - predicted
			rss[response] += residuals[response][row] * residuals[response][row]
		}
	}
	sigma := make([][]float64, p)
	for i := 0; i < p; i++ {
		sigma[i] = make([]float64, p)
		for j := 0; j <= i; j++ {
			var sum float64
			for row := 0; row < nobs; row++ {
				sum += residuals[i][row] * residuals[j][row]
			}
			sigma[i][j] = sum / float64(nobs)
			sigma[j][i] = sigma[i][j]
		}
	}
	logDet, ok := logDetSPD(sigma)
	if !ok {
		return nil, ErrSingular
	}
	return &VARModel{
		names: append([]string(nil), names...), lags: lags, observations: nobs,
		intercept: intercept, coef: coef, residuals: residuals, sigma: sigma, rss: rss,
		logDetSigma: logDet,
	}, nil
}

func buildVARDesign(data [][]float64, lags, holdback, maxCells int) ([][]float64, error) {
	if lags < 1 {
		return nil, ErrInvalidLags
	}
	if holdback < lags {
		return nil, ErrInvalidLags
	}
	p := len(data)
	maxInt := int(^uint(0) >> 1)
	if p == 0 || lags > (maxInt-1)/p {
		return nil, ErrVARDesignTooLarge
	}
	columns := 1 + p*lags
	nobs := len(data[0]) - holdback
	if nobs <= columns {
		return nil, ErrVARTooShort
	}
	if nobs > maxInt/columns || (maxCells > 0 && nobs*columns > maxCells) {
		return nil, ErrVARDesignTooLarge
	}
	design := make([][]float64, nobs)
	for row := 0; row < nobs; row++ {
		t := row + holdback
		x := make([]float64, columns)
		x[0] = 1
		for lag := 1; lag <= lags; lag++ {
			for predictor := 0; predictor < p; predictor++ {
				x[1+(lag-1)*p+predictor] = data[predictor][t-lag]
			}
		}
		design[row] = x
	}
	return design, nil
}

func varDesignLimit(opts *VAROptions) int {
	if opts == nil || opts.MaxDesignCells == 0 {
		return DefaultMaxVARDesignCells
	}
	return opts.MaxDesignCells
}

func varInformationCriteria(lags, variables, observations int, logDet float64) VARLagScore {
	params := variables * (1 + variables*lags)
	n := float64(observations)
	k := float64(params)
	return VARLagScore{
		Lags:                     lags,
		AIC:                      logDet + 2*k/n,
		BIC:                      logDet + math.Log(n)*k/n,
		HQIC:                     logDet + 2*math.Log(math.Log(n))*k/n,
		LogDetResidualCovariance: logDet,
	}
}

func nestedFStatistic(rssRestricted, rssUnrestricted float64, df1, df2 int) (float64, float64) {
	diff := rssRestricted - rssUnrestricted
	if diff < 0 {
		diff = 0
	}
	if rssUnrestricted <= 0 {
		if diff > 0 {
			return math.Inf(1), 0
		}
		return 0, 1
	}
	f := (diff / float64(df1)) / (rssUnrestricted / float64(df2))
	return f, fUpperTail(f, float64(df1), float64(df2))
}

func logDetSPD(matrix [][]float64) (float64, bool) {
	l, ok := cholesky(matrix)
	if !ok {
		return 0, false
	}
	var logDet float64
	for i := range l {
		logDet += 2 * math.Log(l[i][i])
	}
	return logDet, isFinite(logDet)
}

func sampleCorrelation(x, y []float64) (float64, bool) {
	if len(x) != len(y) || len(x) < 2 {
		return 0, false
	}
	var meanX, meanY float64
	for i := range x {
		meanX += x[i]
		meanY += y[i]
	}
	meanX /= float64(len(x))
	meanY /= float64(len(y))
	var covariance, varianceX, varianceY float64
	for i := range x {
		dx := x[i] - meanX
		dy := y[i] - meanY
		covariance += dx * dy
		varianceX += dx * dx
		varianceY += dy * dy
	}
	if varianceX <= 0 || varianceY <= 0 {
		return 0, false
	}
	return covariance / math.Sqrt(varianceX*varianceY), true
}

func cloneMatrix(matrix [][]float64) [][]float64 {
	out := make([][]float64, len(matrix))
	for i := range matrix {
		out[i] = append([]float64(nil), matrix[i]...)
	}
	return out
}

func cloneCube(cube [][][]float64) [][][]float64 {
	out := make([][][]float64, len(cube))
	for i := range cube {
		out[i] = cloneMatrix(cube[i])
	}
	return out
}
