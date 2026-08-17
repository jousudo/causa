package causa

import (
	"errors"
	"math"
)

// Errors returned by ADFTest and ADFResult.Reject.
var (
	// ErrADFInvalidLag is returned when the augmentation lag is negative. Zero
	// is valid (the plain, non-augmented Dickey-Fuller test).
	ErrADFInvalidLag = errors.New("causa: ADF augmentation lag must be non-negative")

	// ErrADFTooShort is returned when the series is too short to fit the ADF
	// regression with a positive residual degrees of freedom for the requested
	// lag and deterministic terms.
	ErrADFTooShort = errors.New("causa: series too short for the requested ADF lag and trend")

	// ErrADFAlpha is returned by Reject for a significance level other than the
	// tabulated 0.01, 0.05, or 0.10.
	ErrADFAlpha = errors.New("causa: ADF significance level must be 0.01, 0.05, or 0.10")
)

// ADFTrend selects the deterministic terms included in the ADF regression
// besides the lagged level and the augmentation lags. The choice changes the
// null distribution of the statistic, so the critical values are trend-specific.
type ADFTrend int

const (
	// ADFNone includes no constant and no trend (a zero-mean series).
	ADFNone ADFTrend = iota
	// ADFConstant includes a constant (drift) — the common default for a series
	// with a non-zero level but no deterministic trend.
	ADFConstant
	// ADFConstantTrend includes a constant and a linear time trend.
	ADFConstantTrend
)

// ADFResult holds the outcome of an Augmented Dickey-Fuller unit-root test.
//
// The headline figure is Statistic: the (one-sided, left-tail) Dickey-Fuller
// t-statistic on the lagged-level coefficient ρ in
//
//	Δyₜ = (deterministic terms) + ρ·yₜ₋₁ + Σᵢ δᵢ·Δyₜ₋ᵢ + εₜ
//
// Under the null hypothesis of a unit root ρ = 0; a sufficiently NEGATIVE
// statistic rejects the unit root in favour of stationarity. Because ρ̂ is a
// near-unit-root estimate, the statistic does NOT follow Student's t — it
// follows the Dickey-Fuller distribution, so Reject compares it against
// trend-specific Dickey-Fuller critical values rather than normal quantiles.
type ADFResult struct {
	// Statistic is the Dickey-Fuller t-statistic on ρ̂ (signed).
	Statistic float64
	// Coefficient is the estimated lagged-level coefficient ρ̂.
	Coefficient float64
	// Lag is the augmentation lag order used.
	Lag int
	// Trend is the deterministic-term specification used.
	Trend ADFTrend
	// Observations is the number of rows in the ADF regression.
	Observations int
}

// ADFTest runs the Augmented Dickey-Fuller unit-root test on y with `lag`
// augmentation lags (lag >= 0; 0 is the plain Dickey-Fuller test) and the
// deterministic terms selected by `trend`.
//
// The statistic is the t-ratio of the lagged-level coefficient ρ̂. It is
// computed from the residual sums of squares of the full regression and the
// same regression with the lagged level removed — the single-restriction F is
// the square of the t-ratio, and the sign of ρ̂ recovers the (signed) t. This
// reuses the package's backward-stable QR least-squares solver (fitOLS) rather
// than forming the normal equations, matching GrangerTest.
//
// Errors: ErrADFInvalidLag (lag < 0), ErrNonFinite (NaN/Inf in y), ErrADFTooShort
// (series too short for a positive residual df), and ErrSingular (a constant or
// collinear design, e.g. a perfectly constant series).
func ADFTest(y []float64, lag int, trend ADFTrend) (*ADFResult, error) {
	if lag < 0 {
		return nil, ErrADFInvalidLag
	}
	n := len(y)
	for i := 0; i < n; i++ {
		if !isFinite(y[i]) {
			return nil, ErrNonFinite
		}
	}

	det := detTermCount(trend) // 0, 1, or 2 deterministic columns
	kFull := det + 1 + lag     // deterministic + lagged level + augmentation lags

	// Row r models Δy at original time t = r + lag + 1 (needs y[t-lag-1] for the
	// deepest augmentation difference), so the earliest usable t is lag+1 and the
	// number of rows is n - lag - 1.
	rows := n - lag - 1
	if rows < kFull+1 {
		return nil, ErrADFTooShort
	}

	// rhoCol is the fixed column index of the lagged level y[t-1] in the full
	// design: after the deterministic terms.
	rhoCol := det

	yResp := make([]float64, rows)
	full := make([][]float64, rows)
	restricted := make([][]float64, rows)
	for r := 0; r < rows; r++ {
		t := r + lag + 1
		yResp[r] = y[t] - y[t-1]

		fr := make([]float64, kFull)
		c := 0
		if trend == ADFConstant || trend == ADFConstantTrend {
			fr[c] = 1
			c++
		}
		if trend == ADFConstantTrend {
			fr[c] = float64(r + 1) // linear trend over the regression sample
			c++
		}
		fr[c] = y[t-1] // lagged level (c == rhoCol here)
		c++
		for i := 1; i <= lag; i++ {
			fr[c] = y[t-i] - y[t-i-1] // Δy_{t-i}
			c++
		}
		full[r] = fr

		// Restricted design = full design without the lagged-level column.
		rr := make([]float64, 0, kFull-1)
		rr = append(rr, fr[:rhoCol]...)
		rr = append(rr, fr[rhoCol+1:]...)
		restricted[r] = rr
	}

	fitFull, err := fitOLS(full, yResp)
	if err != nil {
		return nil, err
	}
	rho := fitFull.coef[rhoCol]

	dfDen := rows - kFull // > 0 by the rows >= kFull+1 guard

	var stat float64
	switch {
	case fitFull.rss <= 0:
		// The full model fits perfectly. The t-ratio diverges unless ρ̂ is zero.
		if rho == 0 {
			stat = 0
		} else {
			stat = math.Copysign(math.Inf(1), rho)
		}
	default:
		var rssRestricted float64
		if kFull-1 == 0 {
			// The restricted model has no regressors (ADFNone with lag 0): its
			// residual is the response itself, so RSS is Σ(Δyₜ)².
			for _, v := range yResp {
				rssRestricted += v * v
			}
		} else {
			fitRes, resErr := fitOLS(restricted, yResp)
			if resErr != nil {
				return nil, resErr
			}
			rssRestricted = fitRes.rss
		}
		diff := rssRestricted - fitFull.rss
		if diff < 0 {
			diff = 0 // clamp floating-point rounding: the full model nests it
		}
		f := diff / (fitFull.rss / float64(dfDen))
		stat = math.Copysign(math.Sqrt(f), rho)
	}

	return &ADFResult{
		Statistic:    stat,
		Coefficient:  rho,
		Lag:          lag,
		Trend:        trend,
		Observations: rows,
	}, nil
}

// Reject reports whether the unit-root null is rejected at alpha (i.e. whether
// the series is judged STATIONARY): true when the statistic is more negative
// than the trend-specific Dickey-Fuller critical value. alpha must be 0.01,
// 0.05, or 0.10.
//
// The critical values are MacKinnon's asymptotic (large-sample) Dickey-Fuller
// values; they do not apply a finite-sample correction, so a borderline
// statistic on a short series should be read as indicative, not definitive.
func (r *ADFResult) Reject(alpha float64) (bool, error) {
	crit, ok := adfCriticalValue(r.Trend, alpha)
	if !ok {
		return false, ErrADFAlpha
	}
	return r.Statistic < crit, nil
}

// detTermCount returns the number of deterministic regression columns for trend.
func detTermCount(trend ADFTrend) int {
	switch trend {
	case ADFConstant:
		return 1
	case ADFConstantTrend:
		return 2
	default:
		return 0
	}
}

// adfCriticalValue returns MacKinnon's asymptotic Dickey-Fuller critical value
// for the given trend and significance level, and whether alpha is tabulated.
// The values are the τ (tau) large-sample quantiles for the no-constant,
// constant, and constant-plus-trend regressions.
func adfCriticalValue(trend ADFTrend, alpha float64) (float64, bool) {
	var table map[float64]float64
	switch trend {
	case ADFNone:
		table = map[float64]float64{0.01: -2.5658, 0.05: -1.9393, 0.10: -1.6156}
	case ADFConstant:
		table = map[float64]float64{0.01: -3.4335, 0.05: -2.8621, 0.10: -2.5671}
	case ADFConstantTrend:
		table = map[float64]float64{0.01: -3.9638, 0.05: -3.4126, 0.10: -3.1279}
	default:
		return 0, false
	}
	crit, ok := table[alpha]
	return crit, ok
}
