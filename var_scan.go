package causa

import (
	"context"
	"errors"
)

const (
	// DefaultMaxVARGrangerTests bounds an all-directions scan before any model
	// fitting. A p-variable system contains p(p-1) ordered hypotheses.
	DefaultMaxVARGrangerTests = 10_000
)

var (
	// ErrVARGrangerTestBudget is returned before a scan whose ordered-pair count
	// exceeds the configured limit.
	ErrVARGrangerTestBudget = errors.New("causa: VAR Granger scan exceeds the configured test budget")
)

// VARGrangerScanOptions configures an all-directions conditional Granger scan.
// The zero value uses alpha 0.05, Holm adjustment, the default test budget, and
// the default VAR design-memory budget.
type VARGrangerScanOptions struct {
	VAROptions
	// Adjustment defaults to PValueHolm. PValueBenjaminiHochberg is an
	// explicitly less conservative FDR option; PValueNone disables correction.
	Adjustment PValueAdjustment
	// Alpha is compared with adjusted p-values. Zero selects 0.05.
	Alpha float64
	// MaxTests bounds p(p-1). Zero selects DefaultMaxVARGrangerTests; a negative
	// value removes the explicit test-count cap.
	MaxTests int
}

// VARGrangerFinding is one ordered cause→effect hypothesis from a scan.
type VARGrangerFinding struct {
	Test           VARGrangerResult
	AdjustedPValue float64
	Significant    bool
}

// VARGrangerScanResult contains a complete hypothesis family. No partial result
// is returned on cancellation, budget exhaustion, singularity, or another fit
// failure.
type VARGrangerScanResult struct {
	Findings   []VARGrangerFinding
	Adjustment PValueAdjustment
	Alpha      float64
	Tests      int
}

// VARGrangerScan tests every ordered variable pair inside one full VAR and
// adjusts the resulting hypothesis family. Holm is the default because it
// controls family-wise error without requiring independent pair tests.
func VARGrangerScan(data [][]float64, names []string, lags int, opts *VARGrangerScanOptions) (*VARGrangerScanResult, error) {
	return VARGrangerScanContext(context.Background(), data, names, lags, opts)
}

// VARGrangerScanContext is VARGrangerScan with cancellation between model fits.
// It reuses one unrestricted design and fit per effect, and one restricted
// design per cause, instead of repeating the full two-model setup for each pair.
func VARGrangerScanContext(ctx context.Context, data [][]float64, names []string, lags int, opts *VARGrangerScanOptions) (*VARGrangerScanResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	resolved, _, err := validateVARData(data, names)
	if err != nil {
		return nil, err
	}
	variables := len(data)
	if variables < 2 {
		return nil, ErrTooFewVariables
	}
	alpha := 0.05
	adjustment := PValueHolm
	maxTests := DefaultMaxVARGrangerTests
	varOptions := (*VAROptions)(nil)
	if opts != nil {
		if opts.Alpha != 0 {
			alpha = opts.Alpha
		}
		adjustment = opts.Adjustment
		if opts.MaxTests != 0 {
			maxTests = opts.MaxTests
		}
		varOptions = &opts.VAROptions
	}
	if !isFinite(alpha) || alpha <= 0 || alpha >= 1 {
		return nil, ErrInvalidAlpha
	}
	if !validPValueAdjustment(adjustment) {
		return nil, ErrInvalidPValueAdjustment
	}
	maxInt := int(^uint(0) >> 1)
	if variables-1 > maxInt/variables {
		return nil, ErrVARGrangerTestBudget
	}
	testCount := variables * (variables - 1)
	if maxTests > 0 && testCount > maxTests {
		return nil, ErrVARGrangerTestBudget
	}
	design, err := buildVARDesign(data, lags, lags, varDesignLimit(varOptions))
	if err != nil {
		return nil, err
	}
	responses := make([][]float64, variables)
	unrestricted := make([]olsResult, variables)
	for effect := 0; effect < variables; effect++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		responses[effect] = append([]float64(nil), data[effect][lags:]...)
		unrestricted[effect], err = fitOLS(design, responses[effect])
		if err != nil {
			return nil, err
		}
	}
	findings := make([]VARGrangerFinding, 0, testCount)
	rawPValues := make([]float64, 0, testCount)
	df1 := lags
	df2 := len(design) - len(design[0])
	for cause := 0; cause < variables; cause++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		restrictedDesign := removeVARCauseLags(design, variables, lags, cause)
		for effect := 0; effect < variables; effect++ {
			if cause == effect {
				continue
			}
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			restricted, err := fitOLS(restrictedDesign, responses[effect])
			if err != nil {
				return nil, err
			}
			f, pValue := nestedFStatistic(restricted.rss, unrestricted[effect].rss, df1, df2)
			conditioned := make([]int, 0, variables-2)
			for variable := 0; variable < variables; variable++ {
				if variable != cause && variable != effect {
					conditioned = append(conditioned, variable)
				}
			}
			test := VARGrangerResult{
				F: f, PValue: pValue, Lags: lags, Observations: len(design), Variables: variables,
				Cause: cause, Effect: effect, CauseName: resolved[cause], EffectName: resolved[effect],
				ConditionedOn: conditioned, RSSRestricted: restricted.rss,
				RSSUnrestricted: unrestricted[effect].rss,
				DFNumerator:     df1, DFDenominator: df2,
			}
			findings = append(findings, VARGrangerFinding{Test: test})
			rawPValues = append(rawPValues, pValue)
		}
	}
	adjusted, err := AdjustPValues(rawPValues, adjustment)
	if err != nil {
		return nil, err
	}
	for index := range findings {
		findings[index].AdjustedPValue = adjusted[index]
		findings[index].Significant = adjusted[index] <= alpha
	}
	return &VARGrangerScanResult{
		Findings: findings, Adjustment: adjustment, Alpha: alpha, Tests: testCount,
	}, nil
}

func removeVARCauseLags(design [][]float64, variables, lags, cause int) [][]float64 {
	restricted := make([][]float64, len(design))
	for row := range design {
		restricted[row] = make([]float64, 0, len(design[row])-lags)
		restricted[row] = append(restricted[row], design[row][0])
		for lag := 0; lag < lags; lag++ {
			base := 1 + lag*variables
			for predictor := 0; predictor < variables; predictor++ {
				if predictor != cause {
					restricted[row] = append(restricted[row], design[row][base+predictor])
				}
			}
		}
	}
	return restricted
}
