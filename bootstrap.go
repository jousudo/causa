package causa

import (
	"errors"
	"math"
	"math/rand"
	"sort"
)

// ErrBootstrap is returned when a bootstrap run cannot produce a confidence
// interval: an invalid confidence level, or so many resamples degenerated (the
// statistic errored — e.g. a collinear resample with no positive-definite sample
// covariance) that fewer than half, or fewer than two, succeeded.
var ErrBootstrap = errors.New("causa: bootstrap failed (bad level or too many degenerate resamples)")

// --- fitting a distribution to data ---------------------------------------

// SampleGaussian fits a multivariate normal to data by its sample mean and
// unbiased sample covariance, returning a GaussianDistribution over variables
// 0..p-1 ready for Expr.EvaluateGaussian. data holds one variable per outer slice
// (data[v] is variable v's sample), all of the same length n ≥ 2.
//
// It is the bridge from raw observations to the continuous evaluator: pair it with
// an identified estimand's EvaluateGaussian to get P(y | do(x)) from data. The
// covariance must be positive definite — with n ≤ p, or a collinear/constant
// variable, it is not, and ErrNotPositiveDefinite is returned (the same guard
// NewGaussian applies). Errors: ErrBadGaussian (no variables), ErrTooFewSamples
// (n < 2), ErrUnequalLengths, ErrNonFinite.
func SampleGaussian(data [][]float64) (*GaussianDistribution, error) {
	p := len(data)
	if p == 0 {
		return nil, ErrBadGaussian
	}
	n := len(data[0])
	if n < 2 {
		return nil, ErrTooFewSamples
	}
	for _, col := range data {
		if len(col) != n {
			return nil, ErrUnequalLengths
		}
		for _, v := range col {
			if !isFinite(v) {
				return nil, ErrNonFinite
			}
		}
	}
	mean := make([]float64, p)
	for v := 0; v < p; v++ {
		var s float64
		for _, x := range data[v] {
			s += x
		}
		mean[v] = s / float64(n)
	}
	cov := make([][]float64, p)
	for i := range cov {
		cov[i] = make([]float64, p)
	}
	for i := 0; i < p; i++ {
		for j := i; j < p; j++ {
			var s float64
			for k := 0; k < n; k++ {
				s += (data[i][k] - mean[i]) * (data[j][k] - mean[j])
			}
			c := s / float64(n-1)
			cov[i][j] = c
			cov[j][i] = c
		}
	}
	return NewGaussian(mean, cov)
}

// SampleDistribution builds the empirical joint distribution of discrete data: the
// relative frequency of each joint state over variables 0..p-1 with the given
// cardinalities, returning a Distribution ready for Expr.Evaluate. data holds one
// variable per outer slice, each value a state in [0, card[v]); all variables share
// the same length n ≥ 1. It is the discrete counterpart of SampleGaussian.
//
// Errors: ErrBadDistribution (no variables, card mismatch, empty sample, or a value
// out of range), ErrUnequalLengths.
func SampleDistribution(data [][]int, card []int) (*Distribution, error) {
	p := len(data)
	if p == 0 || p != len(card) {
		return nil, ErrBadDistribution
	}
	n := len(data[0])
	if n == 0 {
		return nil, ErrBadDistribution
	}
	total := 1
	for _, c := range card {
		if c < 1 {
			return nil, ErrBadDistribution
		}
		total *= c
	}
	for v := 0; v < p; v++ {
		if len(data[v]) != n {
			return nil, ErrUnequalLengths
		}
		for _, s := range data[v] {
			if s < 0 || s >= card[v] {
				return nil, ErrBadDistribution
			}
		}
	}
	prob := make([]float64, total)
	inc := 1.0 / float64(n)
	for k := 0; k < n; k++ {
		idx := 0
		for v := 0; v < p; v++ {
			idx = idx*card[v] + data[v][k]
		}
		prob[idx] += inc
	}
	return NewDistribution(card, prob)
}

// --- the bootstrap engine -------------------------------------------------

// BootstrapOptions configures a bootstrap run. The zero value is usable: 1000
// resamples, a 95% interval, seed 0.
type BootstrapOptions struct {
	// Resamples is the number of bootstrap resamples B (default 1000 when ≤ 0).
	// More resamples tighten the Monte-Carlo error of the interval, not the
	// interval itself.
	Resamples int
	// Level is the confidence level in (0, 1) (default 0.95 when 0). A 0.95 level
	// yields the 2.5% and 97.5% percentiles of the resample statistics.
	Level float64
	// Seed seeds the resampling RNG, so a run is fully reproducible.
	Seed int64
}

// BootstrapResult is a percentile bootstrap confidence interval for a scalar
// statistic.
type BootstrapResult struct {
	// Point is the statistic on the original (un-resampled) sample.
	Point float64
	// Lower and Upper are the percentile-method interval bounds at Level.
	Lower, Upper float64
	// StdErr is the bootstrap standard error: the standard deviation of the
	// resample statistics.
	StdErr float64
	// Level is the confidence level the interval was computed at.
	Level float64
	// Replicates are the successful resample statistics, ascending. Callers wanting
	// a different summary (a custom percentile, a histogram) can read them directly.
	Replicates []float64
}

// Bootstrap runs a nonparametric row-resampling bootstrap of a scalar statistic
// over n observations, returning a percentile confidence interval. On each of B
// resamples it draws n row indices uniformly with replacement and calls stat with
// them; stat computes the statistic on the rows its indices select (it captures the
// data itself, so this engine is agnostic to how the data is stored — float
// columns, integer categories, anything). The index slice passed to stat is fresh
// per call and may be retained.
//
// A resample for which stat returns an error (a degenerate draw — e.g. a collinear
// sample with no positive-definite covariance) or a non-finite value is skipped;
// if fewer than half of B, or fewer than two, succeed, ErrBootstrap is returned.
// The point estimate is stat on the identity indices 0..n-1 and must succeed (its
// error is propagated).
//
// Scope: this is the basic percentile method — no bias-correction or acceleration
// (BCa). It is honest about sampling variability under the model that the data are
// i.i.d. rows; it does not correct for a skewed or biased bootstrap distribution.
// Errors: ErrTooFewSamples (n < 1), ErrBootstrap (bad Level or too many failures),
// or stat's own error on the original sample.
func Bootstrap(n int, stat func(idx []int) (float64, error), opts BootstrapOptions) (*BootstrapResult, error) {
	if n < 1 {
		return nil, ErrTooFewSamples
	}
	b := opts.Resamples
	if b <= 0 {
		b = 1000
	}
	level := opts.Level
	if level == 0 {
		level = 0.95
	}
	if level <= 0 || level >= 1 {
		return nil, ErrBootstrap
	}

	identity := make([]int, n)
	for i := range identity {
		identity[i] = i
	}
	point, err := stat(identity)
	if err != nil {
		return nil, err
	}

	rng := rand.New(rand.NewSource(opts.Seed))
	reps := make([]float64, 0, b)
	for r := 0; r < b; r++ {
		idx := make([]int, n)
		for i := 0; i < n; i++ {
			idx[i] = rng.Intn(n)
		}
		v, err := stat(idx)
		if err != nil || !isFinite(v) {
			continue
		}
		reps = append(reps, v)
	}
	if len(reps) < 2 || len(reps) < b/2 {
		return nil, ErrBootstrap
	}

	sort.Float64s(reps)
	alpha := 1 - level
	return &BootstrapResult{
		Point:      point,
		Lower:      percentile(reps, alpha/2),
		Upper:      percentile(reps, 1-alpha/2),
		StdErr:     stddev(reps),
		Level:      level,
		Replicates: reps,
	}, nil
}

// BootstrapGaussianEffect bootstraps a confidence interval for the linear-Gaussian
// causal effect the estimand identifies: the slope dE[y | do(x)]/dx, a unit
// contrast E[y | do(x=1)] − E[y | do(x=0)]. On each resample it fits a Gaussian to
// the drawn rows (SampleGaussian), evaluates the estimand (EvaluateGaussian) and
// reads the contrast — so it composes the whole continuous path into an interval.
//
// e is the identified estimand (from Identify / IdentifyConditional on the query
// P(y | do(x))); data holds one variable per outer slice, all length n; x and y are
// the intervention and outcome variable indices (distinct, in range). A resample
// whose sample covariance is not positive definite is skipped by the engine.
// Errors: ErrTooFewVariables (fewer than two columns), ErrUnequalLengths,
// ErrBadGaussian (x/y out of range or equal), or the engine's ErrBootstrap.
func (e *Expr) BootstrapGaussianEffect(data [][]float64, x, y int, opts BootstrapOptions) (*BootstrapResult, error) {
	p := len(data)
	if p < 2 {
		return nil, ErrTooFewVariables
	}
	n := len(data[0])
	for _, col := range data {
		if len(col) != n {
			return nil, ErrUnequalLengths
		}
	}
	if x < 0 || x >= p || y < 0 || y >= p || x == y {
		return nil, ErrBadGaussian
	}
	stat := func(idx []int) (float64, error) {
		g, err := SampleGaussian(selectRows(data, idx))
		if err != nil {
			return 0, err
		}
		f, err := e.EvaluateGaussian(g)
		if err != nil {
			return 0, err
		}
		return gaussianEffect(f, x, y)
	}
	return Bootstrap(n, stat, opts)
}

// gaussianEffect reads the unit-contrast interventional slope of the outcome y in
// the intervention x from an evaluated estimand factor.
func gaussianEffect(f *GaussianFactor, x, y int) (float64, error) {
	hi, err := f.Condition(map[int]float64{x: 1})
	if err != nil {
		return 0, err
	}
	lo, err := f.Condition(map[int]float64{x: 0})
	if err != nil {
		return 0, err
	}
	m1, err := hi.MeanAt(y)
	if err != nil {
		return 0, err
	}
	m0, err := lo.MeanAt(y)
	if err != nil {
		return 0, err
	}
	return m1 - m0, nil
}

// selectRows returns a new column-major data set holding the rows named by idx (in
// order), the resampled data a bootstrap replicate works on.
func selectRows(data [][]float64, idx []int) [][]float64 {
	out := make([][]float64, len(data))
	for v := range data {
		col := make([]float64, len(idx))
		for k, r := range idx {
			col[k] = data[v][r]
		}
		out[v] = col
	}
	return out
}

// percentile returns the p-quantile (p in [0,1]) of an ascending slice by linear
// interpolation between order statistics (the NumPy/R type-7 convention).
func percentile(sorted []float64, p float64) float64 {
	m := len(sorted)
	if m == 1 {
		return sorted[0]
	}
	h := p * float64(m-1)
	lo := int(math.Floor(h))
	if lo >= m-1 {
		return sorted[m-1]
	}
	if lo < 0 {
		return sorted[0]
	}
	return sorted[lo] + (h-float64(lo))*(sorted[lo+1]-sorted[lo])
}

// stddev returns the unbiased sample standard deviation.
func stddev(v []float64) float64 {
	n := len(v)
	var mean float64
	for _, x := range v {
		mean += x
	}
	mean /= float64(n)
	var ss float64
	for _, x := range v {
		d := x - mean
		ss += d * d
	}
	return math.Sqrt(ss / float64(n-1))
}
