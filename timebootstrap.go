package causa

import (
	"context"
	"errors"
	"math/rand"
	"sort"
)

var (
	// ErrInvalidBlockLength is returned when a moving-block length is outside
	// [1,n]. The library requires an explicit choice because block length is a
	// statistical modeling decision, not a harmless implementation default.
	ErrInvalidBlockLength = errors.New("causa: moving-bootstrap block length must be in [1,n]")
	// ErrInvalidMeanBlockLength is returned when the stationary-bootstrap mean
	// block length is non-finite or smaller than one.
	ErrInvalidMeanBlockLength = errors.New("causa: stationary-bootstrap mean block length must be finite and >= 1")
)

// MovingBlockBootstrapOptions configures overlapping moving-block resampling.
type MovingBlockBootstrapOptions struct {
	BootstrapOptions
	// BlockLength is the fixed contiguous block length in [1,n]. A run draws
	// starts uniformly from the n-BlockLength+1 overlapping blocks, concatenates
	// sampled blocks, and truncates the final block to n observations.
	BlockLength int
}

// StationaryBootstrapOptions configures Politis-Romano stationary resampling.
type StationaryBootstrapOptions struct {
	BootstrapOptions
	// MeanBlockLength is the expected geometric block length. At every row the
	// sampler starts a fresh uniformly chosen index with probability 1/L;
	// otherwise it advances circularly to the next observation.
	MeanBlockLength float64
}

// MovingBlockBootstrap bootstraps a scalar statistic while preserving serial
// dependence inside fixed-length contiguous blocks. It is the dependent-row
// counterpart of Bootstrap. The original observations are passed to stat as
// indices, so the engine remains data-layout agnostic.
//
// Scope: validity requires an approximately stationary, weakly dependent
// process and a scientifically justified block length. The non-circular moving
// block bootstrap does not make a nonstationary series stationary and can have
// boundary effects. Use StationaryBootstrap when stationary circular resamples
// and geometrically varying blocks are preferable.
func MovingBlockBootstrap(n int, stat func(idx []int) (float64, error), opts MovingBlockBootstrapOptions) (*BootstrapResult, error) {
	return MovingBlockBootstrapContext(context.Background(), n, stat, opts)
}

// MovingBlockBootstrapContext is MovingBlockBootstrap with cancellation between
// statistic callbacks.
func MovingBlockBootstrapContext(ctx context.Context, n int, stat func(idx []int) (float64, error), opts MovingBlockBootstrapOptions) (*BootstrapResult, error) {
	if n < 1 {
		return nil, ErrTooFewSamples
	}
	if opts.BlockLength < 1 || opts.BlockLength > n {
		return nil, ErrInvalidBlockLength
	}
	blockLength := opts.BlockLength
	blockCount := n - blockLength + 1
	sampler := func(rng *rand.Rand, idx []int) {
		position := 0
		for position < len(idx) {
			start := rng.Intn(blockCount)
			for offset := 0; offset < blockLength && position < len(idx); offset++ {
				idx[position] = start + offset
				position++
			}
		}
	}
	return bootstrapDependentContext(ctx, n, stat, opts.BootstrapOptions, sampler)
}

// StationaryBootstrap bootstraps a scalar statistic with the Politis-Romano
// stationary bootstrap. Resampled blocks have a geometric length distribution
// and wrap circularly, so every resampled position has a uniform marginal index.
//
// Scope: this quantifies sampling variability for an approximately stationary,
// weakly dependent process. MeanBlockLength controls the dependence retained;
// choosing it remains the caller's statistical responsibility.
func StationaryBootstrap(n int, stat func(idx []int) (float64, error), opts StationaryBootstrapOptions) (*BootstrapResult, error) {
	return StationaryBootstrapContext(context.Background(), n, stat, opts)
}

// StationaryBootstrapContext is StationaryBootstrap with cancellation between
// statistic callbacks.
func StationaryBootstrapContext(ctx context.Context, n int, stat func(idx []int) (float64, error), opts StationaryBootstrapOptions) (*BootstrapResult, error) {
	if n < 1 {
		return nil, ErrTooFewSamples
	}
	if !isFinite(opts.MeanBlockLength) || opts.MeanBlockLength < 1 {
		return nil, ErrInvalidMeanBlockLength
	}
	restartProbability := 1 / opts.MeanBlockLength
	sampler := func(rng *rand.Rand, idx []int) {
		current := rng.Intn(n)
		idx[0] = current
		for position := 1; position < len(idx); position++ {
			if rng.Float64() < restartProbability {
				current = rng.Intn(n)
			} else {
				current = (current + 1) % n
			}
			idx[position] = current
		}
	}
	return bootstrapDependentContext(ctx, n, stat, opts.BootstrapOptions, sampler)
}

// MovingBlockBootstrapGaussianEffect is the dependent-time-series analogue of
// Expr.BootstrapGaussianEffect, using moving blocks rather than independent rows.
func (e *Expr) MovingBlockBootstrapGaussianEffect(data [][]float64, x, y int, opts MovingBlockBootstrapOptions) (*BootstrapResult, error) {
	return e.MovingBlockBootstrapGaussianEffectContext(context.Background(), data, x, y, opts)
}

// MovingBlockBootstrapGaussianEffectContext adds cooperative cancellation.
func (e *Expr) MovingBlockBootstrapGaussianEffectContext(ctx context.Context, data [][]float64, x, y int, opts MovingBlockBootstrapOptions) (*BootstrapResult, error) {
	n, stat, err := e.gaussianEffectBootstrapStatistic(data, x, y)
	if err != nil {
		return nil, err
	}
	return MovingBlockBootstrapContext(ctx, n, stat, opts)
}

// StationaryBootstrapGaussianEffect is the dependent-time-series analogue of
// Expr.BootstrapGaussianEffect, using geometrically distributed circular blocks.
func (e *Expr) StationaryBootstrapGaussianEffect(data [][]float64, x, y int, opts StationaryBootstrapOptions) (*BootstrapResult, error) {
	return e.StationaryBootstrapGaussianEffectContext(context.Background(), data, x, y, opts)
}

// StationaryBootstrapGaussianEffectContext adds cooperative cancellation.
func (e *Expr) StationaryBootstrapGaussianEffectContext(ctx context.Context, data [][]float64, x, y int, opts StationaryBootstrapOptions) (*BootstrapResult, error) {
	n, stat, err := e.gaussianEffectBootstrapStatistic(data, x, y)
	if err != nil {
		return nil, err
	}
	return StationaryBootstrapContext(ctx, n, stat, opts)
}

type dependentIndexSampler func(rng *rand.Rand, idx []int)

func bootstrapDependentContext(ctx context.Context, n int, stat func(idx []int) (float64, error), opts BootstrapOptions, sampler dependentIndexSampler) (*BootstrapResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if n < 1 {
		return nil, ErrTooFewSamples
	}
	if stat == nil {
		return nil, ErrBootstrap
	}
	b := opts.Resamples
	if b <= 0 {
		b = 1000
	}
	maxResamples := opts.MaxResamples
	if maxResamples == 0 {
		maxResamples = DefaultMaxBootstrapResamples
	}
	if maxResamples > 0 && b > maxResamples {
		return nil, ErrBootstrapBudget
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
	if !isFinite(point) {
		return nil, ErrBootstrap
	}
	rng := rand.New(rand.NewSource(opts.Seed))
	replicates := make([]float64, 0, b)
	for replicate := 0; replicate < b; replicate++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		idx := make([]int, n)
		sampler(rng, idx)
		value, err := stat(idx)
		if err != nil || !isFinite(value) {
			continue
		}
		replicates = append(replicates, value)
	}
	if len(replicates) < 2 || len(replicates) < b/2 {
		return nil, ErrBootstrap
	}
	sort.Float64s(replicates)
	alpha := 1 - level
	return &BootstrapResult{
		Point: point, Lower: percentile(replicates, alpha/2), Upper: percentile(replicates, 1-alpha/2),
		StdErr: stddev(replicates), Level: level, Replicates: replicates,
		Attempted: b, Failed: b - len(replicates),
	}, nil
}

func (e *Expr) gaussianEffectBootstrapStatistic(data [][]float64, x, y int) (int, func([]int) (float64, error), error) {
	p := len(data)
	if p < 2 {
		return 0, nil, ErrTooFewVariables
	}
	n := len(data[0])
	for _, series := range data {
		if len(series) != n {
			return 0, nil, ErrUnequalLengths
		}
	}
	if x < 0 || x >= p || y < 0 || y >= p || x == y {
		return 0, nil, ErrBadGaussian
	}
	stat := func(idx []int) (float64, error) {
		gaussian, err := SampleGaussian(selectRows(data, idx))
		if err != nil {
			return 0, err
		}
		factor, err := e.EvaluateGaussian(gaussian)
		if err != nil {
			return 0, err
		}
		return gaussianEffect(factor, x, y)
	}
	return n, stat, nil
}
