package causa

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"slices"
	"testing"
)

func indexedMean(values []float64) func([]int) (float64, error) {
	return func(idx []int) (float64, error) {
		var sum float64
		for _, i := range idx {
			sum += values[i]
		}
		return sum / float64(len(idx)), nil
	}
}

func TestMovingBlockLengthOneMatchesIIDBootstrap(t *testing.T) {
	values := []float64{1, 4, 2, 8, 3, 7, 5, 9, 6, 10}
	base := BootstrapOptions{Resamples: 200, Level: 0.90, Seed: 17}
	iid, err := Bootstrap(len(values), indexedMean(values), base)
	if err != nil {
		t.Fatal(err)
	}
	block, err := MovingBlockBootstrap(len(values), indexedMean(values),
		MovingBlockBootstrapOptions{BootstrapOptions: base, BlockLength: 1})
	if err != nil {
		t.Fatal(err)
	}
	if iid.Point != block.Point || iid.Lower != block.Lower || iid.Upper != block.Upper ||
		iid.StdErr != block.StdErr || !slices.Equal(iid.Replicates, block.Replicates) {
		t.Fatalf("block length one differs from iid bootstrap:\\niid=%+v\\nblock=%+v", iid, block)
	}
}

func TestMovingBlockBootstrapUsesContiguousBlocks(t *testing.T) {
	const n, length = 17, 4
	calls := 0
	stat := func(idx []int) (float64, error) {
		calls++
		if calls == 1 {
			return 0, nil // point estimate receives identity indices
		}
		for start := 0; start < len(idx); start += length {
			end := start + length
			if end > len(idx) {
				end = len(idx)
			}
			for i := start + 1; i < end; i++ {
				if idx[i] != idx[i-1]+1 {
					t.Fatalf("non-contiguous block in %v", idx)
				}
			}
		}
		return float64(idx[0]), nil
	}
	_, err := MovingBlockBootstrap(n, stat, MovingBlockBootstrapOptions{
		BootstrapOptions: BootstrapOptions{Resamples: 40, Seed: 3},
		BlockLength:      length,
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestDependentBootstrapsRetainARUncertainty(t *testing.T) {
	const n, burn = 800, 300
	rng := rand.New(rand.NewSource(27))
	full := make([]float64, n+burn)
	for i := 1; i < len(full); i++ {
		full[i] = 0.85*full[i-1] + rng.NormFloat64()
	}
	values := append([]float64(nil), full[burn:]...)
	base := BootstrapOptions{Resamples: 600, Level: 0.95, Seed: 9}
	iid, err := Bootstrap(n, indexedMean(values), base)
	if err != nil {
		t.Fatal(err)
	}
	moving, err := MovingBlockBootstrap(n, indexedMean(values),
		MovingBlockBootstrapOptions{BootstrapOptions: base, BlockLength: 24})
	if err != nil {
		t.Fatal(err)
	}
	stationary, err := StationaryBootstrap(n, indexedMean(values),
		StationaryBootstrapOptions{BootstrapOptions: base, MeanBlockLength: 24})
	if err != nil {
		t.Fatal(err)
	}
	if moving.StdErr < 1.8*iid.StdErr || stationary.StdErr < 1.8*iid.StdErr {
		t.Fatalf("serial dependence was not retained: iid=%g moving=%g stationary=%g",
			iid.StdErr, moving.StdErr, stationary.StdErr)
	}
}

func TestStationaryBootstrapReproducible(t *testing.T) {
	values := []float64{2, 3, 5, 7, 11, 13, 17, 19}
	opts := StationaryBootstrapOptions{
		BootstrapOptions: BootstrapOptions{Resamples: 120, Level: 0.9, Seed: 42},
		MeanBlockLength:  3.5,
	}
	a, err := StationaryBootstrap(len(values), indexedMean(values), opts)
	if err != nil {
		t.Fatal(err)
	}
	b, err := StationaryBootstrap(len(values), indexedMean(values), opts)
	if err != nil {
		t.Fatal(err)
	}
	if a.Point != b.Point || a.Lower != b.Lower || a.Upper != b.Upper ||
		a.StdErr != b.StdErr || !slices.Equal(a.Replicates, b.Replicates) {
		t.Fatalf("fixed seed is not reproducible:\\na=%+v\\nb=%+v", a, b)
	}
}

func TestTimeBootstrapErrorsAndCancellation(t *testing.T) {
	stat := func([]int) (float64, error) { return 1, nil }
	if _, err := MovingBlockBootstrap(10, stat, MovingBlockBootstrapOptions{BlockLength: 0}); !errors.Is(err, ErrInvalidBlockLength) {
		t.Errorf("zero block: got %v", err)
	}
	if _, err := MovingBlockBootstrap(10, stat, MovingBlockBootstrapOptions{BlockLength: 11}); !errors.Is(err, ErrInvalidBlockLength) {
		t.Errorf("long block: got %v", err)
	}
	if _, err := StationaryBootstrap(10, stat, StationaryBootstrapOptions{MeanBlockLength: math.Inf(1)}); !errors.Is(err, ErrInvalidMeanBlockLength) {
		t.Errorf("infinite mean block: got %v", err)
	}
	if _, err := StationaryBootstrap(10, stat, StationaryBootstrapOptions{MeanBlockLength: 0.5}); !errors.Is(err, ErrInvalidMeanBlockLength) {
		t.Errorf("short mean block: got %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := MovingBlockBootstrapContext(ctx, 10, stat, MovingBlockBootstrapOptions{BlockLength: 2}); !errors.Is(err, context.Canceled) {
		t.Errorf("moving cancellation: got %v", err)
	}
	if _, err := StationaryBootstrapContext(ctx, 10, stat, StationaryBootstrapOptions{MeanBlockLength: 2}); !errors.Is(err, context.Canceled) {
		t.Errorf("stationary cancellation: got %v", err)
	}
}
