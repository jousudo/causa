package causa

import (
	"context"
	"errors"
)

// ErrMaxTests is returned when a causal-discovery run reaches the configured
// conditional-independence test budget before the search completes.
var ErrMaxTests = errors.New("causa: conditional-independence test budget exhausted")

// DiscoveryDiagnostics describes the work performed by PCStableContext or
// FCIContext. It is returned on both success and failure so callers can audit
// computational cost and distinguish a completed search from a bounded one.
type DiscoveryDiagnostics struct {
	// CITests is the total number of CI-test callbacks invoked, including one
	// that returns an error.
	CITests int
	// SkeletonCITests and PossibleDSepCITests partition CITests by search phase.
	SkeletonCITests     int
	PossibleDSepCITests int
	// MaxConditioningSet is the largest conditioning-set size actually tested.
	MaxConditioningSet int
	// EdgesRemoved counts skeleton and Possible-D-SEP deletions.
	EdgesRemoved int
	// StoppedByMaxConditioningSet reports that MaxCondSet ended a search level.
	StoppedByMaxConditioningSet bool
	// StoppedBySampleSize reports that Fisher-z residual degrees of freedom
	// prevented a deeper conditioning level.
	StoppedBySampleSize bool
	// StoppedByMaxTests accompanies ErrMaxTests.
	StoppedByMaxTests bool
	// Canceled accompanies context.Canceled or context.DeadlineExceeded.
	Canceled bool
}

type discoveryPhase uint8

const (
	discoverySkeleton discoveryPhase = iota
	discoveryPossibleDSep
)

type discoveryRun struct {
	ctx      context.Context
	maxTests int
	diag     *DiscoveryDiagnostics
}

func newDiscoveryRun(ctx context.Context, maxTests int) *discoveryRun {
	if ctx == nil {
		ctx = context.Background()
	}
	return &discoveryRun{ctx: ctx, maxTests: maxTests, diag: &DiscoveryDiagnostics{}}
}

func (r *discoveryRun) check() error {
	if err := r.ctx.Err(); err != nil {
		r.diag.Canceled = true
		return err
	}
	return nil
}

func (r *discoveryRun) test(ci CITest, phase discoveryPhase, data [][]float64, i, j int, cond []int) (float64, error) {
	if err := r.check(); err != nil {
		return 0, err
	}
	if r.maxTests > 0 && r.diag.CITests >= r.maxTests {
		r.diag.StoppedByMaxTests = true
		return 0, ErrMaxTests
	}
	r.diag.CITests++
	if phase == discoverySkeleton {
		r.diag.SkeletonCITests++
	} else {
		r.diag.PossibleDSepCITests++
	}
	if len(cond) > r.diag.MaxConditioningSet {
		r.diag.MaxConditioningSet = len(cond)
	}
	return ci(data, i, j, cond)
}
