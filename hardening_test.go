package causa

import (
	"context"
	"errors"
	"math"
	"testing"
)

func boundedDiscoveryData() [][]float64 {
	const n = 40
	data := make([][]float64, 4)
	for v := range data {
		data[v] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		x := float64(i)
		data[0][i] = x
		data[1][i] = math.Sin(x*0.31) + x*0.01
		data[2][i] = math.Cos(x*0.17) - x*0.02
		data[3][i] = math.Sin(x*0.11) + math.Cos(x*0.07)
	}
	return data
}

func TestDenseStateSpaceLimits(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	if _, err := NewDistribution([]int{maxInt, 2}, nil); !errors.Is(err, ErrStateSpaceTooLarge) {
		t.Fatalf("overflow: got %v, want ErrStateSpaceTooLarge", err)
	}
	if _, err := NewDistribution([]int{DefaultMaxDenseCells + 1}, nil); !errors.Is(err, ErrStateSpaceTooLarge) {
		t.Fatalf("default budget: got %v, want ErrStateSpaceTooLarge", err)
	}
	if _, err := NewDistributionWithOptions([]int{2, 2}, make([]float64, 4), &DenseOptions{MaxCells: 3}); !errors.Is(err, ErrStateSpaceTooLarge) {
		t.Fatalf("custom budget: got %v, want ErrStateSpaceTooLarge", err)
	}
	if _, err := SampleDistribution([][]int{{0}}, []int{DefaultMaxDenseCells + 1}); !errors.Is(err, ErrStateSpaceTooLarge) {
		t.Fatalf("sample distribution: got %v, want ErrStateSpaceTooLarge", err)
	}
}

func TestEvaluateDenseBudgetAndNilInput(t *testing.T) {
	joint := handJointXY()
	if _, err := jointExpr([]int{0, 1}).EvaluateWithOptions(joint, &DenseOptions{MaxCells: 3}); !errors.Is(err, ErrStateSpaceTooLarge) {
		t.Fatalf("got %v, want ErrStateSpaceTooLarge", err)
	}
	var expr *Expr
	if _, err := expr.Evaluate(joint); !errors.Is(err, ErrBadEstimand) {
		t.Fatalf("nil expression: got %v, want ErrBadEstimand", err)
	}
	if _, err := jointExpr([]int{0, 1}).Evaluate(nil); !errors.Is(err, ErrBadDistribution) {
		t.Fatalf("nil joint: got %v, want ErrBadDistribution", err)
	}
}

func TestPCStableContextBudget(t *testing.T) {
	graph, diag, err := PCStableContext(context.Background(), boundedDiscoveryData(), nil, &PCOptions{MaxTests: 1})
	if !errors.Is(err, ErrMaxTests) {
		t.Fatalf("got %v, want ErrMaxTests", err)
	}
	if graph != nil {
		t.Fatal("bounded search returned a partial graph")
	}
	if diag == nil || diag.CITests != 1 || !diag.StoppedByMaxTests {
		t.Fatalf("unexpected diagnostics: %+v", diag)
	}
}

func TestFCIContextBudget(t *testing.T) {
	graph, diag, err := FCIContext(context.Background(), boundedDiscoveryData(), nil, &FCIOptions{MaxTests: 1})
	if !errors.Is(err, ErrMaxTests) {
		t.Fatalf("got %v, want ErrMaxTests", err)
	}
	if graph != nil {
		t.Fatal("bounded search returned a partial PAG")
	}
	if diag == nil || diag.CITests != 1 || !diag.StoppedByMaxTests {
		t.Fatalf("unexpected diagnostics: %+v", diag)
	}
}

func TestDiscoveryContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	graph, diag, err := PCStableContext(ctx, boundedDiscoveryData(), nil, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
	if graph != nil || diag == nil || !diag.Canceled || diag.CITests != 0 {
		t.Fatalf("unexpected cancellation result: graph=%v diagnostics=%+v", graph, diag)
	}
}

func TestDiscoveryDiagnostics(t *testing.T) {
	alwaysDependent := func(_ [][]float64, _, _ int, _ []int) (float64, error) {
		return 0, nil
	}
	graph, diag, err := PCStableContext(context.Background(), boundedDiscoveryData(), nil,
		&PCOptions{CITest: alwaysDependent, MaxCondSet: 1})
	if err != nil {
		t.Fatal(err)
	}
	if graph == nil || diag == nil || diag.CITests == 0 || diag.SkeletonCITests != diag.CITests {
		t.Fatalf("unexpected diagnostics: %+v", diag)
	}
	if diag.MaxConditioningSet != 1 || !diag.StoppedByMaxConditioningSet {
		t.Fatalf("conditioning-set cap not reported: %+v", diag)
	}
}

func TestContextAPIsPreserveLegacyResults(t *testing.T) {
	alwaysIndependent := func(_ [][]float64, _, _ int, _ []int) (float64, error) {
		return 1, nil
	}
	pcOpts := &PCOptions{CITest: alwaysIndependent}
	legacyPC, err := PCStable(boundedDiscoveryData(), nil, pcOpts)
	if err != nil {
		t.Fatal(err)
	}
	contextPC, pcDiag, err := PCStableContext(context.Background(), boundedDiscoveryData(), nil, pcOpts)
	if err != nil {
		t.Fatal(err)
	}
	if legacyPC.String() != contextPC.String() || pcDiag.CITests == 0 {
		t.Fatalf("PC mismatch: legacy=%q context=%q diagnostics=%+v", legacyPC, contextPC, pcDiag)
	}

	fciOpts := &FCIOptions{CITest: alwaysIndependent}
	legacyFCI, err := FCI(boundedDiscoveryData(), nil, fciOpts)
	if err != nil {
		t.Fatal(err)
	}
	contextFCI, fciDiag, err := FCIContext(context.Background(), boundedDiscoveryData(), nil, fciOpts)
	if err != nil {
		t.Fatal(err)
	}
	if legacyFCI.String() != contextFCI.String() || fciDiag.CITests == 0 {
		t.Fatalf("FCI mismatch: legacy=%q context=%q diagnostics=%+v", legacyFCI, contextFCI, fciDiag)
	}
}

func TestBootstrapContextAndBudgets(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	called := false
	_, err := BootstrapContext(ctx, 4, func([]int) (float64, error) {
		called = true
		return 0, nil
	}, BootstrapOptions{Resamples: 10})
	if !errors.Is(err, context.Canceled) || called {
		t.Fatalf("cancellation: err=%v called=%v", err, called)
	}

	_, err = Bootstrap(4, func([]int) (float64, error) { return 1, nil },
		BootstrapOptions{Resamples: 11, MaxResamples: 10})
	if !errors.Is(err, ErrBootstrapBudget) {
		t.Fatalf("got %v, want ErrBootstrapBudget", err)
	}
	if _, err = Bootstrap(4, nil, BootstrapOptions{Resamples: 10}); !errors.Is(err, ErrBootstrap) {
		t.Fatalf("nil statistic: got %v, want ErrBootstrap", err)
	}
	if _, err = Bootstrap(4, func([]int) (float64, error) { return math.NaN(), nil },
		BootstrapOptions{Resamples: 10}); !errors.Is(err, ErrBootstrap) {
		t.Fatalf("non-finite point: got %v, want ErrBootstrap", err)
	}

	result, err := Bootstrap(4, func(idx []int) (float64, error) {
		var sum float64
		for _, i := range idx {
			sum += float64(i)
		}
		return sum / float64(len(idx)), nil
	}, BootstrapOptions{Resamples: 20, Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	if result.Attempted != 20 || result.Failed != 0 || len(result.Replicates) != 20 {
		t.Fatalf("unexpected replicate diagnostics: %+v", result)
	}
}
