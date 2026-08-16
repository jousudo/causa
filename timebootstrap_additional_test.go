package causa

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

func TestTimeBootstrapGaussianEffectVariants(t *testing.T) {
	g, _ := NewDiagram([]string{"X", "Y", "Z"}, [][2]int{{0, 1}, {2, 0}, {2, 1}}, nil)
	identified, _ := Identify(g, []int{1}, []int{0})
	covariance := [][]float64{{2, 5, 1}, {5, 14, 3}, {1, 3, 1}}
	data := gaussianRows(covariance, 800, rand.New(rand.NewSource(73)))
	full, _ := SampleGaussian(data)
	factor, _ := identified.Estimand.EvaluateGaussian(full)
	want, _ := gaussianEffect(factor, 0, 1)

	moving, err := identified.Estimand.MovingBlockBootstrapGaussianEffect(data, 0, 1,
		MovingBlockBootstrapOptions{
			BootstrapOptions: BootstrapOptions{Resamples: 120, Seed: 4},
			BlockLength:      8,
		})
	if err != nil {
		t.Fatal(err)
	}
	stationary, err := identified.Estimand.StationaryBootstrapGaussianEffect(data, 0, 1,
		StationaryBootstrapOptions{
			BootstrapOptions: BootstrapOptions{Resamples: 120, Seed: 4},
			MeanBlockLength:  8,
		})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(moving.Point-want) > 1e-12 || math.Abs(stationary.Point-want) > 1e-12 {
		t.Fatalf("point estimates differ: moving=%g stationary=%g want=%g", moving.Point, stationary.Point, want)
	}
}

func TestTimeBootstrapGaussianEffectErrors(t *testing.T) {
	e := jointExpr([]int{0, 1})
	moving := MovingBlockBootstrapOptions{BlockLength: 2}
	stationary := StationaryBootstrapOptions{MeanBlockLength: 2}
	if _, err := e.MovingBlockBootstrapGaussianEffect([][]float64{{1, 2, 3}}, 0, 1, moving); !errors.Is(err, ErrTooFewVariables) {
		t.Errorf("moving single variable: got %v", err)
	}
	if _, err := e.StationaryBootstrapGaussianEffect([][]float64{{1, 2, 3}, {4, 5}}, 0, 1, stationary); !errors.Is(err, ErrUnequalLengths) {
		t.Errorf("stationary ragged data: got %v", err)
	}
	data := [][]float64{{1, 2, 3}, {4, 5, 6}}
	if _, err := e.MovingBlockBootstrapGaussianEffect(data, 0, 0, moving); !errors.Is(err, ErrBadGaussian) {
		t.Errorf("moving same variable: got %v", err)
	}
	if _, err := e.StationaryBootstrapGaussianEffect(data, -1, 1, stationary); !errors.Is(err, ErrBadGaussian) {
		t.Errorf("stationary bad index: got %v", err)
	}
}

func TestDependentBootstrapEngineErrors(t *testing.T) {
	opts := MovingBlockBootstrapOptions{BlockLength: 2}
	if _, err := MovingBlockBootstrap(10, nil, opts); !errors.Is(err, ErrBootstrap) {
		t.Errorf("nil statistic: got %v", err)
	}
	opts.BootstrapOptions = BootstrapOptions{Resamples: 11, MaxResamples: 10}
	if _, err := MovingBlockBootstrap(10, func([]int) (float64, error) { return 1, nil }, opts); !errors.Is(err, ErrBootstrapBudget) {
		t.Errorf("resample budget: got %v", err)
	}
	opts.BootstrapOptions = BootstrapOptions{Level: 1}
	if _, err := MovingBlockBootstrap(10, func([]int) (float64, error) { return 1, nil }, opts); !errors.Is(err, ErrBootstrap) {
		t.Errorf("bad level: got %v", err)
	}
	opts.BootstrapOptions = BootstrapOptions{Resamples: 10}
	if _, err := MovingBlockBootstrap(10, func([]int) (float64, error) { return math.NaN(), nil }, opts); !errors.Is(err, ErrBootstrap) {
		t.Errorf("non-finite point: got %v", err)
	}
	calls := 0
	flaky := func([]int) (float64, error) {
		calls++
		if calls == 1 {
			return 1, nil
		}
		return 0, ErrSingular
	}
	if _, err := MovingBlockBootstrap(10, flaky, opts); !errors.Is(err, ErrBootstrap) {
		t.Errorf("failed replicates: got %v", err)
	}
}
