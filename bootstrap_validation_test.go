package causa

import (
	"math/rand"
	"testing"
)

// This validates BootstrapGaussianEffect the way a confidence interval must be
// validated: by its COVERAGE. On a known linear-Gaussian SCM with a known true
// causal effect, we repeatedly draw a fresh finite sample, build a 95% bootstrap
// interval for the effect, and count how often the interval contains the truth. A
// correct 95% interval covers the true value about 95% of the time; a badly
// calibrated one does not. This is the continuous, uncertainty-aware companion to
// the point-estimate SCM validation in gauss_validation_test.go.

func TestBootstrapGaussianEffectCoverage(t *testing.T) {
	cases := []struct {
		name       string
		names      []string
		directed   [][2]int
		bidirected [][2]int
		x, y       int
	}{
		// Back-door with an observed confounder Z (the interval must adjust for Z).
		{"backdoor", []string{"X", "Y", "Z"}, [][2]int{{0, 1}, {2, 0}, {2, 1}}, nil, 0, 1},
		// Front door with a latent common cause X↔Y.
		{"frontdoor", []string{"X", "M", "Y"}, [][2]int{{0, 1}, {1, 2}}, [][2]int{{0, 2}}, 0, 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g, err := NewDiagram(tc.names, tc.directed, tc.bidirected)
			if err != nil {
				t.Fatal(err)
			}
			res, err := Identify(g, []int{tc.y}, []int{tc.x})
			if err != nil || !res.Identifiable {
				t.Fatalf("expected identifiable: %v", err)
			}
			// Fix one random SCM parameterization; its structural effect is the truth
			// every interval is judged against.
			b, omega, total, n := linGaussSCM(g, 7)
			cov := observedCov(b, omega, total, n)
			truth := trueSlope(b, total, tc.x, tc.y)

			rng := rand.New(rand.NewSource(1))
			const trials, nobs = 150, 400
			covered := 0
			for tr := 0; tr < trials; tr++ {
				data := gaussianRows(cov, nobs, rng)
				ci, err := res.Estimand.BootstrapGaussianEffect(data, tc.x, tc.y,
					BootstrapOptions{Resamples: 160, Level: 0.95, Seed: int64(tr)})
				if err != nil {
					t.Fatalf("trial %d: %v", tr, err)
				}
				if truth >= ci.Lower && truth <= ci.Upper {
					covered++
				}
			}
			rate := float64(covered) / float64(trials)
			t.Logf("%s: truth=%.4f coverage=%.3f over %d trials", tc.name, truth, rate, trials)
			// The percentile bootstrap is asymptotic; allow a band around the nominal
			// 0.95 that comfortably admits Monte-Carlo and finite-sample slack but
			// still fails a grossly miscalibrated interval.
			if rate < 0.88 || rate > 0.99 {
				t.Errorf("%s: coverage %.3f outside [0.88, 0.99] for a nominal 0.95 interval", tc.name, rate)
			}
		})
	}
}
