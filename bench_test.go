package causa_test

import (
	"math/rand"
	"testing"

	"github.com/jousudo/causa"
)

// benchData builds a deterministic driven pair of the requested length so the
// benchmark measures the test, not the data generation.
func benchData(n int) (cause, effect []float64) {
	rng := rand.New(rand.NewSource(99))
	cause = make([]float64, n)
	effect = make([]float64, n)
	for t := 1; t < n; t++ {
		cause[t] = 0.4*cause[t-1] + rng.NormFloat64()
		effect[t] = 0.5*effect[t-1] + 0.6*cause[t-1] + 0.5*rng.NormFloat64()
	}
	return cause, effect
}

func BenchmarkGrangerTest_n1000_lags4(b *testing.B) {
	cause, effect := benchData(1000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := causa.GrangerTest(cause, effect, 4); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGrangerTest_n5000_lags8(b *testing.B) {
	cause, effect := benchData(5000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := causa.GrangerTest(cause, effect, 8); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGrangerTest_n1000_lags1(b *testing.B) {
	cause, effect := benchData(1000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := causa.GrangerTest(cause, effect, 1); err != nil {
			b.Fatal(err)
		}
	}
}

// benchPCData builds a deterministic n-sample linear-Gaussian panel over p
// variables from a fixed random DAG (edges from lower to higher index), so the
// benchmark measures PCStable, not data generation.
func benchPCData(seed int64, n, p int) [][]float64 {
	rng := rand.New(rand.NewSource(seed))
	w := make([][]float64, p)
	for i := range w {
		w[i] = make([]float64, p)
	}
	for i := 0; i < p; i++ {
		for j := i + 1; j < p; j++ {
			if rng.Float64() < 0.3 {
				w[i][j] = 0.8
			}
		}
	}
	data := make([][]float64, p)
	for i := range data {
		data[i] = make([]float64, n)
	}
	for t := 0; t < n; t++ {
		for node := 0; node < p; node++ {
			v := rng.NormFloat64()
			for parent := 0; parent < node; parent++ {
				if w[parent][node] != 0 {
					v += w[parent][node] * data[parent][t]
				}
			}
			data[node][t] = v
		}
	}
	return data
}

func BenchmarkPCStable_p10_n1000(b *testing.B) {
	data := benchPCData(7, 1000, 10)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := causa.PCStable(data, nil, nil); err != nil {
			b.Fatal(err)
		}
	}
}

// benchLiNGAMData builds a deterministic n-sample non-Gaussian (uniform-noise)
// linear acyclic SEM over p variables from a fixed random DAG (edges from lower
// to higher index), so the benchmark measures DirectLiNGAM, not data generation.
func benchLiNGAMData(seed int64, n, p int) [][]float64 {
	rng := rand.New(rand.NewSource(seed))
	w := make([][]float64, p)
	for i := range w {
		w[i] = make([]float64, p)
	}
	for i := 0; i < p; i++ {
		for j := i + 1; j < p; j++ {
			if rng.Float64() < 0.4 {
				w[i][j] = 0.8
			}
		}
	}
	data := make([][]float64, p)
	for i := range data {
		data[i] = make([]float64, n)
	}
	for t := 0; t < n; t++ {
		for node := 0; node < p; node++ {
			v := rng.Float64()*2 - 1
			for parent := 0; parent < node; parent++ {
				if w[parent][node] != 0 {
					v += w[parent][node] * data[parent][t]
				}
			}
			data[node][t] = v
		}
	}
	return data
}

func BenchmarkFCI_p8_n1000(b *testing.B) {
	data := benchPCData(7, 1000, 8)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := causa.FCI(data, nil, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkIdentify_frontdoorChain(b *testing.B) {
	// X→M1→M2→Y with a latent X↔Y: an identifiable effect that exercises the
	// c-component recursion (Possible-D-SEP-free, symbolic).
	g, err := causa.NewDiagram([]string{"X", "M1", "M2", "Y"},
		[][2]int{{0, 1}, {1, 2}, {2, 3}}, [][2]int{{0, 3}})
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := causa.Identify(g, []int{3}, []int{0}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkIdentifyConditional_context(b *testing.B) {
	// Z→X→Y, Z→Y: a conditional effect P(y|do(x),z) exercising the do-calculus
	// Rule-2 loop (m-separation) plus the ID subroutine.
	g, err := causa.NewDiagram([]string{"X", "Y", "Z"},
		[][2]int{{2, 0}, {0, 1}, {2, 1}}, nil)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := causa.IdentifyConditional(g, []int{1}, []int{0}, []int{2}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDirectLiNGAM_p6_n2000(b *testing.B) {
	data := benchLiNGAMData(7, 2000, 6)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := causa.DirectLiNGAM(data, nil, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFitSEM_p6_n2000(b *testing.B) {
	data := benchLiNGAMData(7, 2000, 6)
	order := []int{0, 1, 2, 3, 4, 5} // benchLiNGAMData edges run low->high index
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := causa.FitSEM(data, nil, order); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkIntervene_p6(b *testing.B) {
	data := benchLiNGAMData(7, 2000, 6)
	s, err := causa.FitSEM(data, nil, []int{0, 1, 2, 3, 4, 5})
	if err != nil {
		b.Fatal(err)
	}
	do := map[string]float64{"V0": 1}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.Intervene(do); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkIdentifyPAG_chain6(b *testing.B) {
	// A o→ B → C → D → E with a latent A ↔ E: an identifiable PAG effect that
	// exercises the Prop.6 recursion over the induced graphs.
	g, err := causa.NewPAG([]string{"A", "B", "C", "D", "E"}, []causa.PAGEdge{
		{A: 0, B: 1, MarkA: causa.Circle, MarkB: causa.Arrow},
		{A: 1, B: 2, MarkA: causa.Tail, MarkB: causa.Arrow},
		{A: 2, B: 3, MarkA: causa.Tail, MarkB: causa.Arrow},
		{A: 3, B: 4, MarkA: causa.Tail, MarkB: causa.Arrow},
		{A: 0, B: 4, MarkA: causa.Arrow, MarkB: causa.Arrow},
	})
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := g.Identify([]int{4}, []int{2}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkIdentifyConditionalPAG_chain(b *testing.B) {
	// A → B → C → D with context B: exercises the Rule-2 m-separation loop plus the
	// IDP subroutine over a PAG.
	g, err := causa.NewPAG([]string{"A", "B", "C", "D"}, []causa.PAGEdge{
		{A: 0, B: 1, MarkA: causa.Tail, MarkB: causa.Arrow},
		{A: 1, B: 2, MarkA: causa.Tail, MarkB: causa.Arrow},
		{A: 2, B: 3, MarkA: causa.Tail, MarkB: causa.Arrow},
	})
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := g.IdentifyConditional([]int{3}, []int{2}, []int{1}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEvaluatePAG(b *testing.B) {
	// A o→ X ← o B, X → Y: identifiable, then evaluate P(Y|do(X)) on a discrete joint.
	g, err := causa.NewPAG([]string{"A", "B", "X", "Y"}, []causa.PAGEdge{
		{A: 0, B: 2, MarkA: causa.Circle, MarkB: causa.Arrow},
		{A: 1, B: 2, MarkA: causa.Circle, MarkB: causa.Arrow},
		{A: 2, B: 3, MarkA: causa.Tail, MarkB: causa.Arrow},
	})
	if err != nil {
		b.Fatal(err)
	}
	r, err := g.Identify([]int{3}, []int{2})
	if err != nil {
		b.Fatal(err)
	}
	prob := make([]float64, 16)
	for i := range prob {
		prob[i] = 1.0 / 16.0
	}
	joint, err := causa.NewDistribution([]int{2, 2, 2, 2}, prob)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := r.Evaluate(joint); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEvaluateGaussian_frontdoorChain(b *testing.B) {
	// Generalized front door X→M1→M2→Y with a latent X↔Y: the c-component estimand
	// is a product of conditionals and marginals, evaluated as a Gaussian factor.
	g, err := causa.NewDiagram([]string{"X", "M1", "M2", "Y"},
		[][2]int{{0, 1}, {1, 2}, {2, 3}}, [][2]int{{0, 3}})
	if err != nil {
		b.Fatal(err)
	}
	r, err := causa.Identify(g, []int{3}, []int{0})
	if err != nil {
		b.Fatal(err)
	}
	cov := [][]float64{
		{2.0, 0.8, 0.5, 0.9},
		{0.8, 2.0, 0.7, 0.6},
		{0.5, 0.7, 2.0, 0.8},
		{0.9, 0.6, 0.8, 2.0},
	}
	joint, err := causa.NewGaussian(make([]float64, 4), cov)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := r.Estimand.EvaluateGaussian(joint); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkBootstrapGaussianEffect_backdoor(b *testing.B) {
	// Back-door SCM data (Z→X, Z→Y, X→Y); 200 resamples of a 500-row sample.
	rng := rand.New(rand.NewSource(1))
	const n = 500
	x := make([]float64, n)
	y := make([]float64, n)
	z := make([]float64, n)
	for i := 0; i < n; i++ {
		zi := rng.NormFloat64()
		xi := zi + rng.NormFloat64()
		z[i], x[i], y[i] = zi, xi, 2*xi+zi+rng.NormFloat64()
	}
	data := [][]float64{x, y, z}
	g, err := causa.NewDiagram([]string{"X", "Y", "Z"}, [][2]int{{0, 1}, {2, 0}, {2, 1}}, nil)
	if err != nil {
		b.Fatal(err)
	}
	r, err := causa.Identify(g, []int{1}, []int{0})
	if err != nil {
		b.Fatal(err)
	}
	opts := causa.BootstrapOptions{Resamples: 200, Seed: 1}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := r.Estimand.BootstrapGaussianEffect(data, 0, 1, opts); err != nil {
			b.Fatal(err)
		}
	}
}
