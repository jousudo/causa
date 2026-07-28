package causa

import (
	"math"
	"math/rand"
	"testing"
)

// This file validates (*PAGIDResult).Evaluate against brute-force truth on random
// discrete latent SCMs — the parameterization-independent standard used for
// Identify. For each latent DAG we (1) derive its PAG with an ORACLE FCI (a
// d-separation oracle fed to this package's own FCI, so the PAG is exactly the one
// the DAG's equivalence class implies), (2) parameterize the DAG as a random
// discrete SCM, compute the observational joint P(V_obs) by marginalizing the
// latents and the true P(y | do(x)) by intervening, and (3) require the evaluated
// PAG estimand to match the truth.

type latentDAG struct {
	name  string
	total int      // total nodes; observed are 0..obs-1, latents obs..total-1
	obs   int      // number of observed variables
	edges [][2]int // directed edges i -> j over node indices
}

func (d latentDAG) dir() [][]bool {
	m := make([][]bool, d.total)
	for i := range m {
		m[i] = make([]bool, d.total)
	}
	for _, e := range d.edges {
		m[e[0]][e[1]] = true
	}
	return m
}

// dSepOracle is a CITest that ignores data and answers by d-separation in the full
// latent DAG (m-separation with no bidirected edges): p=1 when separated (FCI
// deletes the edge), p=0 when connected.
func (d latentDAG) dSepOracle() CITest {
	dir := d.dir()
	emptyBi := make([][]bool, d.total)
	for i := range emptyBi {
		emptyBi[i] = make([]bool, d.total)
	}
	return func(_ [][]float64, i, j int, cond []int) (float64, error) {
		a := maskFromSlice(d.total, []int{i})
		b := maskFromSlice(d.total, []int{j})
		c := maskFromSlice(d.total, cond)
		if mSeparated(dir, emptyBi, d.total, a, b, c) {
			return 1.0, nil
		}
		return 0.0, nil
	}
}

func (d latentDAG) oraclePAG(t *testing.T) *PAG {
	// Dummy observed data — the oracle ignores the values, but FCI validates shape.
	data := make([][]float64, d.obs)
	for i := range data {
		data[i] = make([]float64, 24)
		for k := range data[i] {
			data[i][k] = float64((i*7 + k*3) % 5)
		}
	}
	names := make([]string, d.obs)
	for i := range names {
		names[i] = string(rune('A' + i))
	}
	g, err := FCI(data, names, &FCIOptions{CITest: d.dSepOracle()})
	if err != nil {
		t.Fatalf("%s: oracle FCI: %v", d.name, err)
	}
	return g
}

// randSCM assigns each node a CPT P(node=1 | parents) with random entries; parents
// are taken in ascending index order.
func (d latentDAG) parents(v int) []int {
	dir := d.dir()
	var ps []int
	for i := 0; i < d.total; i++ {
		if dir[i][v] {
			ps = append(ps, i)
		}
	}
	return ps
}

type scm struct {
	dag latentDAG
	cpt [][]float64 // cpt[v][parentKey] = P(v = 1 | parents = parentKey)
}

func (d latentDAG) randSCM(rng *rand.Rand) scm {
	cpt := make([][]float64, d.total)
	for v := 0; v < d.total; v++ {
		np := len(d.parents(v))
		cpt[v] = make([]float64, 1<<np)
		for k := range cpt[v] {
			cpt[v][k] = 0.1 + 0.8*rng.Float64()
		}
	}
	return scm{dag: d, cpt: cpt}
}

func bit(a, k int) int { return (a >> k) & 1 }

func (s scm) parentKey(a, v int) int {
	key, shift := 0, 0
	for _, p := range s.dag.parents(v) {
		key |= bit(a, p) << shift
		shift++
	}
	return key
}

// factor returns P(node v = its value in a | parents in a), optionally skipping the
// intervened node (whose value is fixed, not sampled).
func (s scm) factor(a, v int) float64 {
	p1 := s.cpt[v][s.parentKey(a, v)]
	if bit(a, v) == 1 {
		return p1
	}
	return 1 - p1
}

// obsJoint returns P over the observed variables (marginalizing latents), as a
// Distribution over 0..obs-1 (each binary), in NewDistribution's mixed-radix order.
func (s scm) obsJoint() *Distribution {
	d := s.dag
	prob := make([]float64, 1<<d.obs)
	for a := 0; a < (1 << d.total); a++ {
		p := 1.0
		for v := 0; v < d.total; v++ {
			p *= s.factor(a, v)
		}
		// observed index in mixed-radix, observed var 0 most significant
		oi := 0
		for k := 0; k < d.obs; k++ {
			oi = oi*2 + bit(a, k)
		}
		prob[oi] += p
	}
	card := make([]int, d.obs)
	for i := range card {
		card[i] = 2
	}
	dist, err := NewDistribution(card, prob)
	if err != nil {
		panic(err)
	}
	return dist
}

// trueDo returns P(y = 1 | do(x = xval)) by intervening in the DAG: the x factor is
// replaced by the fixed value, everything else brute-forced.
func (s scm) trueDo(x, xval, y int) float64 {
	d := s.dag
	num := 0.0
	for a := 0; a < (1 << d.total); a++ {
		if bit(a, x) != xval {
			continue
		}
		p := 1.0
		for v := 0; v < d.total; v++ {
			if v == x {
				continue // intervened: not sampled
			}
			p *= s.factor(a, v)
		}
		if bit(a, y) == 1 {
			num += p
		}
	}
	return num
}

// trueDoCond returns P(y = 1 | do(x = xval), z = zval) by intervening in the DAG
// and conditioning on the context.
func (s scm) trueDoCond(x, xval, y, z, zval int) float64 {
	d := s.dag
	num, den := 0.0, 0.0
	for a := 0; a < (1 << d.total); a++ {
		if bit(a, x) != xval || bit(a, z) != zval {
			continue
		}
		p := 1.0
		for v := 0; v < d.total; v++ {
			if v == x {
				continue
			}
			p *= s.factor(a, v)
		}
		den += p
		if bit(a, y) == 1 {
			num += p
		}
	}
	if den == 0 {
		return math.NaN()
	}
	return num / den
}

func TestPAGEvaluateConditionalAgainstSCM(t *testing.T) {
	cases := []struct {
		dag     latentDAG
		x, y, z int
	}{
		{latentDAG{"collider_visible", 4, 4, [][2]int{{0, 2}, {1, 2}, {2, 3}}}, 2, 3, 0},
		{latentDAG{"collider_chain", 5, 5, [][2]int{{0, 2}, {1, 2}, {2, 3}, {3, 4}}}, 2, 4, 0},
		{latentDAG{"latent_conf_BY", 5, 4, [][2]int{{0, 2}, {1, 2}, {2, 3}, {4, 1}, {4, 3}}}, 2, 3, 0},
	}
	validated := 0
	for _, tc := range cases {
		t.Run(tc.dag.name, func(t *testing.T) {
			pag := tc.dag.oraclePAG(t)
			r, err := pag.IdentifyConditional([]int{tc.y}, []int{tc.x}, []int{tc.z})
			if err != nil {
				t.Fatal(err)
			}
			if !r.Identifiable {
				t.Logf("%s: conditional effect not identifiable from the PAG — skipping", tc.dag.name)
				return
			}
			t.Logf("estimand: %s", r)
			validated++
			for seed := int64(1); seed <= 5; seed++ {
				s := tc.dag.randSCM(rand.New(rand.NewSource(seed)))
				joint := s.obsJoint()
				tab, err := r.Evaluate(joint)
				if err != nil {
					t.Fatalf("evaluate: %v", err)
				}
				for _, xval := range []int{0, 1} {
					for _, zval := range []int{0, 1} {
						got, err := tab.ProbAt(map[int]int{tc.x: xval, tc.z: zval, tc.y: 1})
						if err != nil {
							t.Fatalf("ProbAt: %v", err)
						}
						want := s.trueDoCond(tc.x, xval, tc.y, tc.z, zval)
						if math.Abs(got-want) > 1e-9 {
							t.Errorf("seed %d: P(Y=1|do(X=%d),Z=%d) = %.6f, want %.6f (Δ=%.2e)",
								seed, xval, zval, got, want, math.Abs(got-want))
						}
					}
				}
			}
		})
	}
	if validated == 0 {
		t.Fatal("no identifiable conditional case validated")
	}
}

func TestPAGEvaluateAgainstSCM(t *testing.T) {
	// Structures whose PAG (from the oracle FCI) actually makes P(Y|do(X))
	// identifiable — the observed margin has unshielded colliders that orient
	// enough of the graph. (Many DAG-identifiable effects, e.g. front-door, become
	// a complete-triangle all-circles PAG and are NOT identifiable from the PAG.)
	cases := []struct {
		dag  latentDAG
		x, y int // observed indices
	}{
		// A → X ← B (collider orients arrowheads into X), then X → Y visible.
		{latentDAG{"collider_visible", 4, 4, [][2]int{{0, 2}, {1, 2}, {2, 3}}}, 2, 3},
		// The same, then a directed mediation chain X → M → Y.
		{latentDAG{"collider_chain", 5, 5, [][2]int{{0, 2}, {1, 2}, {2, 3}, {3, 4}}}, 2, 4},
		// Y has a second (oriented) parent as well.
		{latentDAG{"two_colliders", 5, 5, [][2]int{{0, 2}, {1, 2}, {2, 3}, {4, 3}}}, 2, 3},
		// A LATENT confounder U on B and Y — exercises the c-component machinery.
		{latentDAG{"latent_conf_BY", 5, 4, [][2]int{{0, 2}, {1, 2}, {2, 3}, {4, 1}, {4, 3}}}, 2, 3},
	}

	for _, tc := range cases {
		t.Run(tc.dag.name, func(t *testing.T) {
			pag := tc.dag.oraclePAG(t)
			t.Logf("oracle PAG:\n%s", pag)
			r, err := pag.Identify([]int{tc.y}, []int{tc.x})
			if err != nil {
				t.Fatal(err)
			}
			if !r.Identifiable {
				t.Fatalf("%s: effect expected identifiable in the PAG, got non-id", tc.dag.name)
			}
			t.Logf("estimand: %s", r)

			for seed := int64(1); seed <= 5; seed++ {
				s := tc.dag.randSCM(rand.New(rand.NewSource(seed)))
				joint := s.obsJoint()
				tab, err := r.Evaluate(joint)
				if err != nil {
					t.Fatalf("evaluate: %v", err)
				}
				for _, xval := range []int{0, 1} {
					got, err := tab.ProbAt(map[int]int{tc.x: xval, tc.y: 1})
					if err != nil {
						t.Fatalf("ProbAt: %v", err)
					}
					want := s.trueDo(tc.x, xval, tc.y)
					if math.Abs(got-want) > 1e-9 {
						t.Errorf("seed %d: P(Y=1|do(X=%d)) = %.6f, want %.6f (Δ=%.2e)",
							seed, xval, got, want, math.Abs(got-want))
					}
				}
			}
		})
	}
}
