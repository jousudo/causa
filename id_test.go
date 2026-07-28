package causa

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

// --- random discrete latent SCM harness ----------------------------------
//
// The rigorous test of an identification algorithm: pick a RANDOM discrete
// structural causal model consistent with the diagram (one hidden common cause
// per bidirected edge), compute both the observational joint P(V) and the TRUE
// interventional P(y | do(x)) by brute force over the full latent model, then
// check that the symbolic estimand — evaluated on P(V) alone — reproduces the
// truth. If the estimand is right it matches for every random parameterization.

type scmNode struct {
	card    int
	parents []int       // global variable indices (observed 0..n-1, latents n..)
	cpt     [][]float64 // cpt[parentConfig] = distribution over `card` states
}

func randDist(k int, rng *rand.Rand) []float64 {
	d := make([]float64, k)
	s := 0.0
	for i := range d {
		d[i] = 0.1 + rng.Float64() // strictly positive => full support, no 0/0
		s += d[i]
	}
	for i := range d {
		d[i] /= s
	}
	return d
}

// buildSCM draws a random discrete SCM for g: each observed vertex is a random
// CPT of its observed parents plus the latents on its incident bidirected edges;
// each bidirected edge contributes one latent root with a random prior.
func buildSCM(g *Diagram, obsCard []int, latCard int, seed int64) (nodes []scmNode, gcardFull []int) {
	rng := rand.New(rand.NewSource(seed))
	n := g.Order()
	var latEdges [][2]int
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if g.HasBidirected(i, j) {
				latEdges = append(latEdges, [2]int{i, j})
			}
		}
	}
	L := len(latEdges)
	gcardFull = make([]int, n+L)
	for v := 0; v < n; v++ {
		gcardFull[v] = obsCard[v]
	}
	for e := 0; e < L; e++ {
		gcardFull[n+e] = latCard
	}
	nodes = make([]scmNode, n+L)
	for v := 0; v < n; v++ {
		parents := append([]int(nil), g.Parents(v)...)
		for e, pr := range latEdges {
			if pr[0] == v || pr[1] == v {
				parents = append(parents, n+e)
			}
		}
		sort.Ints(parents)
		sz := factorSize(parents, gcardFull)
		cpt := make([][]float64, sz)
		for c := 0; c < sz; c++ {
			cpt[c] = randDist(obsCard[v], rng)
		}
		nodes[v] = scmNode{card: obsCard[v], parents: parents, cpt: cpt}
	}
	for e := 0; e < L; e++ {
		nodes[n+e] = scmNode{card: latCard, parents: nil, cpt: [][]float64{randDist(latCard, rng)}}
	}
	return nodes, gcardFull
}

func iota0(m int) []int {
	out := make([]int, m)
	for i := range out {
		out[i] = i
	}
	return out
}

// scmWeight returns the product of the CPT entries for a full assignment,
// skipping the observed variables in `exclude` (used to realize do(X): the
// intervened variables' own mechanisms are removed).
func scmWeight(nodes []scmNode, gcardFull, a []int, exclude map[int]bool, n int) float64 {
	w := 1.0
	for v := 0; v < len(nodes); v++ {
		if v < n && exclude[v] {
			continue
		}
		nd := nodes[v]
		w *= nd.cpt[flatIndex(nd.parents, gcardFull, a)][a[v]]
	}
	return w
}

// observationalJoint sums the full latent model down to a joint over the observed
// variables.
func observationalJoint(nodes []scmNode, gcardFull, obsCard []int, n int) *Distribution {
	observed := iota0(n)
	joint := make([]float64, factorSize(observed, gcardFull))
	forEachAssign(iota0(len(gcardFull)), gcardFull, func(a []int) {
		joint[flatIndex(observed, gcardFull, a)] += scmWeight(nodes, gcardFull, a, nil, n)
	})
	d, err := NewDistribution(obsCard, joint)
	if err != nil {
		panic(err)
	}
	return d
}

// trueInterventional computes P(y | do(x)) over Y∪X by brute-force truncated
// factorization on the full latent model.
func trueInterventional(nodes []scmNode, gcardFull []int, n int, y, x []int) *Distribution {
	yx := sortedUnionInts(y, x)
	truth := make([]float64, factorSize(yx, gcardFull))
	xset := map[int]bool{}
	for _, v := range x {
		xset[v] = true
	}
	var rest []int // observed non-X plus all latents: everything summed over
	for v := 0; v < n; v++ {
		if !xset[v] {
			rest = append(rest, v)
		}
	}
	for e := n; e < len(gcardFull); e++ {
		rest = append(rest, e)
	}
	forEachAssign(x, gcardFull, func(xa []int) {
		forEachAssign(rest, gcardFull, func(ra []int) {
			a := make([]int, len(gcardFull))
			copy(a, ra)
			for _, v := range x {
				a[v] = xa[v]
			}
			truth[flatIndex(yx, gcardFull, a)] += scmWeight(nodes, gcardFull, a, xset, n)
		})
	})
	return &Distribution{vars: yx, card: cardsFor(yx, gcardFull), prob: truth}
}

// validateIdentifiable asserts the query is identifiable and that its estimand,
// evaluated on the observational joint, matches the true interventional
// distribution for several random SCM parameterizations.
func validateIdentifiable(t *testing.T, g *Diagram, y, x []int) {
	t.Helper()
	res, err := Identify(g, y, x)
	if err != nil {
		t.Fatal(err)
	}
	if !res.Identifiable {
		t.Fatalf("expected identifiable, got hedge: %s", res)
	}
	n := g.Order()
	obsCard := make([]int, n)
	for i := range obsCard {
		obsCard[i] = 2
	}
	for _, seed := range []int64{1, 2, 3, 20260728} {
		nodes, gcardFull := buildSCM(g, obsCard, 2, seed)
		joint := observationalJoint(nodes, gcardFull, obsCard, n)
		got, err := res.Estimand.Evaluate(joint)
		if err != nil {
			t.Fatal(err)
		}
		want := trueInterventional(nodes, gcardFull, n, y, x)
		if !sortedEqual(got.Vars(), want.vars) {
			t.Fatalf("seed %d: estimand vars = %v, want %v\nestimand: %s", seed, got.Vars(), want.vars, res)
		}
		for i := range want.prob {
			if math.Abs(got.prob[i]-want.prob[i]) > 1e-9 {
				t.Fatalf("seed %d: estimand ≠ truth at cell %d: got %g want %g\nestimand: %s\ngot:  %v\nwant: %v",
					seed, i, got.prob[i], want.prob[i], res, got.prob, want.prob)
			}
		}
	}
}

// --- ground-truth graphs -------------------------------------------------

func TestIDBackdoor(t *testing.T) {
	// Confounder Z observed: X→Y, Z→X, Z→Y. Classic back-door.
	// X=0, Y=1, Z=2.
	g, err := NewDiagram([]string{"X", "Y", "Z"},
		[][2]int{{0, 1}, {2, 0}, {2, 1}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	validateIdentifiable(t, g, []int{1}, []int{0})
}

func TestIDFrontdoor(t *testing.T) {
	// X→M→Y with a latent common cause of X and Y: X↔Y. The front-door case —
	// identifiable DESPITE the unobserved confounder, which is why FCI/ID matter.
	// X=0, M=1, Y=2.
	g, err := NewDiagram([]string{"X", "M", "Y"},
		[][2]int{{0, 1}, {1, 2}}, [][2]int{{0, 2}})
	if err != nil {
		t.Fatal(err)
	}
	validateIdentifiable(t, g, []int{2}, []int{0})
}

func TestIDDirectNoConfounding(t *testing.T) {
	// X→Y, nothing else: P(y|do(x)) = P(y|x).
	g, err := NewDiagram([]string{"X", "Y"}, [][2]int{{0, 1}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	validateIdentifiable(t, g, []int{1}, []int{0})
}

func TestIDEmptyIntervention(t *testing.T) {
	// do() with empty X is just the observational marginal P(y).
	g, err := NewDiagram([]string{"X", "Y"}, [][2]int{{0, 1}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	validateIdentifiable(t, g, []int{1}, nil)
}

func TestIDFrontdoorChain(t *testing.T) {
	// Generalized front door: X→M1→M2→Y fully mediates X→Y, with a latent common
	// cause X↔Y. The mediating chain still identifies the effect.
	// X=0, M1=1, M2=2, Y=3.
	g, err := NewDiagram([]string{"X", "M1", "M2", "Y"},
		[][2]int{{0, 1}, {1, 2}, {2, 3}}, [][2]int{{0, 3}})
	if err != nil {
		t.Fatal(err)
	}
	validateIdentifiable(t, g, []int{3}, []int{0})
}

func TestIDConfoundedFirstLegNotIdentifiable(t *testing.T) {
	// X→Z→Y with the FIRST leg confounded (X↔Z) is a bow arc feeding Y: the effect
	// P(y|do(x)) inherits the non-identifiability of P(z|do(x)). X=0, Z=1, Y=2.
	g, err := NewDiagram([]string{"X", "Z", "Y"},
		[][2]int{{0, 1}, {1, 2}}, [][2]int{{0, 1}})
	if err != nil {
		t.Fatal(err)
	}
	res, err := Identify(g, []int{2}, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	if res.Identifiable {
		t.Errorf("confounded first leg must be non-identifiable, got %s", res)
	}
}

func TestIDBowArcNotIdentifiable(t *testing.T) {
	// The bow arc: X→Y with a latent common cause X↔Y. The canonical
	// NON-identifiable effect. X=0, Y=1.
	g, err := NewDiagram([]string{"X", "Y"}, [][2]int{{0, 1}}, [][2]int{{0, 1}})
	if err != nil {
		t.Fatal(err)
	}
	res, err := Identify(g, []int{1}, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	if res.Identifiable {
		t.Errorf("bow arc must be non-identifiable, got estimand %s", res)
	}
	larger, smaller := res.Hedge()
	if len(larger) == 0 || len(smaller) == 0 {
		t.Errorf("expected a hedge witness, got %v ⊇ %v", larger, smaller)
	}
}

func TestIDInstrumentNotIdentifiable(t *testing.T) {
	// Instrumental variable: Z→X→Y with a latent X↔Y. The effect of X on Y is
	// NOT point-identifiable. Z=0, X=1, Y=2.
	g, err := NewDiagram([]string{"Z", "X", "Y"},
		[][2]int{{0, 1}, {1, 2}}, [][2]int{{1, 2}})
	if err != nil {
		t.Fatal(err)
	}
	res, err := Identify(g, []int{2}, []int{1})
	if err != nil {
		t.Fatal(err)
	}
	if res.Identifiable {
		t.Errorf("IV effect X→Y must be non-identifiable, got %s", res)
	}
}

// --- input validation ----------------------------------------------------

func TestIdentifyErrors(t *testing.T) {
	g, err := NewDiagram([]string{"X", "Y", "Z"}, [][2]int{{0, 1}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		y, x []int
		want error
	}{
		{"empty y", nil, []int{0}, ErrEmptyOutcome},
		{"overlap", []int{1}, []int{1}, ErrOverlappingQuery},
		{"out of range", []int{9}, []int{0}, ErrEdgeOutOfRange},
		{"dup in y", []int{1, 1}, []int{0}, ErrDuplicateQueryVar},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := Identify(g, tc.y, tc.x); err == nil || !isErr(err, tc.want) {
				t.Errorf("got %v, want %v", err, tc.want)
			}
		})
	}
}

func isErr(got, want error) bool { return got == want }
