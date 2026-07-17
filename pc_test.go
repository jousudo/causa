package causa

import (
	"errors"
	"math"
	"math/rand"
	"sort"
	"testing"
)

// --- Synthetic linear-Gaussian SEM generator ----------------------------
//
// genSEM draws n i.i.d. samples from a linear structural equation model whose
// weighted adjacency is w (w[parent][child] is the structural coefficient, 0 =
// no edge) and whose topological order is topo. Each variable is the weighted
// sum of its parents plus independent standard-Gaussian noise. The draw is
// seeded so the whole test suite is deterministic.
func genSEM(seed int64, n, p int, w [][]float64, topo []int) [][]float64 {
	rng := rand.New(rand.NewSource(seed))
	data := make([][]float64, p)
	for i := range data {
		data[i] = make([]float64, n)
	}
	for t := 0; t < n; t++ {
		for _, node := range topo {
			v := rng.NormFloat64()
			for parent := 0; parent < p; parent++ {
				if w[parent][node] != 0 {
					v += w[parent][node] * data[parent][t]
				}
			}
			data[node][t] = v
		}
	}
	return data
}

func zeroW(p int) [][]float64 {
	w := make([][]float64, p)
	for i := range w {
		w[i] = make([]float64, p)
	}
	return w
}

// --- Edge-set comparison helpers ----------------------------------------

func edgeSet(edges []Edge) map[[2]int]bool {
	m := make(map[[2]int]bool, len(edges))
	for _, e := range edges {
		m[[2]int{e.From, e.To}] = true
	}
	return m
}

// assertCPDAG checks that g has exactly the given directed and undirected edges
// (undirected given as From<To pairs).
func assertCPDAG(t *testing.T, g *CPDAG, wantDir, wantUndir [][2]int) {
	t.Helper()
	gotDir := edgeSet(g.DirectedEdges())
	gotUndir := edgeSet(g.UndirectedEdges())
	wantDirSet := make(map[[2]int]bool)
	for _, e := range wantDir {
		wantDirSet[e] = true
	}
	wantUndirSet := make(map[[2]int]bool)
	for _, e := range wantUndir {
		wantUndirSet[e] = true
	}
	if !equalSet(gotDir, wantDirSet) {
		t.Errorf("directed edges = %v, want %v\n%s", sortedPairs(gotDir), sortedPairs(wantDirSet), g)
	}
	if !equalSet(gotUndir, wantUndirSet) {
		t.Errorf("undirected edges = %v, want %v\n%s", sortedPairs(gotUndir), sortedPairs(wantUndirSet), g)
	}
}

func equalSet(a, b map[[2]int]bool) bool {
	if len(a) != len(b) {
		return false
	}
	for k := range a {
		if !b[k] {
			return false
		}
	}
	return true
}

func sortedPairs(m map[[2]int]bool) [][2]int {
	out := make([][2]int, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i][0] != out[j][0] {
			return out[i][0] < out[j][0]
		}
		return out[i][1] < out[j][1]
	})
	return out
}

// --- Ground-truth recovery on known small DAGs --------------------------

func TestPCChainUndirected(t *testing.T) {
	// A -> B -> C. Markov equivalence class = {A->B->C, A<-B->C, A<-B<-C}: the
	// skeleton is recovered but no edge is oriented (no unshielded collider).
	w := zeroW(3)
	w[0][1] = 0.8
	w[1][2] = 0.8
	data := genSEM(1, 5000, 3, w, []int{0, 1, 2})
	g, err := PCStable(data, []string{"A", "B", "C"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertCPDAG(t, g, nil, [][2]int{{0, 1}, {1, 2}})
}

func TestPCForkUndirected(t *testing.T) {
	// Fork B -> A, B -> C. Same equivalence class as the chain: skeleton A-B-C,
	// no v-structure (B separates A and C), so nothing is oriented.
	w := zeroW(3)
	w[1][0] = 0.8
	w[1][2] = 0.8
	data := genSEM(2, 5000, 3, w, []int{1, 0, 2})
	g, err := PCStable(data, []string{"A", "B", "C"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertCPDAG(t, g, nil, [][2]int{{0, 1}, {1, 2}})
}

func TestPCColliderOriented(t *testing.T) {
	// Collider A -> C <- B. A and B are marginally independent, so the v-structure
	// is oriented: both arrowheads into C are compelled.
	w := zeroW(3)
	w[0][2] = 0.8
	w[1][2] = 0.8
	data := genSEM(3, 5000, 3, w, []int{0, 1, 2})
	g, err := PCStable(data, []string{"A", "B", "C"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertCPDAG(t, g, [][2]int{{0, 2}, {1, 2}}, nil)
}

func TestPCDiamond(t *testing.T) {
	// A -> B, A -> C, B -> D, C -> D. The only unshielded collider is B -> D <- C;
	// A-B and A-C remain reversible. Expected CPDAG: B->D, C->D directed; A-B, A-C
	// undirected.
	w := zeroW(4)
	w[0][1] = 0.8
	w[0][2] = 0.8
	w[1][3] = 0.8
	w[2][3] = 0.8
	data := genSEM(4, 6000, 4, w, []int{0, 1, 2, 3})
	g, err := PCStable(data, []string{"A", "B", "C", "D"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertCPDAG(t, g, [][2]int{{1, 3}, {2, 3}}, [][2]int{{0, 1}, {0, 2}})
}

// TestPCRandomDAGSkeletonAndColliders builds a random 7-node DAG (edges only
// from lower to higher index, so index order is a valid topological order) and
// checks recovery against ground truth derived INDEPENDENTLY from the true DAG:
// the recovered skeleton must match exactly, every recovered arrowhead must
// agree with the true edge direction (no reversals), and every true unshielded
// collider must be oriented.
func TestPCRandomDAGSkeletonAndColliders(t *testing.T) {
	const p = 7
	rng := rand.New(rand.NewSource(20260716))
	w := zeroW(p)
	trueEdge := make([][]bool, p) // trueEdge[i][j] => i -> j in the true DAG
	for i := range trueEdge {
		trueEdge[i] = make([]bool, p)
	}
	for i := 0; i < p; i++ {
		for j := i + 1; j < p; j++ {
			if rng.Float64() < 0.4 {
				c := 0.8
				if rng.Float64() < 0.5 {
					c = -0.8
				}
				w[i][j] = c
				trueEdge[i][j] = true
			}
		}
	}
	topo := []int{0, 1, 2, 3, 4, 5, 6}
	data := genSEM(999, 6000, p, w, topo)

	g, err := PCStable(data, nil, &PCOptions{Alpha: 0.05})
	if err != nil {
		t.Fatal(err)
	}

	trueAdj := func(i, j int) bool { return trueEdge[i][j] || trueEdge[j][i] }

	// 1) Skeleton must match exactly.
	for i := 0; i < p; i++ {
		for j := i + 1; j < p; j++ {
			if g.Adjacent(i, j) != trueAdj(i, j) {
				t.Errorf("skeleton mismatch on (%d,%d): got adjacent=%v, want %v", i, j, g.Adjacent(i, j), trueAdj(i, j))
			}
		}
	}

	// 2) Every recovered directed edge must match the true orientation.
	for _, e := range g.DirectedEdges() {
		if !trueEdge[e.From][e.To] {
			t.Errorf("recovered %d->%d, but true DAG has no such directed edge (reversed or spurious)", e.From, e.To)
		}
	}

	// 3) Every true unshielded collider must be oriented in the recovered CPDAG.
	for k := 0; k < p; k++ {
		for i := 0; i < p; i++ {
			if i == k || !trueEdge[i][k] {
				continue
			}
			for j := 0; j < p; j++ {
				if j == k || j == i || !trueEdge[j][k] {
					continue
				}
				if trueAdj(i, j) {
					continue // shielded
				}
				// i -> k <- j is a true unshielded collider.
				if !g.Directed(i, k) || !g.Directed(j, k) {
					t.Errorf("true unshielded collider %d->%d<-%d not oriented (got %d-%d dir=%v, %d-%d dir=%v)",
						i, k, j, i, k, g.Directed(i, k), j, k, g.Directed(j, k))
				}
			}
		}
	}
}

// --- v-structure conflict policy ----------------------------------------

// TestVStructureConflictPolicy locks the never-reverse conflict policy. Two
// unshielded colliders compete for opposite arrowheads on the shared edge 1-2:
// the collider centred at node 1 (processed first, k=1) wins with 2->1, and the
// later collider centred at node 2 (k=2) is denied its 1->2 arrowhead rather than
// silently reversing the edge.
func TestVStructureConflictPolicy(t *testing.T) {
	// Skeleton: 1-2, 0-2, 1-3; 0,1 non-adjacent; 2,3 non-adjacent.
	const p = 4
	adj := make([][]bool, p)
	for i := range adj {
		adj[i] = make([]bool, p)
	}
	setUndirected := func(i, j int) { adj[i][j] = true; adj[j][i] = true }
	setUndirected(1, 2)
	setUndirected(0, 2)
	setUndirected(1, 3)

	sepset := make([][][]int, p)
	for i := range sepset {
		sepset[i] = make([][]int, p)
	}
	// Non-adjacent pairs, each separated by the empty set (so both colliders fire).
	setSep := func(i, j int) { sepset[i][j] = []int{}; sepset[j][i] = []int{} }
	setSep(0, 1)
	setSep(2, 3)
	setSep(0, 3)

	orientVStructures(adj, sepset, p)

	if !directed(adj, 2, 1) {
		t.Errorf("expected the first collider's arrowhead 2->1 to be preserved, got adj[2][1]=%v adj[1][2]=%v", adj[2][1], adj[1][2])
	}
	if directed(adj, 1, 2) {
		t.Error("edge 1-2 was reversed to 1->2: conflict policy must never overwrite an existing arrowhead")
	}
	if !directed(adj, 0, 2) {
		t.Error("non-conflicting arrowhead 0->2 should still be applied")
	}
	if !directed(adj, 3, 1) {
		t.Error("arrowhead 3->1 should be applied")
	}
}

// --- Meek rules, one minimal graph per rule ------------------------------

func newAdj(p int) [][]bool {
	adj := make([][]bool, p)
	for i := range adj {
		adj[i] = make([]bool, p)
	}
	return adj
}
func addDirected(adj [][]bool, i, j int)   { adj[i][j] = true; adj[j][i] = false }
func addUndirected(adj [][]bool, i, j int) { adj[i][j] = true; adj[j][i] = true }

func TestMeekR1(t *testing.T) {
	// a->b, b-c, a,c non-adjacent  =>  b->c.  (a=0,b=1,c=2)
	adj := newAdj(3)
	addDirected(adj, 0, 1)
	addUndirected(adj, 1, 2)
	if !meekR1(adj, 1, 2, 3) {
		t.Fatal("meekR1 should fire on b-c")
	}
	applyMeekRules(adj, 3)
	if !directed(adj, 1, 2) {
		t.Errorf("R1 should orient 1->2; got adj[1][2]=%v adj[2][1]=%v", adj[1][2], adj[2][1])
	}
}

func TestMeekR2(t *testing.T) {
	// a->c->b and a-b  =>  a->b.  (a=0,b=1,c=2)
	adj := newAdj(3)
	addDirected(adj, 0, 2)
	addDirected(adj, 2, 1)
	addUndirected(adj, 0, 1)
	if !meekR2(adj, 0, 1, 3) {
		t.Fatal("meekR2 should fire on a-b")
	}
	applyMeekRules(adj, 3)
	if !directed(adj, 0, 1) {
		t.Errorf("R2 should orient 0->1; got adj[0][1]=%v adj[1][0]=%v", adj[0][1], adj[1][0])
	}
}

func TestMeekR3(t *testing.T) {
	// a-b, a-c, a-d, c->b, d->b, c,d non-adjacent  =>  a->b. (a=0,b=1,c=2,d=3)
	adj := newAdj(4)
	addUndirected(adj, 0, 1)
	addUndirected(adj, 0, 2)
	addUndirected(adj, 0, 3)
	addDirected(adj, 2, 1)
	addDirected(adj, 3, 1)
	if !meekR3(adj, 0, 1, 4) {
		t.Fatal("meekR3 should fire on a-b")
	}
	// R1/R2 must not be the reason it orients.
	if meekR1(adj, 0, 1, 4) || meekR2(adj, 0, 1, 4) {
		t.Fatal("R3 test graph unexpectedly triggers R1/R2")
	}
	applyMeekRules(adj, 4)
	if !directed(adj, 0, 1) {
		t.Errorf("R3 should orient 0->1; got adj[0][1]=%v adj[1][0]=%v", adj[0][1], adj[1][0])
	}
}

func TestMeekR4(t *testing.T) {
	// a-b, a-c, a-d, d->c, c->b, b,d non-adjacent  =>  a->b. (a=0,b=1,c=2,d=3)
	adj := newAdj(4)
	addUndirected(adj, 0, 1)
	addUndirected(adj, 0, 2)
	addUndirected(adj, 0, 3)
	addDirected(adj, 3, 2)
	addDirected(adj, 2, 1)
	if !meekR4(adj, 0, 1, 4) {
		t.Fatal("meekR4 should fire on a-b")
	}
	if meekR1(adj, 0, 1, 4) || meekR2(adj, 0, 1, 4) || meekR3(adj, 0, 1, 4) {
		t.Fatal("R4 test graph unexpectedly triggers R1/R2/R3")
	}
	applyMeekRules(adj, 4)
	if !directed(adj, 0, 1) {
		t.Errorf("R4 should orient 0->1; got adj[0][1]=%v adj[1][0]=%v", adj[0][1], adj[1][0])
	}
}

// --- Determinism ---------------------------------------------------------

func TestPCDeterministic(t *testing.T) {
	w := zeroW(4)
	w[0][2] = 0.8
	w[1][2] = 0.8
	w[2][3] = 0.8
	data := genSEM(11, 4000, 4, w, []int{0, 1, 2, 3})
	g1, err := PCStable(data, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	g2, err := PCStable(data, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if g1.String() != g2.String() {
		t.Errorf("non-deterministic output:\n%s\nvs\n%s", g1, g2)
	}
}

// TestPCSkeletonPermutationInvariant is the defining PC-stable guarantee: the
// recovered skeleton does not depend on the order in which the variables are
// presented.
func TestPCSkeletonPermutationInvariant(t *testing.T) {
	const p = 5
	w := zeroW(p)
	w[0][2] = 0.8 // collider at 2
	w[1][2] = 0.8
	w[2][3] = 0.8 // chain onward
	w[3][4] = 0.8
	data := genSEM(5, 5000, p, w, []int{0, 1, 2, 3, 4})

	base, err := PCStable(data, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	perm := []int{3, 0, 4, 1, 2} // perm[newIndex] = oldIndex
	permData := make([][]float64, p)
	for newIdx, oldIdx := range perm {
		permData[newIdx] = data[oldIdx]
	}
	permG, err := PCStable(permData, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	for oi := 0; oi < p; oi++ {
		for oj := oi + 1; oj < p; oj++ {
			ni, nj := indexOf(perm, oi), indexOf(perm, oj)
			if base.Adjacent(oi, oj) != permG.Adjacent(ni, nj) {
				t.Errorf("skeleton not permutation-invariant on original pair (%d,%d): base=%v permuted=%v",
					oi, oj, base.Adjacent(oi, oj), permG.Adjacent(ni, nj))
			}
		}
	}
}

func indexOf(perm []int, oldIdx int) int {
	for newIdx, o := range perm {
		if o == oldIdx {
			return newIdx
		}
	}
	return -1
}

// --- Input validation ----------------------------------------------------

func TestPCStableErrors(t *testing.T) {
	good := genSEM(1, 100, 3, func() [][]float64 {
		w := zeroW(3)
		w[0][1] = 0.8
		w[1][2] = 0.8
		return w
	}(), []int{0, 1, 2})

	ragged := [][]float64{{1, 2, 3, 4}, {1, 2, 3}}
	withNaNData := [][]float64{{1, 2, 3, 4}, {1, 2, math.NaN(), 4}}

	tests := []struct {
		name  string
		data  [][]float64
		names []string
		opts  *PCOptions
		want  error
	}{
		{"one variable", [][]float64{{1, 2, 3, 4}}, nil, nil, ErrTooFewVariables},
		{"ragged", ragged, nil, nil, ErrUnequalLengths},
		{"too few samples", [][]float64{{1, 2, 3}, {4, 5, 6}}, nil, nil, ErrTooFewSamples},
		{"nan", withNaNData, nil, nil, ErrNonFinite},
		{"name count", good, []string{"only-one"}, nil, ErrNameCount},
		{"alpha zero-is-default-ok/negative", good, nil, &PCOptions{Alpha: -0.1}, ErrInvalidAlpha},
		{"alpha too big", good, nil, &PCOptions{Alpha: 1.5}, ErrInvalidAlpha},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := PCStable(tc.data, tc.names, tc.opts)
			if !errors.Is(err, tc.want) {
				t.Errorf("got err=%v, want %v", err, tc.want)
			}
		})
	}
}

// TestPCCITestErrorPropagates ensures a CITest failure aborts the run rather than
// being silently swallowed.
func TestPCCITestErrorPropagates(t *testing.T) {
	data := genSEM(1, 100, 3, func() [][]float64 {
		w := zeroW(3)
		w[0][1] = 0.8
		w[1][2] = 0.8
		return w
	}(), []int{0, 1, 2})
	sentinel := errors.New("boom")
	failing := func(_ [][]float64, _, _ int, _ []int) (float64, error) { return 0, sentinel }
	_, err := PCStable(data, nil, &PCOptions{CITest: failing})
	if !errors.Is(err, sentinel) {
		t.Errorf("got err=%v, want sentinel from CITest", err)
	}
}
