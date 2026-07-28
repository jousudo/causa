package causa

import (
	"errors"
	"math"
	"testing"
)

// --- hand-built PAG helpers ----------------------------------------------

func newMarks(p int) [][]Mark {
	m := make([][]Mark, p)
	for i := range m {
		m[i] = make([]Mark, p)
	}
	return m
}

// pagEdge sets the endpoint marks of the edge i-j: atI is the mark touching i,
// atJ the mark touching j.
func pagEdge(m [][]Mark, i, j int, atI, atJ Mark) {
	m[j][i] = atI
	m[i][j] = atJ
}

// --- Possible-D-SEP ------------------------------------------------------

// TestPossibleDSep checks that Possible-D-SEP reaches through consecutive
// colliders to a NON-neighbour — the reachability PC's adjacency-only search
// cannot see and that gives FCI its extra power. On A o→ B ↔ C ←o D the set
// Possible-D-SEP(A) is {B, C, D}: B is a neighbour, C is reached because B is a
// collider on A-B-C, and D because C is a collider on B-C-D.
func TestPossibleDSep(t *testing.T) {
	const p = 4 // A=0, B=1, C=2, D=3
	m := newMarks(p)
	pagEdge(m, 0, 1, Circle, Arrow) // A o→ B
	pagEdge(m, 1, 2, Arrow, Arrow)  // B ↔ C
	pagEdge(m, 2, 3, Arrow, Circle) // C ←o D  (arrow at C, circle at D)

	pds := possibleDSep(m, p, 0)
	for _, v := range []int{1, 2, 3} {
		if !pds[v] {
			t.Errorf("Possible-D-SEP(A) should contain node %d", v)
		}
	}
	if pds[0] {
		t.Error("Possible-D-SEP(A) must not contain A itself")
	}
}

// --- individual Zhang rules ----------------------------------------------

func TestRuleR1(t *testing.T) {
	// a *→ b, b o—o c, a,c non-adjacent  =>  b → c.  (a=0,b=1,c=2)
	m := newMarks(3)
	pagEdge(m, 0, 1, Circle, Arrow)  // a *→ b (arrowhead at b)
	pagEdge(m, 1, 2, Circle, Circle) // b o—o c
	if !ruleR1(m, 3) {
		t.Fatal("R1 should fire")
	}
	if !isDirected(m, 1, 2) {
		t.Errorf("R1 should orient 1→2; got mark at 2=%v, mark at 1=%v", m[1][2], m[2][1])
	}
}

func TestRuleR2(t *testing.T) {
	// a → b *→ c, a o—o c  =>  a *→ c (arrowhead at c).  (a=0,b=1,c=2)
	m := newMarks(3)
	pagEdge(m, 0, 1, Tail, Arrow)    // a → b
	pagEdge(m, 1, 2, Circle, Arrow)  // b *→ c
	pagEdge(m, 0, 2, Circle, Circle) // a o—o c
	if !ruleR2(m, 3) {
		t.Fatal("R2 should fire")
	}
	if !isArrow(m, 2, 0) { // arrowhead at c on edge a-c
		t.Errorf("R2 should place an arrowhead at c; mark at 2=%v", m[0][2])
	}
	if !isCircle(m, 0, 2) {
		t.Error("R2 must not touch the a end of a-c (should stay a o→ c)")
	}
}

func TestRuleR3(t *testing.T) {
	// a *→ b ←* c, a *—o d o—* c, d *—o b, a,c non-adjacent  =>  d *→ b.
	// (a=0,b=1,c=2,d=3)
	m := newMarks(4)
	pagEdge(m, 0, 1, Circle, Arrow)  // a *→ b
	pagEdge(m, 2, 1, Circle, Arrow)  // c *→ b
	pagEdge(m, 0, 3, Circle, Circle) // a *—o d
	pagEdge(m, 2, 3, Circle, Circle) // c *—o d
	pagEdge(m, 3, 1, Circle, Circle) // d *—o b
	if !ruleR3(m, 4) {
		t.Fatal("R3 should fire")
	}
	if !isArrow(m, 1, 3) { // arrowhead at b on edge d-b
		t.Errorf("R3 should place an arrowhead at b from d; mark at 1=%v", m[3][1])
	}
}

// TestRuleR4Collider exercises the discriminating-path rule when the middle node
// is NOT in the separating set: the edge closes into the double collider
// a ↔ b ↔ c. Discriminating path ⟨θ, a, b, c⟩ with θ=0, a=1, b=2, c=3.
func TestRuleR4Collider(t *testing.T) {
	const p = 4
	m := newMarks(p)
	pagEdge(m, 0, 1, Circle, Arrow) // θ o→ a
	pagEdge(m, 1, 2, Arrow, Arrow)  // a ↔ b
	pagEdge(m, 1, 3, Tail, Arrow)   // a → c
	pagEdge(m, 2, 3, Circle, Arrow) // b o→ c  (circle at b — the edge to orient)

	sepset := make([][][]int, p)
	for i := range sepset {
		sepset[i] = make([][]int, p)
	}
	// b (=2) NOT in Sepset(θ=0, c=3): close the collider.
	sepset[0][3] = []int{}
	sepset[3][0] = []int{}

	if !ruleR4(m, sepset, p) {
		t.Fatal("R4 should fire")
	}
	if !(m[2][1] == Arrow && m[1][2] == Arrow) {
		t.Errorf("R4 (collider branch) should orient b ↔ c; got mark at b=%v, mark at c=%v", m[2][1], m[1][2])
	}
}

// TestRuleR4Directed exercises R4 when the middle node IS in the separating set:
// the edge is oriented b → c instead.
func TestRuleR4Directed(t *testing.T) {
	const p = 4
	m := newMarks(p)
	pagEdge(m, 0, 1, Circle, Arrow) // θ o→ a
	pagEdge(m, 1, 2, Arrow, Arrow)  // a ↔ b
	pagEdge(m, 1, 3, Tail, Arrow)   // a → c
	pagEdge(m, 2, 3, Circle, Arrow) // b o→ c

	sepset := make([][][]int, p)
	for i := range sepset {
		sepset[i] = make([][]int, p)
	}
	// b (=2) IS in Sepset(θ=0, c=3): orient b → c.
	sepset[0][3] = []int{2}
	sepset[3][0] = []int{2}

	if !ruleR4(m, sepset, p) {
		t.Fatal("R4 should fire")
	}
	if !isDirected(m, 2, 3) {
		t.Errorf("R4 (directed branch) should orient b → c; got mark at c=%v, mark at b=%v", m[2][3], m[3][2])
	}
}

func TestRuleR8(t *testing.T) {
	// a → b → c, a o→ c  =>  a → c.  (a=0,b=1,c=2)
	m := newMarks(3)
	pagEdge(m, 0, 1, Tail, Arrow)   // a → b
	pagEdge(m, 1, 2, Tail, Arrow)   // b → c
	pagEdge(m, 0, 2, Circle, Arrow) // a o→ c
	if !ruleR8(m, 3) {
		t.Fatal("R8 should fire")
	}
	if !isDirected(m, 0, 2) {
		t.Errorf("R8 should orient a → c; got mark at c=%v, mark at a=%v", m[0][2], m[2][0])
	}
}

func TestRuleR9(t *testing.T) {
	// a o→ c, and an uncovered p.d. path a-b-e-c whose second node b is not
	// adjacent to c  =>  a → c.  (a=0,c=3, path 0-1-2-3)
	const p = 4
	m := newMarks(p)
	pagEdge(m, 0, 3, Circle, Arrow)  // a o→ c
	pagEdge(m, 0, 1, Circle, Circle) // a o—o v1
	pagEdge(m, 1, 2, Circle, Circle) // v1 o—o v2
	pagEdge(m, 2, 3, Circle, Circle) // v2 o—o c
	if !ruleR9(m, p) {
		t.Fatal("R9 should fire")
	}
	if !isDirected(m, 0, 3) {
		t.Errorf("R9 should orient a → c; got mark at c=%v, mark at a=%v", m[0][3], m[3][0])
	}
}

func TestRuleR10(t *testing.T) {
	// a o→ c, parents b→c and d→c, two uncovered p.d. paths a-b and a-d whose
	// first nodes (b, d) are distinct and non-adjacent  =>  a → c.
	// (a=0, c=1, b=2, d=3)
	const p = 4
	m := newMarks(p)
	pagEdge(m, 0, 1, Circle, Arrow)  // a o→ c
	pagEdge(m, 2, 1, Tail, Arrow)    // b → c
	pagEdge(m, 3, 1, Tail, Arrow)    // d → c
	pagEdge(m, 0, 2, Circle, Circle) // a o—o b
	pagEdge(m, 0, 3, Circle, Circle) // a o—o d
	if !ruleR10(m, p) {
		t.Fatal("R10 should fire")
	}
	if !isDirected(m, 0, 1) {
		t.Errorf("R10 should orient a → c; got mark at c=%v, mark at a=%v", m[0][1], m[1][0])
	}
}

// --- end-to-end ground truth on data with a latent confounder ------------

// latentDoubleCollider builds the canonical FCI example. Over the observed
// {A,B,C,D} plus a HIDDEN common cause L of B and C, the true DAG is
// A→B, L→B, L→C, D→C. FCI must recover A o→ B ↔ C ←o D: the bidirected B↔C is the
// latent common cause that a causal-sufficiency method (PCStable) cannot express.
// It returns the four OBSERVED variables (L dropped).
func latentDoubleCollider(seed int64, n int) [][]float64 {
	const p = 5 // A=0, B=1, C=2, D=3, L=4
	w := zeroW(p)
	w[0][1] = 0.9 // A → B
	w[4][1] = 0.9 // L → B
	w[4][2] = 0.9 // L → C
	w[3][2] = 0.9 // D → C
	topo := []int{0, 3, 4, 1, 2}
	full := genSEM(seed, n, p, w, topo)
	return full[:4] // observed only; L is latent
}

func TestFCILatentConfounder(t *testing.T) {
	data := latentDoubleCollider(20260727, 12000)
	g, err := FCI(data, []string{"A", "B", "C", "D"}, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Skeleton: exactly A-B, B-C, C-D.
	wantAdj := map[[2]int]bool{{0, 1}: true, {1, 2}: true, {2, 3}: true}
	for i := 0; i < 4; i++ {
		for j := i + 1; j < 4; j++ {
			if g.Adjacent(i, j) != wantAdj[[2]int{i, j}] {
				t.Errorf("adjacency (%d,%d) = %v, want %v\n%s", i, j, g.Adjacent(i, j), wantAdj[[2]int{i, j}], g)
			}
		}
	}

	// The whole point: B ↔ C recovered as a latent common cause.
	if !g.Bidirected(1, 2) {
		t.Errorf("expected B ↔ C (bidirected, latent confounder); got mark at B=%v, mark at C=%v\n%s",
			g.MarkAt(2, 1), g.MarkAt(1, 2), g)
	}
	// A o→ B: arrowhead at B, circle at A.
	if !(g.MarkAt(0, 1) == Arrow && g.MarkAt(1, 0) == Circle) {
		t.Errorf("expected A o→ B; got mark at B=%v, mark at A=%v\n%s", g.MarkAt(0, 1), g.MarkAt(1, 0), g)
	}
	// D o→ C: arrowhead at C, circle at D.
	if !(g.MarkAt(3, 2) == Arrow && g.MarkAt(2, 3) == Circle) {
		t.Errorf("expected D o→ C; got mark at C=%v, mark at D=%v\n%s", g.MarkAt(3, 2), g.MarkAt(2, 3), g)
	}
}

// TestFCIPlainCollider: with NO latent confounder, a plain collider A→C←B (A,B
// marginally independent) comes back A o→ C ←o B — arrowheads into C, the far
// ends undetermined (FCI cannot add tails without further constraints).
func TestFCIPlainCollider(t *testing.T) {
	w := zeroW(3)
	w[0][2] = 0.8 // A → C
	w[1][2] = 0.8 // B → C
	data := genSEM(3, 6000, 3, w, []int{0, 1, 2})
	g, err := FCI(data, []string{"A", "B", "C"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if g.Adjacent(0, 1) {
		t.Errorf("A and B should be non-adjacent\n%s", g)
	}
	if !(g.MarkAt(0, 2) == Arrow && g.MarkAt(2, 0) == Circle) {
		t.Errorf("expected A o→ C; %s", g)
	}
	if !(g.MarkAt(1, 2) == Arrow && g.MarkAt(2, 1) == Circle) {
		t.Errorf("expected B o→ C; %s", g)
	}
}

// TestFCIChainAllCircles: a chain A→B→C is one equivalence class with the fork
// and the other chain, so — exactly like the CPDAG — nothing is oriented and the
// PAG is A o—o B o—o C.
func TestFCIChainAllCircles(t *testing.T) {
	w := zeroW(3)
	w[0][1] = 0.8
	w[1][2] = 0.8
	data := genSEM(1, 6000, 3, w, []int{0, 1, 2})
	g, err := FCI(data, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !g.Adjacent(0, 1) || !g.Adjacent(1, 2) || g.Adjacent(0, 2) {
		t.Errorf("expected skeleton A-B-C\n%s", g)
	}
	for _, e := range g.Edges() {
		if e.MarkA != Circle || e.MarkB != Circle {
			t.Errorf("chain should leave every endpoint a circle; got %s", g)
		}
	}
}

// --- determinism and permutation invariance ------------------------------

func TestFCIDeterministic(t *testing.T) {
	data := latentDoubleCollider(7, 8000)
	g1, err := FCI(data, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	g2, err := FCI(data, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if g1.String() != g2.String() {
		t.Errorf("non-deterministic output:\n%s\nvs\n%s", g1, g2)
	}
}

// TestFCISkeletonPermutationInvariant: the FCI skeleton, like the PC-stable one,
// must not depend on the order in which variables are presented.
func TestFCISkeletonPermutationInvariant(t *testing.T) {
	const p = 4
	data := latentDoubleCollider(11, 9000)

	base, err := FCI(data, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	perm := []int{2, 0, 3, 1} // perm[newIndex] = oldIndex
	permData := make([][]float64, p)
	for newIdx, oldIdx := range perm {
		permData[newIdx] = data[oldIdx]
	}
	permG, err := FCI(permData, nil, nil)
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

// --- input validation ----------------------------------------------------

func TestFCIErrors(t *testing.T) {
	good := genSEM(1, 100, 3, func() [][]float64 {
		w := zeroW(3)
		w[0][1] = 0.8
		w[1][2] = 0.8
		return w
	}(), []int{0, 1, 2})

	ragged := [][]float64{{1, 2, 3, 4}, {1, 2, 3}}
	withNaN := [][]float64{{1, 2, 3, 4}, {1, 2, math.NaN(), 4}}

	tests := []struct {
		name  string
		data  [][]float64
		names []string
		opts  *FCIOptions
		want  error
	}{
		{"one variable", [][]float64{{1, 2, 3, 4}}, nil, nil, ErrTooFewVariables},
		{"ragged", ragged, nil, nil, ErrUnequalLengths},
		{"too few samples", [][]float64{{1, 2, 3}, {4, 5, 6}}, nil, nil, ErrTooFewSamples},
		{"nan", withNaN, nil, nil, ErrNonFinite},
		{"name count", good, []string{"only-one"}, nil, ErrNameCount},
		{"alpha negative", good, nil, &FCIOptions{Alpha: -0.1}, ErrInvalidAlpha},
		{"alpha too big", good, nil, &FCIOptions{Alpha: 1.5}, ErrInvalidAlpha},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := FCI(tc.data, tc.names, tc.opts)
			if !errors.Is(err, tc.want) {
				t.Errorf("got err=%v, want %v", err, tc.want)
			}
		})
	}
}

// TestFCICITestErrorPropagates ensures a CITest failure aborts the run.
func TestFCICITestErrorPropagates(t *testing.T) {
	data := genSEM(1, 100, 3, func() [][]float64 {
		w := zeroW(3)
		w[0][1] = 0.8
		w[1][2] = 0.8
		return w
	}(), []int{0, 1, 2})
	sentinel := errors.New("boom")
	failing := func(_ [][]float64, _, _ int, _ []int) (float64, error) { return 0, sentinel }
	_, err := FCI(data, nil, &FCIOptions{CITest: failing})
	if !errors.Is(err, sentinel) {
		t.Errorf("got err=%v, want sentinel from CITest", err)
	}
}
