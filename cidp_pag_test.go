package causa

import (
	"fmt"
	"reflect"
	"testing"
)

// --- primitives: graph manipulations -----------------------------------------

func TestXUpperRemovesEdgesInto(t *testing.T) {
	// Z → X: xUpper({X}) removes the arrowhead-into-X edge entirely.
	m := newMarks(2)
	pagEdge(m, 0, 1, Tail, Arrow) // Z → X
	up := xUpper(m, 2, maskOf(2, 1))
	if fciAdjacent(up, 0, 1) {
		t.Error("xUpper({X}) should have removed the edge into X")
	}
	// The original is untouched (xUpper copies).
	if !fciAdjacent(m, 0, 1) {
		t.Error("xUpper must not mutate its input")
	}
}

func TestXLowerRemovesVisibleOut(t *testing.T) {
	// W *→ Z → X: Z → X is visible (W not adjacent to X), so xLower({Z}) drops it.
	m := newMarks(3)               // W=0, Z=1, X=2
	pagEdge(m, 0, 1, Arrow, Arrow) // W ↔ Z  (arrowhead into Z)
	pagEdge(m, 1, 2, Tail, Arrow)  // Z → X
	if !visibleEdgePAG(m, 3, 1, 2) {
		t.Fatal("precondition: Z→X visible")
	}
	low := xLower(m, 3, maskOf(3, 1))
	if fciAdjacent(low, 1, 2) {
		t.Error("xLower({Z}) should remove the visible edge Z→X")
	}
	if !fciAdjacent(low, 0, 1) {
		t.Error("xLower must leave the edge into Z intact")
	}
}

// --- primitive: definite-status m-separation ---------------------------------

func TestPagMSeparationCollider(t *testing.T) {
	// A → C ← B: A and B are m-separated given ∅ (collider blocks) but m-connected
	// given C (collider opened).
	m := newMarks(3) // A=0, C=1, B=2
	pagEdge(m, 0, 1, Tail, Arrow)
	pagEdge(m, 2, 1, Tail, Arrow)
	a, b := maskOf(3, 0), maskOf(3, 2)
	if !pagMSeparated(m, 3, a, b, maskOf(3)) {
		t.Error("A ⫫ B | ∅ should hold (unopened collider)")
	}
	if pagMSeparated(m, 3, a, b, maskOf(3, 1)) {
		t.Error("A ⫫ B | C should FAIL (collider opened)")
	}
}

func TestPagMSeparationChain(t *testing.T) {
	// A → B → C: A and C are m-connected given ∅, separated given B.
	m := newMarks(3)
	pagEdge(m, 0, 1, Tail, Arrow)
	pagEdge(m, 1, 2, Tail, Arrow)
	a, c := maskOf(3, 0), maskOf(3, 2)
	if pagMSeparated(m, 3, a, c, maskOf(3)) {
		t.Error("A ⫫ C | ∅ should FAIL (open chain)")
	}
	if !pagMSeparated(m, 3, a, c, maskOf(3, 1)) {
		t.Error("A ⫫ C | B should hold (chain blocked)")
	}
}

func TestDirectedDescendants(t *testing.T) {
	m := newMarks(4) // A→B→C, D↔C
	pagEdge(m, 0, 1, Tail, Arrow)
	pagEdge(m, 1, 2, Tail, Arrow)
	pagEdge(m, 2, 3, Arrow, Arrow) // C ↔ D (not directed)
	de := directedDescendants(m, 4, 0)
	if !reflect.DeepEqual(maskToSorted(de), []int{0, 1, 2}) {
		t.Errorf("de(A) = %v, want {0,1,2}", maskToSorted(de))
	}
}

// --- z empty reduces to IdentifyPAG ------------------------------------------

func TestCIDPEmptyContextReducesToIDP(t *testing.T) {
	m := newMarks(3)
	pagEdge(m, 0, 1, Arrow, Arrow) // Z ↔ X
	pagEdge(m, 1, 2, Tail, Arrow)  // X → Y
	g := mkPAG([]string{"Z", "X", "Y"}, m)
	rc, _ := g.IdentifyConditional([]int{2}, []int{1}, nil)
	ri, _ := g.Identify([]int{2}, []int{1})
	if rc.Identifiable != ri.Identifiable || rc.String() != ri.String() {
		t.Errorf("CIDP with empty z must equal IDP: cidp=%v/%q idp=%v/%q",
			rc.Identifiable, rc, ri.Identifiable, ri)
	}
}

func TestCIDPQueryValidation(t *testing.T) {
	m := newMarks(3)
	pagEdge(m, 0, 1, Circle, Arrow)
	g := mkPAG([]string{"X", "Y", "Z"}, m)
	if _, err := g.IdentifyConditional(nil, []int{0}, []int{2}); err != ErrEmptyOutcome {
		t.Errorf("empty Y: got %v, want ErrEmptyOutcome", err)
	}
	if _, err := g.IdentifyConditional([]int{1}, []int{0}, []int{0}); err != ErrOverlappingQuery {
		t.Errorf("x∩z overlap: got %v, want ErrOverlappingQuery", err)
	}
}

// --- reference-differential battery ------------------------------------------

// TestCIDPAgainstReference locks the CIDP identifiability decision on a battery of
// conditional queries whose verdicts were confirmed case-for-case against the
// authors' reference PAGId::CIDP in R on byte-identical adjacency matrices.
func TestCIDPAgainstReference(t *testing.T) {
	type tc struct {
		name    string
		n       int
		edges   [][4]int // {i, j, markAtI, markAtJ}; marks 1=circle,2=arrow,3=tail
		y, x, z []int
		want    bool
	}
	cases := []tc{
		{"zx_xy_zy_directed", 3, [][4]int{{2, 0, 3, 2}, {0, 1, 3, 2}, {2, 1, 3, 2}}, []int{1}, []int{0}, []int{2}, false},
		{"xy_zy_indep_ctx", 3, [][4]int{{0, 1, 3, 2}, {2, 1, 3, 2}}, []int{1}, []int{0}, []int{2}, false},
		{"zoX_xy", 3, [][4]int{{2, 0, 1, 2}, {0, 1, 3, 2}}, []int{1}, []int{0}, []int{2}, true},
		{"xoY_wY_ctx", 3, [][4]int{{0, 1, 1, 2}, {2, 1, 3, 2}}, []int{1}, []int{0}, []int{2}, false},
		{"frontdoorish_ctx", 4, [][4]int{{3, 0, 3, 2}, {0, 1, 3, 2}, {1, 2, 3, 2}, {0, 2, 2, 2}}, []int{2}, []int{1}, []int{0}, false},
		{"bidirected_ctx", 3, [][4]int{{0, 1, 2, 2}, {2, 1, 3, 2}}, []int{1}, []int{0}, []int{2}, true},
		{"chain4_ctx", 4, [][4]int{{0, 1, 3, 2}, {1, 2, 3, 2}, {2, 3, 3, 2}}, []int{3}, []int{2}, []int{1}, true},
		{"zx_xbiy_zy_ctx", 3, [][4]int{{2, 0, 3, 2}, {1, 2, 2, 2}, {2, 1, 3, 2}}, []int{1}, []int{0}, []int{2}, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := newMarks(c.n)
			names := make([]string, c.n)
			for i := range names {
				names[i] = fmt.Sprintf("V%d", i)
			}
			for _, e := range c.edges {
				pagEdge(m, e[0], e[1], Mark(e[2]), Mark(e[3]))
			}
			r, err := mkPAG(names, m).IdentifyConditional(c.y, c.x, c.z)
			if err != nil {
				t.Fatal(err)
			}
			if r.Identifiable != c.want {
				t.Errorf("Identifiable = %v, want %v (reference PAGId::CIDP)", r.Identifiable, c.want)
			}
		})
	}
}
