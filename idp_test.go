package causa

import (
	"reflect"
	"testing"
)

// mkPAG builds a PAG from names and a mark matrix for tests.
func mkPAG(names []string, m [][]Mark) *PAG { return &PAG{names: names, mark: m} }

func maskOf(n int, idx ...int) []bool { return maskFromSlice(n, idx) }

// --- primitive: visibleEdge --------------------------------------------------

func TestVisibleEdgeScenario1(t *testing.T) {
	// Z → X → Y with Z not adjacent to Y makes X → Y visible: the arrowhead Z *→ X
	// from a vertex not touching Y witnesses that X → Y is not confounded.
	m := newMarks(3) // Z=0, X=1, Y=2
	pagEdge(m, 0, 1, Tail, Arrow)
	pagEdge(m, 1, 2, Tail, Arrow)
	g := mkPAG([]string{"Z", "X", "Y"}, m)
	if !g.visibleEdge(1, 2) {
		t.Error("X→Y should be visible (Z *→ X, Z not adjacent to Y)")
	}
	// The lone root edge Z → X has no such witness and is invisible.
	if g.visibleEdge(0, 1) {
		t.Error("Z→X should be invisible (no vertex with an arrowhead into Z)")
	}
}

func TestVisibleEdgeScenario2(t *testing.T) {
	// Discriminating-path visibility: θ ↔ C ↔ X → Z with C → Z and θ not adjacent
	// to Z. The edge X → Z is visible only through the discriminating path.
	m := newMarks(4)               // θ=0, C=1, X=2, Z=3
	pagEdge(m, 0, 1, Arrow, Arrow) // θ ↔ C
	pagEdge(m, 1, 2, Arrow, Arrow) // C ↔ X
	pagEdge(m, 1, 3, Tail, Arrow)  // C → Z
	pagEdge(m, 2, 3, Tail, Arrow)  // X → Z
	g := mkPAG([]string{"T", "C", "X", "Z"}, m)
	if !g.visibleEdge(2, 3) {
		t.Error("X→Z should be visible through the discriminating path θ↔C↔X→Z")
	}
	// Remove θ (its edges) and the discriminating path is gone: X→Z invisible.
	m2 := newMarks(4)
	pagEdge(m2, 1, 2, Arrow, Arrow)
	pagEdge(m2, 1, 3, Tail, Arrow)
	pagEdge(m2, 2, 3, Tail, Arrow)
	g2 := mkPAG([]string{"T", "C", "X", "Z"}, m2)
	if g2.visibleEdge(2, 3) {
		t.Error("without the discriminating path X→Z must be invisible")
	}
}

// --- primitive: buckets ------------------------------------------------------

func TestPagBuckets(t *testing.T) {
	// A o-o B o-o C, D on its own: one bucket {A,B,C} and a singleton {D}.
	m := newMarks(4)
	pagEdge(m, 0, 1, Circle, Circle)
	pagEdge(m, 1, 2, Circle, Circle)
	g := mkPAG([]string{"A", "B", "C", "D"}, m)
	buckets := g.pagBuckets(maskOf(4, 0, 1, 2, 3))
	if len(buckets) != 2 {
		t.Fatalf("expected 2 buckets, got %d", len(buckets))
	}
	if !reflect.DeepEqual(maskToSorted(buckets[0]), []int{0, 1, 2}) {
		t.Errorf("first bucket = %v, want {0,1,2}", maskToSorted(buckets[0]))
	}
	if !reflect.DeepEqual(maskToSorted(buckets[1]), []int{3}) {
		t.Errorf("second bucket = %v, want {3}", maskToSorted(buckets[1]))
	}
}

// --- primitive: possible ancestors / descendants -----------------------------

func TestPagPossReach(t *testing.T) {
	// A → B o→ C, and D ↔ C. Possibly-directed reachability from A is {A,B,C}
	// (the ↔ to D is not directable, so D is not a possible descendant).
	m := newMarks(4)                // A=0,B=1,C=2,D=3
	pagEdge(m, 0, 1, Tail, Arrow)   // A → B
	pagEdge(m, 1, 2, Circle, Arrow) // B o→ C
	pagEdge(m, 2, 3, Arrow, Arrow)  // C ↔ D
	g := mkPAG([]string{"A", "B", "C", "D"}, m)
	full := maskOf(4, 0, 1, 2, 3)

	de := g.pagPossDescendants(full, maskOf(4, 0))
	if !reflect.DeepEqual(maskToSorted(de), []int{0, 1, 2}) {
		t.Errorf("possDe(A) = %v, want {0,1,2}", maskToSorted(de))
	}
	an := g.pagPossAncestors(full, maskOf(4, 2))
	if !reflect.DeepEqual(maskToSorted(an), []int{0, 1, 2}) {
		t.Errorf("possAn(C) = %v, want {0,1,2}", maskToSorted(an))
	}
}

// --- end-to-end: empty intervention -----------------------------------------

func TestIDPEmptyIntervention(t *testing.T) {
	// P(y | do(∅)) is the observational marginal — always identifiable.
	m := newMarks(2)
	pagEdge(m, 0, 1, Circle, Arrow) // X o→ Y
	g := mkPAG([]string{"X", "Y"}, m)
	r, err := g.Identify([]int{1}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !r.Identifiable {
		t.Fatal("P(Y) must be identifiable")
	}
	if got := r.String(); got != "Σ_{X}[P(X,Y)]" {
		t.Errorf("estimand = %q, want Σ_{X}[P(X,Y)]", got)
	}
}

// --- end-to-end: the minimal non-identifiable PAG ---------------------------

func TestIDPCircleArrowNonID(t *testing.T) {
	// X o→ Y is the minimal non-identifiable effect: the equivalence class holds
	// both X → Y (where do(x) matters) and X ↔ Y (where it does not), so no single
	// estimand is valid across the class.
	m := newMarks(2)
	pagEdge(m, 0, 1, Circle, Arrow)
	g := mkPAG([]string{"X", "Y"}, m)
	r, err := g.Identify([]int{1}, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	if r.Identifiable {
		t.Errorf("X o→ Y: P(Y|do(X)) must NOT be identifiable, got %s", r)
	}
}

// --- end-to-end: a directed visible edge is identifiable --------------------

func TestIDPVisibleDirectedEdge(t *testing.T) {
	// Z *→ X → Y with Z ↔ X (a latent Z-X confounder) and Z not adjacent to Y.
	// X → Y is visible, so P(Y|do(X)) is identifiable even though X itself is
	// confounded with Z.
	m := newMarks(3)               // Z=0, X=1, Y=2
	pagEdge(m, 0, 1, Arrow, Arrow) // Z ↔ X
	pagEdge(m, 1, 2, Tail, Arrow)  // X → Y
	g := mkPAG([]string{"Z", "X", "Y"}, m)
	if !g.visibleEdge(1, 2) {
		t.Fatal("precondition: X→Y should be visible")
	}
	r, err := g.Identify([]int{2}, []int{1})
	if err != nil {
		t.Fatal(err)
	}
	if !r.Identifiable {
		t.Errorf("visible X→Y: P(Y|do(X)) must be identifiable, got %s", r)
	}
	t.Logf("estimand: %s", r)
}

// --- query validation reuses the ID contract --------------------------------

func TestIDPQueryValidation(t *testing.T) {
	m := newMarks(2)
	pagEdge(m, 0, 1, Circle, Arrow)
	g := mkPAG([]string{"X", "Y"}, m)
	if _, err := g.Identify(nil, []int{0}); err != ErrEmptyOutcome {
		t.Errorf("empty Y: got %v, want ErrEmptyOutcome", err)
	}
	if _, err := g.Identify([]int{0}, []int{0}); err != ErrOverlappingQuery {
		t.Errorf("overlap: got %v, want ErrOverlappingQuery", err)
	}
	if _, err := g.Identify([]int{5}, nil); err != ErrEdgeOutOfRange {
		t.Errorf("out of range: got %v, want ErrEdgeOutOfRange", err)
	}
}
