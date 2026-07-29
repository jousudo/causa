package causa

import (
	"errors"
	"testing"
)

// Selection-bias orientation rules (Zhang 2008, R5–R7) — the only rules that
// produce undirected (—, tail–tail) edges, and the sole content of FCI's opt-in
// SelectionBias mode. The per-rule tests drive the rule functions directly on
// hand-built mark matrices (the pattern of TestRuleR1..R4/R8..R10 in fci_test.go);
// the end-to-end test locks the byte-identical guarantee on no-selection data.

// --- R5: uncovered circle path => undirected cycle ------------------------

// TestRuleR5FiresOn4Cycle: an uncovered 4-cycle of o—o edges a—g—t—b—a (both
// diagonals a-t and g-b absent) is R5's minimal witness. R5 orients the edge
// a o—o b AND every edge of the connecting path a-g-t-b as undirected, so all four
// cycle edges come back as —.
func TestRuleR5FiresOn4Cycle(t *testing.T) {
	// a=0, g=1, t=2, b=3.
	m := newMarks(4)
	pagEdge(m, 0, 1, Circle, Circle) // a o—o g
	pagEdge(m, 1, 2, Circle, Circle) // g o—o t
	pagEdge(m, 2, 3, Circle, Circle) // t o—o b
	pagEdge(m, 0, 3, Circle, Circle) // a o—o b (the edge R5 keys on)
	// diagonals a-t (0-2) and g-b (1-3) absent => the cycle is uncovered.

	if !ruleR5(m, 4) {
		t.Fatal("R5 should fire on the uncovered 4-cycle")
	}
	for _, e := range [][2]int{{0, 1}, {1, 2}, {2, 3}, {0, 3}} {
		if !isUndirected(m, e[0], e[1]) {
			t.Errorf("R5 should orient edge %d—%d undirected; got mark at %d=%v, at %d=%v",
				e[0], e[1], e[0], m[e[1]][e[0]], e[1], m[e[0]][e[1]])
		}
	}
}

// TestRuleR5SkipsTriangle: a triangle of o—o edges has no uncovered circle path
// (the only alternate route between any two nodes passes through the third, which
// is adjacent to both endpoints), so R5 must not fire and must not introduce any
// tail.
func TestRuleR5SkipsTriangle(t *testing.T) {
	m := newMarks(3)
	pagEdge(m, 0, 1, Circle, Circle)
	pagEdge(m, 1, 2, Circle, Circle)
	pagEdge(m, 0, 2, Circle, Circle)
	if ruleR5(m, 3) {
		t.Fatal("R5 must not fire on a covered triangle")
	}
	for i := 0; i < 3; i++ {
		for j := i + 1; j < 3; j++ {
			if !isCircleBoth(m, i, j) {
				t.Errorf("R5 must leave the triangle all-circle; edge %d-%d changed", i, j)
			}
		}
	}
}

// --- R6: a — b o—* c => tail at b -----------------------------------------

// TestRuleR6: an undirected a — b forces a tail at b on b o—o c (an arrowhead at b
// would need a into b, contradicting the tail).
func TestRuleR6(t *testing.T) {
	m := newMarks(3)
	pagEdge(m, 0, 1, Tail, Tail)     // a — b (undirected)
	pagEdge(m, 1, 2, Circle, Circle) // b o—o c
	if !ruleR6(m, 3) {
		t.Fatal("R6 should fire")
	}
	if !isTail(m, 1, 2) { // tail at b on edge b-c
		t.Errorf("R6 should place a tail at b on b-c; got mark at b=%v", m[2][1])
	}
	if !isCircle(m, 2, 1) {
		t.Error("R6 must not touch the c end of b-c")
	}
}

// TestRuleR6NeedsUndirected: with a → b (not undirected) R6 has no antecedent.
func TestRuleR6NeedsUndirected(t *testing.T) {
	m := newMarks(3)
	pagEdge(m, 0, 1, Tail, Arrow)    // a → b
	pagEdge(m, 1, 2, Circle, Circle) // b o—o c
	if ruleR6(m, 3) {
		t.Fatal("R6 must not fire without an undirected a — b")
	}
}

// --- R7: a —o b o—* c, a,c non-adjacent => tail at b ----------------------

func TestRuleR7(t *testing.T) {
	m := newMarks(3)
	pagEdge(m, 0, 1, Tail, Circle)   // a —o b (tail at a, circle at b)
	pagEdge(m, 1, 2, Circle, Circle) // b o—o c
	// a (0) and c (2) non-adjacent.
	if !ruleR7(m, 3) {
		t.Fatal("R7 should fire")
	}
	if !isTail(m, 1, 2) { // tail at b on edge b-c
		t.Errorf("R7 should place a tail at b on b-c; got mark at b=%v", m[2][1])
	}
}

// TestRuleR7NeedsNonAdjacent: if a and c are adjacent the triple is shielded and
// R7 must not fire (this is exactly what distinguishes R7 from R6).
func TestRuleR7NeedsNonAdjacent(t *testing.T) {
	m := newMarks(3)
	pagEdge(m, 0, 1, Tail, Circle)   // a —o b
	pagEdge(m, 1, 2, Circle, Circle) // b o—o c
	pagEdge(m, 0, 2, Circle, Circle) // a o—o c  => a,c adjacent
	if ruleR7(m, 3) {
		t.Fatal("R7 must not fire when a and c are adjacent")
	}
}

// --- end-to-end: byte-identical on no-selection data ----------------------

// TestFCISelectionByteIdenticalNoSelection: on data with no selection bias the
// opt-in SelectionBias mode must return the IDENTICAL PAG as the default — R5–R7
// have nothing to fire on. Uses the latent-confounder fixture (A o→ B ↔ C ←o D),
// the same generator the default-mode tests use.
func TestFCISelectionByteIdenticalNoSelection(t *testing.T) {
	data := latentDoubleCollider(20260727, 12000)
	base, err := FCI(data, []string{"A", "B", "C", "D"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	sel, err := FCI(data, []string{"A", "B", "C", "D"}, &FCIOptions{SelectionBias: true})
	if err != nil {
		t.Fatal(err)
	}
	p := base.Order()
	for i := 0; i < p; i++ {
		for j := 0; j < p; j++ {
			if base.MarkAt(i, j) != sel.MarkAt(i, j) {
				t.Fatalf("SelectionBias changed the PAG on no-selection data at (%d,%d): base=%v sel=%v\nbase:\n%s\nsel:\n%s",
					i, j, base.MarkAt(i, j), sel.MarkAt(i, j), base, sel)
			}
		}
	}
	// And no undirected edge was produced.
	for i := 0; i < p; i++ {
		for j := i + 1; j < p; j++ {
			if sel.MarkAt(i, j) == Tail && sel.MarkAt(j, i) == Tail {
				t.Errorf("SelectionBias produced an undirected edge %d—%d on no-selection data", i, j)
			}
		}
	}
}

// --- Part B: IDP/CIDP refuse a selection PAG ------------------------------

// TestIdentifyPAGRefusesSelectionBias: a PAG carrying an undirected (—) edge is
// outside the IDP/CIDP no-selection-bias scope; every identify entry point must
// return ErrSelectionBiasUnsupported instead of a silently wrong estimand.
func TestIdentifyPAGRefusesSelectionBias(t *testing.T) {
	// A — B (undirected, selection), B → Y. Query P(Y | do(A)). Nodes A=0,B=1,Y=2.
	g, err := NewPAG([]string{"A", "B", "Y"}, []PAGEdge{
		{A: 0, B: 1, MarkA: Tail, MarkB: Tail},  // A — B
		{A: 1, B: 2, MarkA: Tail, MarkB: Arrow}, // B → Y
	})
	if err != nil {
		t.Fatal(err)
	}
	y, x, z := []int{2}, []int{0}, []int{1}

	if _, err := IdentifyPAG(g, y, x); !errors.Is(err, ErrSelectionBiasUnsupported) {
		t.Errorf("IdentifyPAG: got err=%v, want ErrSelectionBiasUnsupported", err)
	}
	if _, err := g.Identify(y, x); !errors.Is(err, ErrSelectionBiasUnsupported) {
		t.Errorf("(*PAG).Identify: got err=%v, want ErrSelectionBiasUnsupported", err)
	}
	if _, err := IdentifyConditionalPAG(g, y, x, z); !errors.Is(err, ErrSelectionBiasUnsupported) {
		t.Errorf("IdentifyConditionalPAG: got err=%v, want ErrSelectionBiasUnsupported", err)
	}
	if _, err := g.IdentifyConditional(y, x, z); !errors.Is(err, ErrSelectionBiasUnsupported) {
		t.Errorf("(*PAG).IdentifyConditional: got err=%v, want ErrSelectionBiasUnsupported", err)
	}
}

// TestIdentifyPAGNoSelectionUnaffected: a PAG with no undirected edge is not
// refused — the guard is inert on in-scope inputs. On the directed chain A→B→Y the
// effect P(Y|do(B)) is identifiable (the reference-suite "chain_directed" case).
func TestIdentifyPAGNoSelectionUnaffected(t *testing.T) {
	g, err := NewPAG([]string{"A", "B", "Y"}, []PAGEdge{
		{A: 0, B: 1, MarkA: Tail, MarkB: Arrow}, // A → B
		{A: 1, B: 2, MarkA: Tail, MarkB: Arrow}, // B → Y
	})
	if err != nil {
		t.Fatal(err)
	}
	res, err := IdentifyPAG(g, []int{2}, []int{1})
	if err != nil {
		t.Fatalf("unexpected error on a no-selection PAG: %v", err)
	}
	if !res.Identifiable {
		t.Error("A→B→Y: P(Y|do(B)) should be identifiable")
	}
}
