package causa

import (
	"errors"
	"testing"
)

func TestNewDiagramValid(t *testing.T) {
	// Front-door: X → M → Y, X ↔ Y.  X=0, M=1, Y=2.
	g, err := NewDiagram([]string{"X", "M", "Y"},
		[][2]int{{0, 1}, {1, 2}}, [][2]int{{0, 2}})
	if err != nil {
		t.Fatal(err)
	}
	if g.Order() != 3 {
		t.Fatalf("order = %d, want 3", g.Order())
	}
	if !g.HasDirected(0, 1) || !g.HasDirected(1, 2) {
		t.Error("missing directed edges")
	}
	if !g.HasBidirected(0, 2) || !g.HasBidirected(2, 0) {
		t.Error("bidirected edge must be symmetric")
	}
	if g.HasDirected(1, 0) {
		t.Error("spurious reverse directed edge")
	}
}

func TestNewDiagramInferN(t *testing.T) {
	// nil names: n inferred as max index + 1.
	g, err := NewDiagram(nil, [][2]int{{0, 2}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if g.Order() != 3 {
		t.Errorf("inferred order = %d, want 3", g.Order())
	}
	if got := g.Nodes(); got[2] != "V2" {
		t.Errorf("default name = %q, want V2", got[2])
	}
}

func TestNewDiagramErrors(t *testing.T) {
	tests := []struct {
		name       string
		names      []string
		directed   [][2]int
		bidirected [][2]int
		want       error
	}{
		{"cycle", []string{"A", "B"}, [][2]int{{0, 1}, {1, 0}}, nil, ErrCyclic},
		{"self loop directed", []string{"A", "B"}, [][2]int{{0, 0}}, nil, ErrSelfLoop},
		{"self loop bidirected", []string{"A", "B"}, nil, [][2]int{{1, 1}}, ErrSelfLoop},
		{"out of range", []string{"A", "B"}, [][2]int{{0, 2}}, nil, ErrEdgeOutOfRange},
		{"name count", []string{"A"}, [][2]int{{0, 1}}, nil, ErrEdgeOutOfRange},
		{"empty", nil, nil, nil, ErrTooFewVariables},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewDiagram(tc.names, tc.directed, tc.bidirected)
			if !errors.Is(err, tc.want) {
				t.Errorf("got %v, want %v", err, tc.want)
			}
		})
	}
}

func TestAncestors(t *testing.T) {
	// Chain 0 → 1 → 2 → 3, plus 4 isolated.
	g, err := NewDiagram([]string{"A", "B", "C", "D", "E"},
		[][2]int{{0, 1}, {1, 2}, {2, 3}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	n := g.Order()
	full := maskFromSlice(n, []int{0, 1, 2, 3, 4})

	// An({C}) = {A,B,C}.
	anc := g.ancestors(full, maskFromSlice(n, []int{2}), nil)
	if got := maskToSorted(anc); !sortedEqual(got, []int{0, 1, 2}) {
		t.Errorf("An(C) = %v, want [0 1 2]", got)
	}

	// With edges into B (=1) cut, An({C}) in the do-graph = {B,C} only.
	cut := maskFromSlice(n, []int{1})
	anc = g.ancestors(full, maskFromSlice(n, []int{2}), cut)
	if got := maskToSorted(anc); !sortedEqual(got, []int{1, 2}) {
		t.Errorf("An(C) with edges into B cut = %v, want [1 2]", got)
	}
}

func TestCComponents(t *testing.T) {
	// Bidirected chain 0 ↔ 1 ↔ 2, and 3 alone. C-components: {0,1,2} and {3}.
	g, err := NewDiagram([]string{"A", "B", "C", "D"}, nil,
		[][2]int{{0, 1}, {1, 2}})
	if err != nil {
		t.Fatal(err)
	}
	n := g.Order()
	comps := g.cComponents(maskFromSlice(n, []int{0, 1, 2, 3}))
	if len(comps) != 2 {
		t.Fatalf("got %d components, want 2", len(comps))
	}
	if got := maskToSorted(comps[0]); !sortedEqual(got, []int{0, 1, 2}) {
		t.Errorf("comp[0] = %v, want [0 1 2]", got)
	}
	if got := maskToSorted(comps[1]); !sortedEqual(got, []int{3}) {
		t.Errorf("comp[1] = %v, want [3]", got)
	}

	// Restricting the subset to {0,2,3} breaks the 0↔1↔2 chain (1 is gone):
	// now 0 and 2 are separate singletons.
	comps = g.cComponents(maskFromSlice(n, []int{0, 2, 3}))
	if len(comps) != 3 {
		t.Errorf("on subset {0,2,3} got %d components, want 3 singletons", len(comps))
	}
}

func TestTopoSub(t *testing.T) {
	// 0 → 2, 1 → 2, 2 → 3.  A valid order must place 0,1 before 2 before 3;
	// determinism breaks ties to the lowest index, so the order is [0 1 2 3].
	g, err := NewDiagram(nil, [][2]int{{0, 2}, {1, 2}, {2, 3}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	n := g.Order()
	order, ok := topoSub(g.dir, maskFromSlice(n, []int{0, 1, 2, 3}))
	if !ok {
		t.Fatal("unexpected cycle")
	}
	want := []int{0, 1, 2, 3}
	for i := range want {
		if order[i] != want[i] {
			t.Errorf("topo order = %v, want %v", order, want)
			break
		}
	}
	// Restricting to {1,2,3} still yields a valid order [1 2 3].
	order, ok = topoSub(g.dir, maskFromSlice(n, []int{1, 2, 3}))
	if !ok || len(order) != 3 || order[0] != 1 {
		t.Errorf("sub-order = %v (ok=%v), want [1 2 3]", order, ok)
	}
}
