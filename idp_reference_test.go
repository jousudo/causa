package causa

import (
	"fmt"
	"testing"
)

// TestIDPAgainstReference locks the identifiability decision on a battery of PAGs
// whose verdicts were confirmed, case for case, against the authors' reference
// implementation (PAGId::IDP in R) driven on the byte-identical adjacency matrices
// — pcalg's amat.pag encoding (1=circle, 2=arrowhead, 3=tail) coincides with this
// package's Mark values and mark[i][j] convention, so the fixtures transfer
// verbatim. Every case below matched the reference.
func TestIDPAgainstReference(t *testing.T) {
	type edge struct {
		i, j     int
		atI, atJ Mark
	}
	type tc struct {
		name  string
		n     int
		edges []edge
		y, x  []int
		want  bool
	}
	cases := []tc{
		// The minimal non-identifiable effect: the circle admits both X→Y and X↔Y.
		{"circle_arrow_XoY", 2, []edge{{0, 1, Circle, Arrow}}, []int{1}, []int{0}, false},
		// X↔Y: do(X) cannot affect Y across the class, so P(Y|do(X))=P(Y).
		{"bidirected_XY", 2, []edge{{0, 1, Arrow, Arrow}}, []int{1}, []int{0}, true},
		// Wholly undetermined edge: not identifiable.
		{"oo_XY", 2, []edge{{0, 1, Circle, Circle}}, []int{1}, []int{0}, false},
		// Z↔X, X→Y with X→Y visible: identifiable despite X being confounded.
		{"directed_visible", 3, []edge{{0, 1, Arrow, Arrow}, {1, 2, Tail, Arrow}}, []int{2}, []int{1}, true},
		// Z→X→Y fully directed chain.
		{"chain_directed", 3, []edge{{0, 1, Tail, Arrow}, {1, 2, Tail, Arrow}}, []int{2}, []int{1}, true},
		// X→M, M→Y, X↔Y (this exact mark pattern is non-identifiable as a class).
		{"xm_my_xbiy", 3, []edge{{0, 1, Tail, Arrow}, {1, 2, Tail, Arrow}, {0, 2, Arrow, Arrow}}, []int{2}, []int{0}, false},
		// Z→X, X↔Y.
		{"zx_xbiy", 3, []edge{{0, 1, Tail, Arrow}, {1, 2, Arrow, Arrow}}, []int{2}, []int{1}, true},
		// Four-node mixed graph with a circle, directed chain and a latent arc.
		{"mixed4", 4, []edge{{0, 1, Circle, Arrow}, {1, 2, Tail, Arrow}, {2, 3, Tail, Arrow}, {0, 3, Arrow, Arrow}}, []int{3}, []int{2}, true},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := newMarks(c.n)
			names := make([]string, c.n)
			for i := range names {
				names[i] = fmt.Sprintf("V%d", i)
			}
			for _, e := range c.edges {
				pagEdge(m, e.i, e.j, e.atI, e.atJ)
			}
			r, err := mkPAG(names, m).Identify(c.y, c.x)
			if err != nil {
				t.Fatal(err)
			}
			if r.Identifiable != c.want {
				t.Errorf("Identifiable = %v, want %v (reference PAGId::IDP); estimand=%s",
					r.Identifiable, c.want, r)
			}
		})
	}
}
