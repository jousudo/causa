package causa

// IdentifyConditional runs the Shpitser–Pearl IDC algorithm and decides whether a
// CONDITIONAL interventional distribution P(y | do(x), z) — the effect of setting
// x on y WITHIN the subpopulation where the (observed) variables z take a value —
// is identifiable from the observational distribution in the causal diagram g,
// returning its estimand when it is.
//
// y, x and z are pairwise-disjoint variable-index sets: y (outcome) must be
// non-empty, x (intervention) and z (conditioning context) may each be empty. It
// generalizes Identify: with z empty the query reduces to P(y | do(x)).
//
// Algorithm. It applies do-calculus Rule 2 to move each conditioning variable
// that behaves like an intervention out of z and into x — testing, for a
// candidate z′, whether (Y ⫫ Z′ | X, Z∖Z′) by m-separation in the manipulated
// graph with the edges into X and the edges out of Z′ removed — until none
// remains movable. The residual conditional is then P(y | do(x), z) =
// P_x(y, z) / P_x(z), where P_x(y, z) is identified by the ID algorithm and the
// denominator by summing out y. If that joint is not identifiable, neither is the
// conditional effect (Identifiable is false, with a hedge).
//
// The result is symbolic (an Expr over P(V)); evaluate it on discrete data with
// Estimand.Evaluate as for Identify. Errors mirror Identify, over all three sets:
// ErrEmptyOutcome, ErrEdgeOutOfRange, ErrDuplicateQueryVar, ErrOverlappingQuery.
//
// Reference: Shpitser & Pearl, "Identification of Conditional Interventional
// Distributions", UAI 2006.
func (g *Diagram) IdentifyConditional(y, x, z []int) (*IDResult, error) {
	return IdentifyConditional(g, y, x, z)
}

// IdentifyConditional is the package-level form; see (*Diagram).IdentifyConditional.
func IdentifyConditional(g *Diagram, y, x, z []int) (*IDResult, error) {
	n := g.Order()
	if err := validateConditionalQuery(n, y, x, z); err != nil {
		return nil, err
	}
	ymask := maskFromSlice(n, y)
	xmask := maskFromSlice(n, x)
	zmask := maskFromSlice(n, z)
	full := make([]bool, n)
	allVars := make([]int, n)
	for i := range full {
		full[i] = true
		allVars[i] = i
	}

	est, hd := g.idc(ymask, xmask, zmask, jointExpr(allVars), full)
	if hd != nil {
		return &IDResult{names: g.names, Identifiable: false, hedgeR: hd.r, hedgeS: hd.s}, nil
	}
	return &IDResult{names: g.names, Identifiable: true, Estimand: est}, nil
}

func validateConditionalQuery(n int, y, x, z []int) error {
	if len(y) == 0 {
		return ErrEmptyOutcome
	}
	seen := make([]int, n) // 0 unseen, else which set (1=y,2=x,3=z)
	for tag, set := range [][]int{y, x, z} {
		for _, v := range set {
			if v < 0 || v >= n {
				return ErrEdgeOutOfRange
			}
			switch {
			case seen[v] == tag+1:
				return ErrDuplicateQueryVar
			case seen[v] != 0:
				return ErrOverlappingQuery
			}
			seen[v] = tag + 1
		}
	}
	return nil
}

// idc is the IDC recursion: it moves conditioning variables into the intervention
// set while do-calculus Rule 2 licenses it, then forms the residual conditional
// from an ID-identified joint.
func (g *Diagram) idc(y, x, z []bool, P *Expr, full []bool) (*Expr, *hedge) {
	// Rule 2: while some z′ ∈ Z satisfies (Y ⫫ Z′ | X, Z∖Z′) in the graph with
	// edges into X and out of Z′ removed, move z′ from Z into X.
	for {
		moved := false
		for _, zp := range maskToSorted(z) {
			dir2, bi2 := g.manipulated(x, zp)
			b := make([]bool, len(full))
			b[zp] = true
			c := maskUnion(x, maskDiff(z, b)) // X ∪ (Z ∖ {z′})
			if mSeparated(dir2, bi2, len(full), y, b, c) {
				x = maskUnion(x, b)
				z = maskDiff(z, b)
				moved = true
				break
			}
		}
		if !moved {
			break
		}
	}

	// Residual: P(y | do(x), z) = P_x(y, z) / P_x(z).
	joint, hd := g.id(maskUnion(y, z), x, P, full)
	if hd != nil {
		return nil, hd
	}
	if maskEmpty(z) {
		return joint, nil // no conditioning left: this is P(y | do(x))
	}
	den := marginalExpr(maskToSorted(y), joint) // P_x(z) = Σ_y P_x(y, z)
	return ratioExpr(joint, den), nil
}

// manipulated returns the directed and bidirected adjacency of the graph
// G_{\overline{X}, \underline{z′}}: g with every edge INTO a vertex of x removed
// (directed heads and incident bidirected arcs) and every directed edge OUT OF z′
// removed. It realises the graph do-calculus Rule 2's independence test runs in.
func (g *Diagram) manipulated(x []bool, zp int) (dir, bi [][]bool) {
	n := g.Order()
	dir = make([][]bool, n)
	bi = make([][]bool, n)
	for i := 0; i < n; i++ {
		dir[i] = make([]bool, n)
		bi[i] = make([]bool, n)
		for j := 0; j < n; j++ {
			// directed i→j kept unless it points into X, or emanates from z′.
			if g.dir[i][j] && !x[j] && i != zp {
				dir[i][j] = true
			}
			// bidirected kept unless incident to X (a latent arrowhead into X).
			if g.bi[i][j] && !x[i] && !x[j] {
				bi[i][j] = true
			}
		}
	}
	return dir, bi
}

// --- m-separation for ADMGs ----------------------------------------------

// mSeparated reports whether the vertex sets a and b are m-separated given c in
// the ADMG with directed adjacency `dir` (dir[i][j] == i → j) and bidirected
// adjacency `bi` over n vertices. It uses the moralization criterion: a and b are
// m-separated by c iff, in the moral graph of the subgraph induced on the
// ancestors of a ∪ b ∪ c, every path from a to b passes through c.
func mSeparated(dir, bi [][]bool, n int, a, b, c []bool) bool {
	anc := directedAncestors(dir, maskUnion(maskUnion(a, b), c), n)

	// Build the moral (undirected) graph over the ancestral set.
	adj := make([][]bool, n)
	for i := range adj {
		adj[i] = make([]bool, n)
	}
	link := func(u, v int) {
		if u != v {
			adj[u][v] = true
			adj[v][u] = true
		}
	}
	for u := 0; u < n; u++ {
		if !anc[u] {
			continue
		}
		for v := 0; v < n; v++ {
			if anc[v] && (dir[u][v] || dir[v][u] || bi[u][v]) {
				link(u, v)
			}
		}
	}
	// Marry the "parents" of each vertex — the endpoints of arrowheads into it
	// (directed heads and bidirected arcs) — so colliders do not separate.
	for w := 0; w < n; w++ {
		if !anc[w] {
			continue
		}
		var parents []int
		for u := 0; u < n; u++ {
			if anc[u] && (dir[u][w] || bi[u][w]) {
				parents = append(parents, u)
			}
		}
		for i := 0; i < len(parents); i++ {
			for j := i + 1; j < len(parents); j++ {
				link(parents[i], parents[j])
			}
		}
	}

	// Separated iff no a-vertex reaches a b-vertex without passing through c.
	visited := make([]bool, n)
	var stack []int
	for v := 0; v < n; v++ {
		if a[v] && anc[v] && !c[v] {
			visited[v] = true
			stack = append(stack, v)
		}
	}
	for len(stack) > 0 {
		u := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if b[u] {
			return false // reached b: m-connected
		}
		for w := 0; w < n; w++ {
			if adj[u][w] && !visited[w] && !c[w] {
				visited[w] = true
				stack = append(stack, w)
			}
		}
	}
	return true
}

// directedAncestors returns the mask of ancestors of `targets` (inclusive) via
// the directed edges of `dir`.
func directedAncestors(dir [][]bool, targets []bool, n int) []bool {
	anc := make([]bool, n)
	var stack []int
	for v := 0; v < n; v++ {
		if targets[v] {
			anc[v] = true
			stack = append(stack, v)
		}
	}
	for len(stack) > 0 {
		v := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		for p := 0; p < n; p++ {
			if dir[p][v] && !anc[p] {
				anc[p] = true
				stack = append(stack, p)
			}
		}
	}
	return anc
}
