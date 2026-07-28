package causa

// This file implements IDP: identification of marginal causal effects from a PAG
// (a Markov equivalence class of MAGs), as opposed to Identify, which works over a
// single fully specified causal diagram. Where Identify answers "is P(y | do(x))
// identifiable in THIS ADMG?", IDP answers the harder question "is it identifiable
// in EVERY MAG the data leaves possible?" — the guarantee an autonomous system
// needs when the graph itself was discovered from data (by FCI) rather than
// asserted.
//
// The algorithm is Jaber, Ribeiro, Zhang & Bareinboim, "Causal Identification
// under Markov Equivalence: Calculus, Algorithm, and Completeness" (NeurIPS 2022),
// as implemented in the authors' PAGId R package. It is sound and complete for the
// no-selection-bias PAGs that this library's FCI produces.
//
// Convention. A PAG's mark matrix uses the same endpoint encoding as pcalg's
// amat.pag: g.mark[i][j] is the mark located AT node j on the edge i–j, so the
// fci.go predicates (isArrow/isTail/isCircle, all reading "the mark at `node`
// looking from `other`") port the pcalg index expressions one-to-one. Structure is
// always read on the graph induced over the current working set, but edge
// VISIBILITY (visibleEdge) is judged on the full PAG — exactly as PAGId threads
// amat_v (full) alongside the induced amat.

// PAGIDResult is the outcome of IdentifyPAG: whether the marginal interventional
// distribution P(y | do(x)) is identifiable across the whole Markov equivalence
// class the PAG represents, and, when it is, a symbolic estimand.
//
// Symbolic-only estimand. Unlike IDResult.Estimand (an ID estimand over a single
// diagram, which Evaluate turns into numbers), the PAG estimand is provided for
// RENDERING only. It is the exact Jaber-Zhang-Bareinboim identification formula —
// a composition of interventional c-factors Q[·] via Propositions 6 and 7 — and it
// String()s faithfully, but its numeric evaluation semantics (the meaning of the
// nested Q[T](·|·) terms as functions over the observational P(V)) are NOT part of
// this release's validated surface. Do not call Estimand.Evaluate on a PAG
// estimand and rely on the number; the identifiability DECISION is the sound,
// tested guarantee here.
type PAGIDResult struct {
	names []string

	// Identifiable reports whether P(y | do(x)) is identifiable in every MAG of
	// the equivalence class described by the PAG. When false, no estimand exists
	// (a valid mathematical answer, not an error).
	Identifiable bool

	// Estimand is the symbolic identification formula, valid only when
	// Identifiable. See the type doc: render-only in this release.
	Estimand *Expr
}

// String renders the estimand with the PAG's variable names when identifiable, or
// a "not identifiable" line otherwise.
func (r *PAGIDResult) String() string {
	if !r.Identifiable {
		return "not identifiable from the PAG"
	}
	return renderExpr(r.Estimand, r.names)
}

// Identify decides whether the marginal causal effect P(y | do(x)) is identifiable
// from the PAG g and the observational distribution, returning its symbolic
// estimand when it is. It is the PAG (equivalence-class) analogue of the
// single-diagram Identify: see IdentifyPAG for the full contract.
func (g *PAG) Identify(y, x []int) (*PAGIDResult, error) {
	return IdentifyPAG(g, y, x)
}

// IdentifyPAG runs the IDP algorithm on a PAG. It decides whether P(y | do(x)) —
// the effect of setting the variables x on the distribution of the variables y —
// is identifiable from the observational distribution across the ENTIRE Markov
// equivalence class the PAG stands for, and returns the identification formula
// when it is.
//
// y and x are variable-index sets: y (outcome) must be non-empty, x (intervention)
// may be empty (then the query is the observational marginal P(y)), and the two
// must be disjoint with no repeats — the same query contract as Identify.
//
// This is strictly harder than identification over one causal diagram. FCI returns
// a PAG, not a single graph, because latent confounders leave the direction of
// some edges undetermined (the circle marks). An effect is IDP-identifiable only
// when the SAME estimand is valid for every MAG in the class; a single circle can
// be the difference between identifiable and not. IDP is sound and complete for
// this problem under the library's standing no-selection-bias scope.
//
// When Identifiable is false the effect is genuinely not identifiable from the
// equivalence class — a legitimate answer, not an error. Errors are returned only
// for malformed queries: ErrEmptyOutcome, ErrEdgeOutOfRange, ErrDuplicateQueryVar,
// ErrOverlappingQuery.
//
// Reference: Jaber, Ribeiro, Zhang & Bareinboim, "Causal Identification under
// Markov Equivalence: Calculus, Algorithm, and Completeness", NeurIPS 2022;
// implemented after the authors' PAGId package (functions IDP, getPCComponentA,
// getRegion, getBucketList, visibleEdge). See Identify for the single-diagram case.
func IdentifyPAG(g *PAG, y, x []int) (*PAGIDResult, error) {
	n := g.Order()
	if err := validateQuery(n, y, x); err != nil {
		return nil, err
	}
	ymask := maskFromSlice(n, y)
	xmask := maskFromSlice(n, x)

	full := make([]bool, n)
	allVars := make([]int, n)
	for i := range full {
		full[i] = true
		allVars[i] = i
	}

	// d = PossAn(Y) in the PAG induced on V \ X.
	vMinusX := maskDiff(full, xmask)
	d := g.pagPossAncestors(vMinusX, ymask)

	// Compute Q[d] from Q[V] = P(V), then marginalise down to Y.
	qd, ok := g.idpIdentify(d, full, jointExpr(allVars))
	if !ok {
		return &PAGIDResult{names: g.names, Identifiable: false}, nil
	}
	est := marginalExpr(maskToSorted(maskDiff(d, ymask)), qd)
	return &PAGIDResult{names: g.names, Identifiable: true, Estimand: est}, nil
}

// idpIdentify computes Q[C] — the interventional c-factor Q[C] = P_{V\C}(c) — from
// the c-factor Q[T] of the current working set t (its symbolic form supplied as
// Qt), following the recursive `identify(C, T)` of Jaber-Zhang-Bareinboim (PAGId's
// IDP). It returns the estimand for Q[C] and whether it is identifiable. C ⊆ T.
//
// The two reduction rules:
//
//	Prop. 6 — a bucket B ⊆ T\C whose pc-component (in the T-induced PAG) meets its
//	  possible descendants only inside B lets us drop B: Q[T\B] = Q[T](t) /
//	  Q[T](b | t\possDe(B)), and we recurse identify(C, T\B). This shrinks T.
//	Prop. 7 — a bucket B ⊆ C whose region is a proper subset of C splits C by
//	  region: Q[C] = Q[R_b]·Q[R_{C\R_b}] / Q[R_b ∩ R_{C\R_b}], each region
//	  identified recursively. This shrinks C.
//
// If neither rule applies (and C is not already T or empty), Q[C] is not
// identifiable.
func (g *PAG) idpIdentify(c, t []bool, qt *Expr) (*Expr, bool) {
	// C empty: Q[∅] = 1.
	if maskEmpty(c) {
		return emptyProduct(), true
	}
	// T\C empty (so C == T): Q[C] = Q[T].
	if maskEmpty(maskDiff(t, c)) {
		return qt, true
	}

	buckets := g.pagBuckets(t)
	tMinusC := maskDiff(t, c)

	// Prop. 6: a bucket wholly inside T\C that can be marginalised away.
	for _, b := range buckets {
		if !maskSubset(b, tMinusC) {
			continue
		}
		pcb := g.pagPCComponent(t, b)         // pc-component of B in P_T
		possDeB := g.pagPossDescendants(t, b) // possible descendants of B in P_T
		inter := maskIntersect(pcb, possDeB)
		if !maskSubset(inter, b) {
			continue
		}
		// Q[T\B] = Q[T](t) / Q[T](b | t\possDe(B)).
		cond := maskDiff(t, possDeB) // t \ possDe(B)
		newQt := ratioExpr(qt, condition(qt, maskToSorted(b), maskToSorted(cond), maskToSorted(t)))
		return g.idpIdentify(c, maskDiff(t, b), newQt)
	}

	// Prop. 7: a bucket wholly inside C whose region does not cover C.
	for _, b := range buckets {
		if !maskSubset(b, c) {
			continue
		}
		regionB := g.pagRegion(c, b)
		if maskEmpty(maskDiff(c, regionB)) { // region(B) == C — no split
			continue
		}
		cMinusRb := maskDiff(c, regionB)
		regionCmb := g.pagRegion(c, cMinusRb)
		regionInter := maskIntersect(regionB, regionCmb)

		qRb, ok1 := g.idpIdentify(regionB, t, qt)
		qCmb, ok2 := g.idpIdentify(regionCmb, t, qt)
		qInter, ok3 := g.idpIdentify(regionInter, t, qt)
		if !ok1 || !ok2 || !ok3 {
			return nil, false
		}
		return ratioExpr(productExpr([]*Expr{qRb, qCmb}), qInter), true
	}

	// Neither proposition applies: Q[C] is not identifiable from Q[T].
	return nil, false
}

// --- PAG primitives (structure on the induced `active` set, visibility on full) --
//
// Every primitive below reads local structure only among nodes marked in `active`
// (the graph induced on the current working set), but any visibleEdge call is
// evaluated on the whole PAG — the amat_v / amat split of PAGId.

// pagPossDescendants returns the mask of possible descendants of the source set
// `src` within the induced graph `active`: every node reachable from a src by a
// potentially-directed path (each step u→w has no arrowhead at u and no tail at w).
// Sources are included. Matches pcalg possDe(type="pag", ds=FALSE).
func (g *PAG) pagPossDescendants(active, src []bool) []bool {
	n := g.Order()
	reach := make([]bool, n)
	var stack []int
	for v := 0; v < n; v++ {
		if active[v] && src[v] {
			reach[v] = true
			stack = append(stack, v)
		}
	}
	for len(stack) > 0 {
		u := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		for w := 0; w < n; w++ {
			if active[w] && !reach[w] && fciAdjacent(g.mark, u, w) && potentiallyDirected(g.mark, u, w) {
				reach[w] = true
				stack = append(stack, w)
			}
		}
	}
	return reach
}

// pagPossAncestors returns the mask of possible ancestors of the target set `tgt`
// within the induced graph `active`: every node that reaches a tgt by a
// potentially-directed path (the reverse of pagPossDescendants). Targets included.
func (g *PAG) pagPossAncestors(active, tgt []bool) []bool {
	n := g.Order()
	reach := make([]bool, n)
	var stack []int
	for v := 0; v < n; v++ {
		if active[v] && tgt[v] {
			reach[v] = true
			stack = append(stack, v)
		}
	}
	for len(stack) > 0 {
		u := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		for w := 0; w < n; w++ {
			// w is a possible ancestor of u iff w→u is potentially directed.
			if active[w] && !reach[w] && fciAdjacent(g.mark, w, u) && potentiallyDirected(g.mark, w, u) {
				reach[w] = true
				stack = append(stack, w)
			}
		}
	}
	return reach
}

// pagBucketOf returns the bucket of `node` in the induced graph `active`: the set
// of nodes connected to it through a chain of circle–circle (o–o) edges, node
// included. A node with no o–o neighbour is its own singleton bucket.
func (g *PAG) pagBucketOf(active []bool, node int) []bool {
	n := g.Order()
	bucket := make([]bool, n)
	bucket[node] = true
	stack := []int{node}
	for len(stack) > 0 {
		u := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		for v := 0; v < n; v++ {
			if active[v] && !bucket[v] && g.isCircleCircle(u, v) {
				bucket[v] = true
				stack = append(stack, v)
			}
		}
	}
	return bucket
}

// pagBuckets partitions the induced graph `active` into its buckets (o–o connected
// components), returned as masks ordered by smallest member (deterministic).
func (g *PAG) pagBuckets(active []bool) [][]bool {
	n := g.Order()
	assigned := make([]bool, n)
	var out [][]bool
	for v := 0; v < n; v++ {
		if !active[v] || assigned[v] {
			continue
		}
		b := g.pagBucketOf(active, v)
		for w := 0; w < n; w++ {
			if b[w] {
				assigned[w] = true
			}
		}
		out = append(out, b)
	}
	return out
}

// pagPCComponent returns the "possible C-component" of the node set `nodesA` in the
// induced graph `active`, per PAGId's getPCComponentA: for each A, take A together
// with its invisible possible children, close under bidirected (↔) edges, then add
// the invisible possible parents of everything reached plus A's o–o neighbours.
func (g *PAG) pagPCComponent(active, nodesA []bool) []bool {
	n := g.Order()
	result := make([]bool, n)
	for a := 0; a < n; a++ {
		if !nodesA[a] {
			continue
		}
		// Start = {A} ∪ invisible possible children of A.
		pcc := g.invPossChildren(active, a)
		pcc[a] = true
		// Close under bidirected edges within `active`.
		stack := maskToSorted(pcc)
		for len(stack) > 0 {
			u := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			for v := 0; v < n; v++ {
				if active[v] && !pcc[v] && g.isBidirected(u, v) {
					pcc[v] = true
					stack = append(stack, v)
				}
			}
		}
		// Add invisible possible parents of every pcc node.
		add := make([]bool, n)
		for node := 0; node < n; node++ {
			if !pcc[node] {
				continue
			}
			pp := g.invPossParents(active, node)
			for v := 0; v < n; v++ {
				if pp[v] {
					add[v] = true
				}
			}
		}
		// Add A's o–o neighbours.
		for v := 0; v < n; v++ {
			if active[v] && g.isCircleCircle(a, v) {
				add[v] = true
			}
		}
		for v := 0; v < n; v++ {
			if pcc[v] || add[v] {
				result[v] = true
			}
		}
	}
	return result
}

// pagRegion returns the region of `nodesA` in the induced graph `active`: the union
// of the buckets of every node in the possible C-component of nodesA (PAGId
// getRegion).
func (g *PAG) pagRegion(active, nodesA []bool) []bool {
	n := g.Order()
	pcc := g.pagPCComponent(active, nodesA)
	region := make([]bool, n)
	for v := 0; v < n; v++ {
		if !pcc[v] {
			continue
		}
		b := g.pagBucketOf(active, v)
		for w := 0; w < n; w++ {
			if b[w] {
				region[w] = true
			}
		}
	}
	return region
}

// invPossChildren returns the "invisible possible children" of A within `active`:
// nodes V with A o→ V, plus nodes V with A → V where that directed edge is INVISIBLE
// in the full PAG. Visibility (visibleEdge) is judged on the whole graph.
func (g *PAG) invPossChildren(active []bool, a int) []bool {
	n := g.Order()
	out := make([]bool, n)
	for v := 0; v < n; v++ {
		if v == a || !active[v] {
			continue
		}
		// A o→ V: circle at A, arrow at V.
		if isCircle(g.mark, a, v) && isArrow(g.mark, v, a) {
			out[v] = true
			continue
		}
		// A → V (tail at A, arrow at V), invisible.
		if isTail(g.mark, a, v) && isArrow(g.mark, v, a) && !g.visibleEdge(a, v) {
			out[v] = true
		}
	}
	return out
}

// invPossParents returns the "invisible possible parents" of A within `active`:
// nodes V with V o→ A, plus nodes V with V → A where that directed edge is INVISIBLE
// in the full PAG.
func (g *PAG) invPossParents(active []bool, a int) []bool {
	n := g.Order()
	out := make([]bool, n)
	for v := 0; v < n; v++ {
		if v == a || !active[v] {
			continue
		}
		// V o→ A: circle at V, arrow at A.
		if isCircle(g.mark, v, a) && isArrow(g.mark, a, v) {
			out[v] = true
			continue
		}
		// V → A (tail at V, arrow at A), invisible.
		if isTail(g.mark, v, a) && isArrow(g.mark, a, v) && !g.visibleEdge(v, a) {
			out[v] = true
		}
	}
	return out
}

// isCircleCircle reports an o–o edge between i and j (circle at both ends).
func (g *PAG) isCircleCircle(i, j int) bool {
	return isCircle(g.mark, i, j) && isCircle(g.mark, j, i)
}

// isBidirected reports an i ↔ j edge (arrowhead at both ends).
func (g *PAG) isBidirected(i, j int) bool {
	return isArrow(g.mark, i, j) && isArrow(g.mark, j, i)
}

// --- edge visibility (judged on the full PAG) ----------------------------

// visibleEdge reports whether the directed edge x → z is visible in the PAG, i.e.
// definitely not confounded by a latent common cause of x and z (Zhang 2008;
// pcalg::visibleEdge). Two witnessing scenarios:
//
//  1. some vertex c not adjacent to z has an arrowhead into x (c *→ x), or
//  2. there is a discriminating collider path into x whose interior vertices are
//     all parents of z, starting from an x ↔ c with c → z.
//
// A visible edge behaves like an ordinary directed edge of a DAG for
// identification; an invisible one may hide confounding. Evaluated on the whole
// graph, never an induced subgraph.
func (g *PAG) visibleEdge(x, z int) bool {
	n := g.Order()
	// Scenario 1: ∃ c with c *→ x (arrowhead at x) and c not adjacent to z.
	for c := 0; c < n; c++ {
		if c == x || c == z {
			continue
		}
		if fciAdjacent(g.mark, x, c) && isArrow(g.mark, x, c) && !fciAdjacent(g.mark, z, c) {
			return true
		}
	}
	// Scenario 2: ∃ c with x ↔ c and c → z, admitting a minimal discriminating path.
	for c := 0; c < n; c++ {
		if c == x || c == z {
			continue
		}
		if g.isBidirected(x, c) && isTail(g.mark, c, z) && isArrow(g.mark, z, c) {
			if g.minDiscrPathExists(c, x, z) {
				return true
			}
		}
	}
	return false
}

// minDiscrPathExists reports whether a minimal discriminating path exists for the
// triple ⟨a, b, c⟩ (a the collider vertex adjacent to b, c the far endpoint),
// ported from pcalg::minDiscrPath — only its existence matters to visibleEdge. The
// search grows collider paths back from a: every interior vertex must be a collider
// on the path AND a parent of c, terminating when it reaches a vertex not adjacent
// to c.
func (g *PAG) minDiscrPathExists(a, b, c int) bool {
	n := g.Order()
	visited := make([]bool, n)
	visited[a], visited[b], visited[c] = true, true, true

	var list [][]int
	// Seed: paths [a, d] for every d *→ a not yet visited.
	for d := 0; d < n; d++ {
		if visited[d] {
			continue
		}
		if fciAdjacent(g.mark, a, d) && isArrow(g.mark, a, d) { // d *→ a
			list = append(list, []int{a, d})
		}
	}
	for len(list) > 0 {
		mpath := list[0]
		list = list[1:]
		d := mpath[len(mpath)-1]
		if !fciAdjacent(g.mark, c, d) { // d not adjacent to c: discriminating path found
			return true
		}
		pred := mpath[len(mpath)-2]
		// d → c (tail at d, arrow at c) and pred *→ d (d a collider): extend.
		if isTail(g.mark, d, c) && isArrow(g.mark, c, d) && isArrow(g.mark, d, pred) {
			visited[d] = true
			base := mpath[1:] // drop the first element, as pcalg does
			for r := 0; r < n; r++ {
				if r == d || visited[r] {
					continue
				}
				if fciAdjacent(g.mark, d, r) && isArrow(g.mark, d, r) { // r *→ d
					np := append(append([]int(nil), base...), r)
					list = append(list, np)
				}
			}
		}
	}
	return false
}

// --- small mask helpers --------------------------------------------------

// maskSubset reports whether a ⊆ b.
func maskSubset(a, b []bool) bool {
	for i := range a {
		if a[i] && !b[i] {
			return false
		}
	}
	return true
}

// emptyProduct is the estimand constant 1 (an empty product of factors); it renders
// as "1" and evaluates to the scalar 1.
func emptyProduct() *Expr { return &Expr{kind: exprProduct} }
