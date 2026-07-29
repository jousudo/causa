package causa

// This file implements CIDP: identification of CONDITIONAL causal effects from a
// PAG — P(y | do(x), z), the effect of x on y within the context z — the
// equivalence-class analogue of IdentifyConditional (which works over one asserted
// diagram). It is the conditional companion to IdentifyPAG, the way IDC is to ID.
//
// The algorithm is the Jaber-Zhang-Bareinboim CIDP (NeurIPS 2022), ported from the
// authors' PAGId package (function CIDP). It rests on the PAG do-calculus Rule 2,
// which in turn needs definite-status m-separation over a PAG and the two graph
// manipulations (remove edges into a set; remove VISIBLE edges out of a set).
// Structure and visibility follow the same conventions as idp.go.

// IdentifyConditional decides whether the conditional causal effect P(y | do(x), z)
// is identifiable from the PAG g, returning its symbolic estimand when it is. It is
// the PAG (equivalence-class) analogue of the single-diagram
// (*Diagram).IdentifyConditional; see IdentifyConditionalPAG for the full contract.
func (g *PAG) IdentifyConditional(y, x, z []int) (*PAGIDResult, error) {
	return IdentifyConditionalPAG(g, y, x, z)
}

// IdentifyConditionalPAG runs the CIDP algorithm on a PAG. It decides whether
// P(y | do(x), z) — the effect of setting x on y WITHIN the subpopulation where the
// observed context z takes a value — is identifiable from the observational
// distribution across the entire Markov equivalence class the PAG represents, and
// returns the identification formula when it is.
//
// y, x and z are pairwise-disjoint variable-index sets: y (outcome) must be
// non-empty, x (intervention) and z (context) may each be empty. With z empty the
// query reduces to IdentifyPAG (the marginal effect).
//
// Algorithm. Like the single-diagram IDC, it uses do-calculus Rule 2 to move each
// context variable that behaves like an intervention out of z and into x — decided
// here by definite-status m-separation over the PAG in the manipulated graph
// P_{\overline{W}, \underline{X}} — plus a preliminary phase (Line 2) that either
// absorbs the buckets straddling the possible-ancestor set into the intervention or
// declares non-identifiability. The residual is then P_x(y, z) / Σ_y P_x(y, z),
// with the joint identified by IDP. If that joint is not identifiable, neither is
// the conditional effect.
//
// Scope mirrors IdentifyPAG: the identifiability DECISION is the sound, tested
// guarantee (cross-checked against the reference PAGId::CIDP), and the estimand is
// turned into numbers with (*PAGIDResult).Evaluate (validated against brute-force
// truth on random latent SCMs). A PAG with an undirected (—) edge is refused with
// ErrSelectionBiasUnsupported, mirroring IdentifyPAG: identification under selection
// bias is out of scope. Errors: ErrEmptyOutcome, ErrEdgeOutOfRange,
// ErrDuplicateQueryVar, ErrOverlappingQuery, ErrSelectionBiasUnsupported.
//
// Reference: Jaber, Ribeiro, Zhang & Bareinboim, "Causal Identification under
// Markov Equivalence: Calculus, Algorithm, and Completeness", NeurIPS 2022
// (Algorithm CIDP); implemented after the authors' PAGId package.
func IdentifyConditionalPAG(g *PAG, y, x, z []int) (*PAGIDResult, error) {
	n := g.Order()
	if err := validateConditionalQuery(n, y, x, z); err != nil {
		return nil, err
	}
	if hasUndirectedEdge(g) {
		return nil, ErrSelectionBiasUnsupported
	}
	if len(z) == 0 {
		return IdentifyPAG(g, y, x)
	}

	ymask := maskFromSlice(n, y)
	xmask := maskFromSlice(n, x)
	zmask := maskFromSlice(n, z)
	full := make([]bool, n)
	for i := range full {
		full[i] = true
	}

	// Buckets of the full PAG (fixed across the algorithm).
	buckets := g.pagBuckets(full)

	// Line 2: while a bucket straddles D = PossAn(Y∪Z) in P_{V\X} (meets it but is
	// not contained in it), try to move its X-part into the context via Rule 2;
	// failure there is non-identifiability.
	for {
		d := g.pagPossAncestors(maskDiff(full, xmask), maskUnion(ymask, zmask))
		bi := straddlingBucket(buckets, d)
		if bi == nil {
			break
		}
		xprime := maskIntersect(bi, xmask)
		w := maskDiff(xmask, xprime)
		if !g.rule2(xprime, ymask, zmask, w) {
			return &PAGIDResult{names: g.names, y: y, x: x, z: z, Identifiable: false}, nil
		}
		xmask = w
		zmask = maskUnion(zmask, xprime)
	}

	// Line 9: partition Z by bucket; while some part z_i can be moved into X by
	// Rule 2, move it. (A context variable that acts as an intervention.)
	zparts := zPartition(buckets, zmask)
	for {
		idx := -1
		for i, zi := range zparts {
			if g.rule2(zi, ymask, maskDiff(zmask, zi), xmask) {
				idx = i
				break
			}
		}
		if idx < 0 {
			break
		}
		zi := zparts[idx]
		xmask = maskUnion(xmask, zi)
		zmask = maskDiff(zmask, zi)
		zparts = append(zparts[:idx], zparts[idx+1:]...)
	}

	// Residual: P_x(y | z) = P_x(y, z) / Σ_y P_x(y, z), the joint by IDP.
	inner, err := IdentifyPAG(g, maskToSorted(maskUnion(ymask, zmask)), maskToSorted(xmask))
	if err != nil {
		return nil, err
	}
	if !inner.Identifiable {
		return &PAGIDResult{names: g.names, y: y, x: x, z: z, Identifiable: false}, nil
	}
	den := marginalExpr(maskToSorted(ymask), inner.Estimand)
	return &PAGIDResult{names: g.names, y: y, x: x, z: z, Identifiable: true, Estimand: ratioExpr(inner.Estimand, den)}, nil
}

// straddlingBucket returns the first bucket that meets d but is not contained in it
// (bi ∩ d ≠ ∅ and bi ⊄ d), or nil if none — PAGId's checkCondLine2.
func straddlingBucket(buckets [][]bool, d []bool) []bool {
	for _, bi := range buckets {
		if !maskEmpty(maskIntersect(bi, d)) && !maskSubset(bi, d) {
			return bi
		}
	}
	return nil
}

// zPartition splits z by bucket: for each bucket, the (non-empty) part of z it
// contains — PAGId's zpartition construction.
func zPartition(buckets [][]bool, z []bool) [][]bool {
	var parts [][]bool
	for _, bi := range buckets {
		inter := maskIntersect(bi, z)
		if !maskEmpty(inter) {
			parts = append(parts, inter)
		}
	}
	return parts
}

// --- do-calculus Rule 2 over a PAG ---------------------------------------

// rule2 reports whether do-calculus Rule 2 applies: X and Y are m-separated by
// W ∪ Z in the manipulated PAG P_{\overline{W}, \underline{X}} (edges into W
// removed, then VISIBLE edges out of X removed). When it holds, intervening on X is
// exchangeable with conditioning on X for the effect on Y in that context.
func (g *PAG) rule2(x, y, z, w []bool) bool {
	n := g.Order()
	up := xUpper(g.mark, n, w) // P_{\overline{W}}
	low := xLower(up, n, x)    // then P_{…, \underline{X}}
	return pagMSeparated(low, n, x, y, maskUnion(w, z))
}

// xUpper returns a copy of the PAG with every edge INTO a vertex of s removed (an
// arrowhead at that vertex): the manipulation P_{\overline{S}}.
func xUpper(m [][]Mark, n int, s []bool) [][]Mark {
	out := copyMarks(m, n)
	for si := 0; si < n; si++ {
		if !s[si] {
			continue
		}
		for v := 0; v < n; v++ {
			if v != si && isArrow(out, si, v) { // arrowhead at si from v: v *→ si
				out[v][si] = NoMark
				out[si][v] = NoMark
			}
		}
	}
	return out
}

// xLower returns a copy of the PAG with every VISIBLE directed edge OUT OF a vertex
// of s removed: the manipulation P_{\underline{S}}. A node v is reached by a visible
// edge out of si when si → v (tail at si) and that edge is visible on the given
// (already-upper-manipulated) graph.
func xLower(m [][]Mark, n int, s []bool) [][]Mark {
	out := copyMarks(m, n)
	for si := 0; si < n; si++ {
		if !s[si] {
			continue
		}
		for v := 0; v < n; v++ {
			if v == si {
				continue
			}
			if isTail(out, si, v) && fciAdjacent(out, si, v) && visibleEdgePAG(m, n, si, v) {
				out[v][si] = NoMark
				out[si][v] = NoMark
			}
		}
	}
	return out
}

func copyMarks(m [][]Mark, n int) [][]Mark {
	out := make([][]Mark, n)
	for i := 0; i < n; i++ {
		out[i] = append([]Mark(nil), m[i]...)
	}
	return out
}

// --- definite-status m-separation over a PAG -----------------------------

// pagMSeparated reports whether every x ∈ xs is m-separated from every y ∈ ys given
// zset in the PAG m, using definite-status paths (Zhang's m-separation for PAGs): a
// path is active only if every consecutive triple is a collider with a descendant
// in zset, or a definite non-collider outside zset. m-separated iff no such active
// path exists between the sets.
func pagMSeparated(m [][]Mark, n int, xs, ys, zset []bool) bool {
	for x := 0; x < n; x++ {
		if !xs[x] {
			continue
		}
		for y := 0; y < n; y++ {
			if !ys[y] || y == x {
				continue
			}
			if defConnectingPathExists(m, n, x, y, zset) {
				return false
			}
		}
	}
	return true
}

// defConnectingPathExists reports whether a definite-status path from x to y active
// given zset exists in the PAG m — PAGId's getDefConPath. It grows paths from x,
// pruning any whose last interior triple is not definite-status-connecting.
func defConnectingPathExists(m [][]Mark, n, x, y int, zset []bool) bool {
	list := [][]int{{x}}
	inPath := func(path []int, v int) bool {
		for _, u := range path {
			if u == v {
				return true
			}
		}
		return false
	}
	for len(list) > 0 {
		cur := list[0]
		L := len(cur)
		last := cur[L-1]
		if last == y && L == 2 { // the direct edge x — y
			return true
		}
		if L > 2 {
			vi, vm, vj := cur[L-3], cur[L-2], cur[L-1]
			if !isDefMConTriplet(m, n, vi, vm, vj, zset) {
				list = list[1:] // dead path: this triple blocks it
				continue
			}
			if last == y {
				return true // an active definite-status path reached y
			}
		}
		// Extend by the neighbours of `last` not already on the path.
		var adj []int
		for v := 0; v < n; v++ {
			if v != last && fciAdjacent(m, last, v) && !inPath(cur, v) {
				adj = append(adj, v)
			}
		}
		if len(adj) == 0 {
			list = list[1:]
			continue
		}
		reachY := false
		rest := adj[:0]
		for _, v := range adj {
			if v == y {
				reachY = true
			} else {
				rest = append(rest, v)
			}
		}
		if reachY {
			// Replace the front with cur+y (processed next), keep extending the rest.
			list[0] = append(append([]int(nil), cur...), y)
		} else {
			list = list[1:]
		}
		for _, v := range rest {
			list = append(list, append(append([]int(nil), cur...), v))
		}
	}
	return false
}

// isDefMConTriplet reports whether the triple ⟨vi, vm, vj⟩ is definite-status
// connecting given zset: a collider (arrowheads at vm from both sides) with a
// directed descendant of vm in zset, or a definite non-collider (a tail at vm, or a
// circle at vm on both sides with vi, vj non-adjacent) whose middle is not in zset.
func isDefMConTriplet(m [][]Mark, n, vi, vm, vj int, zset []bool) bool {
	collider := isArrow(m, vm, vi) && isArrow(m, vm, vj)
	if collider {
		de := directedDescendants(m, n, vm)
		for v := 0; v < n; v++ {
			if de[v] && zset[v] {
				return true // a descendant of the collider is in Z: active
			}
		}
		return false
	}
	tailAtVm := isTail(m, vm, vi) || isTail(m, vm, vj)
	circleBoth := isCircle(m, vm, vi) && isCircle(m, vm, vj) && !fciAdjacent(m, vi, vj)
	if tailAtVm || circleBoth { // definite non-collider
		return !zset[vm] // active iff the middle is not conditioned on
	}
	return false // not a definite-status triple
}

// directedDescendants returns the mask of vertices reachable from src by directed
// edges (tail at the source end, arrowhead at the target end), src included —
// pcalg::searchAM(type="de").
func directedDescendants(m [][]Mark, n, src int) []bool {
	de := make([]bool, n)
	de[src] = true
	stack := []int{src}
	for len(stack) > 0 {
		u := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		for v := 0; v < n; v++ {
			if !de[v] && isDirected(m, u, v) { // u → v
				de[v] = true
				stack = append(stack, v)
			}
		}
	}
	return de
}
