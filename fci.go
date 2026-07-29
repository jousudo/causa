package causa

import (
	"errors"
	"fmt"
	"strings"
)

// PAG construction errors.
var (
	// ErrInvalidPAGMark is returned by NewPAG when an edge endpoint is not one of
	// Circle, Arrow, or Tail (NoMark is not a valid endpoint of a present edge).
	ErrInvalidPAGMark = errors.New("causa: PAG edge endpoint must be a circle, arrowhead, or tail")
	// ErrDuplicatePAGEdge is returned by NewPAG when two edges name the same pair.
	ErrDuplicatePAGEdge = errors.New("causa: duplicate PAG edge")
)

// Mark is the endpoint mark on one end of a PAG edge. A PAG (see PAG) has three
// kinds of endpoint mark; the pair of marks on an edge encodes what the data
// determined about that adjacency.
//
//	Circle (o) — undetermined: the equivalence class contains PAGs with either a
//	             tail or an arrowhead here.
//	Arrow  (>) — an arrowhead: this endpoint is NOT an ancestor of the other (it
//	             is a definite non-cause across the whole equivalence class).
//	Tail   (-) — a tail: this endpoint IS an ancestor of the other.
//
// NoMark marks a non-adjacency (there is no edge, so no endpoint).
type Mark uint8

const (
	// NoMark means the two nodes are non-adjacent — there is no edge endpoint.
	NoMark Mark = iota
	// Circle (drawn "o") is an undetermined endpoint.
	Circle
	// Arrow (drawn ">") is an arrowhead endpoint: a definite non-ancestor.
	Arrow
	// Tail (drawn "-") is a tail endpoint: a definite ancestor.
	Tail
)

// String renders a single endpoint mark as one character: "o", ">", "-", or, for
// NoMark, "." (no endpoint).
func (m Mark) String() string {
	switch m {
	case Circle:
		return "o"
	case Arrow:
		return ">"
	case Tail:
		return "-"
	default:
		return "."
	}
}

// PAGEdge is one edge of a PAG between nodes A and B (reported once, A < B), with
// the endpoint mark at each end: MarkA is the mark touching A, MarkB the mark
// touching B. The common configurations are A o-o B (both circles, wholly
// undetermined), A o→ B (Circle at A, Arrow at B), A → B (Tail at A, Arrow at B,
// a definite direction) and A ↔ B (Arrow at both — a definite latent common
// cause).
type PAGEdge struct {
	A, B         int
	MarkA, MarkB Mark
}

// PAG is a Partial Ancestral Graph: the output of FCI, and the latent-confounder
// analogue of the CPDAG that PCStable returns. It represents a Markov equivalence
// class of maximal ancestral graphs (MAGs) — the structures that remain possible
// once unobserved common causes are admitted.
//
// It is NOT a single causal graph, and its edges are read endpoint by endpoint,
// not as whole arrows:
//
//   - An arrowhead (Arrow) at a node is compelled: across every graph in the
//     class that endpoint is an arrowhead, so the node is a definite NON-ancestor
//     of the other along this edge.
//   - A tail (Tail) is likewise compelled the other way: a definite ancestor.
//   - A circle (Circle) is undetermined: the class contains graphs with a tail
//     here and graphs with an arrowhead.
//
// Hence A → B means B is not a cause of A in any member of the class (a definite
// direction), while A ↔ B means neither causes the other and a latent common
// cause is implied. A o→ B fixes the arrowhead at B but leaves A's end open, and
// A o-o B is wholly undetermined. Reading a circle as "no relationship" is the
// classic error — it means "direction not identified", exactly as an undirected
// CPDAG edge does.
type PAG struct {
	names []string
	// mark[i][j] is the endpoint mark located AT node j on the edge between i and
	// j; NoMark when i and j are non-adjacent. The matrix is kept consistent:
	// mark[i][j] == NoMark iff mark[j][i] == NoMark.
	mark [][]Mark
}

// NewPAG builds a PAG directly from its edges, for identifying effects (IdentifyPAG)
// over a partial ancestral graph obtained outside this package — asserted by hand,
// or discovered by another tool — without re-running FCI. names labels the vertices
// and fixes their count; each PAGEdge gives an endpoint pair A, B (A ≠ B) with the
// mark at each end (MarkA at A, MarkB at B), every mark one of Circle, Arrow, or
// Tail. Report each edge once; a pair may not repeat.
//
// The endpoint encoding matches the PAG type: an edge {A, B, Tail, Arrow} is the
// directed A → B, {A, B, Arrow, Arrow} is the bidirected A ↔ B, {A, B, Circle,
// Arrow} is A o→ B, and {A, B, Circle, Circle} is A o—o B. NewPAG does not verify
// that the marks form a valid MAG equivalence class (a maximally informative,
// consistent PAG); it validates only the structural well-formedness above and
// trusts the caller for the rest, exactly as NewDiagram trusts an asserted ADMG.
//
// Errors: ErrTooFewVariables (no names), ErrEdgeOutOfRange (an endpoint index out
// of range), ErrSelfLoop (A == B), ErrInvalidPAGMark (a NoMark or unknown endpoint),
// ErrDuplicatePAGEdge (a repeated pair).
func NewPAG(names []string, edges []PAGEdge) (*PAG, error) {
	if len(names) == 0 {
		return nil, ErrTooFewVariables
	}
	n := len(names)
	m := make([][]Mark, n)
	for i := range m {
		m[i] = make([]Mark, n)
	}
	valid := func(mk Mark) bool { return mk == Circle || mk == Arrow || mk == Tail }
	for _, e := range edges {
		if e.A < 0 || e.A >= n || e.B < 0 || e.B >= n {
			return nil, ErrEdgeOutOfRange
		}
		if e.A == e.B {
			return nil, ErrSelfLoop
		}
		if !valid(e.MarkA) || !valid(e.MarkB) {
			return nil, ErrInvalidPAGMark
		}
		if m[e.A][e.B] != NoMark {
			return nil, ErrDuplicatePAGEdge
		}
		m[e.A][e.B] = e.MarkB // mark at B
		m[e.B][e.A] = e.MarkA // mark at A
	}
	return &PAG{names: append([]string(nil), names...), mark: m}, nil
}

// Order returns the number of variables (nodes) in the graph.
func (g *PAG) Order() int { return len(g.names) }

// Nodes returns a copy of the variable names, indexed as in the input data.
func (g *PAG) Nodes() []string { return append([]string(nil), g.names...) }

// Adjacent reports whether i and j share an edge.
func (g *PAG) Adjacent(i, j int) bool { return g.mark[i][j] != NoMark }

// MarkAt returns the endpoint mark located at node j on the edge between i and j,
// or NoMark if i and j are non-adjacent. Note the asymmetry: MarkAt(i, j) is the
// mark touching j, and MarkAt(j, i) the mark touching i.
func (g *PAG) MarkAt(i, j int) Mark { return g.mark[i][j] }

// Directed reports whether the edge is the definite direction i → j (a tail at i
// and an arrowhead at j): i is an ancestor of j and j is not a cause of i in any
// graph of the equivalence class.
func (g *PAG) Directed(i, j int) bool {
	return g.mark[i][j] == Arrow && g.mark[j][i] == Tail
}

// Bidirected reports whether the edge is i ↔ j (an arrowhead at both ends):
// neither is a cause of the other and a latent common cause of i and j is
// implied. This is the structure PCStable's causal-sufficiency assumption cannot
// express and FCI exists to recover.
func (g *PAG) Bidirected(i, j int) bool {
	return g.mark[i][j] == Arrow && g.mark[j][i] == Arrow
}

// Edges returns every edge once, as PAGEdge values with A < B, in ascending
// (A, B) order.
func (g *PAG) Edges() []PAGEdge {
	var edges []PAGEdge
	p := len(g.mark)
	for i := 0; i < p; i++ {
		for j := i + 1; j < p; j++ {
			if g.mark[i][j] != NoMark {
				edges = append(edges, PAGEdge{A: i, B: j, MarkA: g.mark[j][i], MarkB: g.mark[i][j]})
			}
		}
	}
	return edges
}

// String renders the PAG as a human-readable, deterministic multi-line summary: a
// header with the node count and names, then one line per edge "A <sym> B" with
// sym the three-character endpoint rendering (o-o, o->, -->, <->, and so on),
// edges sorted by (A, B).
func (g *PAG) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "PAG with %d nodes %v:", len(g.names), g.names)
	for _, e := range g.Edges() {
		fmt.Fprintf(&b, "\n  %s %s %s", g.names[e.A], edgeSymbol(e.MarkA, e.MarkB), g.names[e.B])
	}
	return b.String()
}

// edgeSymbol renders the three-character middle of an edge from its two endpoint
// marks, e.g. (Circle, Arrow) -> "o->", (Tail, Arrow) -> "-->", (Arrow, Arrow) ->
// "<->". The left mark is drawn on its own side (Arrow as "<"), the right mark on
// its (Arrow as ">"), joined by "-".
func edgeSymbol(left, right Mark) string {
	leftCh := map[Mark]byte{Circle: 'o', Arrow: '<', Tail: '-'}[left]
	rightCh := map[Mark]byte{Circle: 'o', Arrow: '>', Tail: '-'}[right]
	return string([]byte{leftCh, '-', rightCh})
}

// FCIOptions configures FCI. A nil *FCIOptions, or a zero-valued field within it,
// selects the documented default for that setting. The fields mirror PCOptions:
// FCI shares the PC-stable skeleton search and the same conditional-independence
// machinery.
type FCIOptions struct {
	// Alpha is the significance level of the conditional-independence test: an
	// edge is deleted when a test reports a p-value greater than Alpha. Zero
	// selects the default of 0.05; any other value must satisfy 0 < Alpha < 1 or
	// FCI returns ErrInvalidAlpha.
	Alpha float64

	// MaxCondSet caps the size of the conditioning sets both the skeleton search
	// and the Possible-D-SEP refinement will consider. Zero (the default) means
	// unbounded up to the honest small-sample cap. A positive value trades
	// completeness for speed on dense graphs.
	MaxCondSet int

	// CITest is the conditional-independence test. Nil selects FisherZTest (the
	// linear-Gaussian default). See CITest for the contract a custom test must
	// satisfy.
	CITest CITest

	// SelectionBias, when true, admits selection bias: FCI additionally applies
	// Zhang's orientation rules R5, R6 and R7, the only rules that can introduce
	// undirected (—, tail–tail) edges. An undirected edge records an ancestral
	// relation to a latent SELECTION variable — a variable the sampling process
	// implicitly conditioned on — so under this mode a Tail at α on α—β means α is
	// an ancestor of β OR of a selection variable, not β alone.
	//
	// The default (false) keeps the standing no-selection-bias scope: R5–R7 never
	// run, no — edge is ever produced, and the returned PAG is byte-identical to
	// what earlier versions returned. Turning it on when the data has no selection
	// bias is harmless — R5–R7 have nothing to fire on — but it is not free, so it
	// stays opt-in.
	SelectionBias bool
}

// FCI runs the Fast Causal Inference algorithm of Spirtes, Glymour & Scheines and
// returns the estimated PAG. Unlike PCStable, FCI does NOT assume causal
// sufficiency: it is sound and complete for structure discovery when unobserved
// common causes (latent confounders) may exist, which it reports as bidirected
// (↔) edges.
//
// Input. Identical to PCStable: data holds one variable per outer slice, every
// variable of length n; names, if non-nil, has one entry per variable (else
// V0, V1, …). The default test treats the data as continuous linear-Gaussian.
//
// Algorithm. Five phases:
//
//  1. Initial skeleton. The order-independent PC-stable adjacency search
//     (shared with PCStable) thins the complete graph and records a separating
//     set for every deleted pair. All surviving edges start as o-o.
//
//  2. Colliders. Every unshielded triple i o-o k o-o j with i, j non-adjacent is
//     oriented i *→ k ←* j iff k is not in the separating set of (i, j), placing
//     arrowheads into k.
//
//  3. Possible-D-SEP refinement (the step that makes it FCI, not PC). The PC
//     skeleton can leave edges that are removable only by conditioning on nodes
//     NOT adjacent to the pair. For each still-adjacent pair, FCI searches
//     subsets of Possible-D-SEP for a separating set and deletes the edge when it
//     finds one, recording the new separating set. Possible-D-SEP is snapshotted
//     before any deletion, so — like the PC-stable skeleton — the result does not
//     depend on the order pairs are visited.
//
//  4. Re-orient. All surviving edges are reset to o-o and the unshielded-collider
//     rule is applied again, now with the refined skeleton and separating sets.
//
//  5. Orientation to completeness. Zhang's rules R1–R4 and R8–R10 are applied to
//     closure, propagating arrowheads and tails without contradiction.
//
// Selection bias. By default this assumes there is NO selection bias (no
// conditioning induced by the sampling process): it runs Zhang's rules R1–R4 and
// R8–R10, the sound and COMPLETE set for the latent-confounder-only class, and
// never produces a tail–tail (—) edge. Setting FCIOptions.SelectionBias admits
// selection bias by additionally running R5, R6 and R7 — the only rules that
// produce — edges — which is again sound and complete (Zhang 2008) for the larger
// class that also permits latent selection variables. The skeleton, collider and
// R1–R4/R8–R10 phases are identical in both modes; only the extra three rules and
// the interpretation of a Tail differ (see FCIOptions.SelectionBias). On data
// with no selection bias the two modes return the identical PAG, R5–R7 having
// nothing to fire on.
//
// Determinism. Every phase iterates variable indices, enumerates subsets in
// lexicographic order, and searches paths in index order; there is no dependence
// on map iteration order. The same input always yields the same PAG, and the
// skeleton is invariant to the order in which the variables are presented.
//
// Assumptions. Faithfulness and a correct CITest (the default assumes
// linear-Gaussian data), plus the no-selection-bias scope above. Causal
// sufficiency is NOT required — that is the whole point. The honest small-sample
// cap of the CI test carries over: conditioning sets stop growing once
// n − |S| − 3 < 1, so independencies needing larger conditioning sets than the
// sample supports cannot be tested.
//
// Errors: the same set as PCStable — ErrTooFewVariables, ErrUnequalLengths,
// ErrTooFewSamples, ErrNonFinite, ErrNameCount, ErrInvalidAlpha, and any error
// surfaced by the CITest (which aborts the run).
//
// Reference: Spirtes, Glymour & Scheines, "Causation, Prediction, and Search"
// (2nd ed., 2000), ch. 6 (FCI, Possible-D-SEP); Zhang, "On the completeness of
// orientation rules for causal discovery in the presence of latent confounders
// and selection bias", Artificial Intelligence 172 (2008) 1873–1896 (rules
// R1–R10); Colombo & Maathuis, JMLR 15 (2014) (order-independent skeleton).
func FCI(data [][]float64, names []string, opts *FCIOptions) (*PAG, error) {
	p := len(data)
	if p < 2 {
		return nil, ErrTooFewVariables
	}
	n := len(data[0])
	for _, v := range data {
		if len(v) != n {
			return nil, ErrUnequalLengths
		}
	}
	if n < 4 {
		return nil, ErrTooFewSamples
	}
	for _, v := range data {
		for _, x := range v {
			if !isFinite(x) {
				return nil, ErrNonFinite
			}
		}
	}

	switch {
	case names == nil:
		names = make([]string, p)
		for i := range names {
			names[i] = fmt.Sprintf("V%d", i)
		}
	case len(names) != p:
		return nil, ErrNameCount
	default:
		names = append([]string(nil), names...)
	}

	alpha := 0.05
	maxCond := 0
	ci := CITest(FisherZTest)
	selection := false
	if opts != nil {
		if opts.Alpha != 0 {
			if opts.Alpha <= 0 || opts.Alpha >= 1 {
				return nil, ErrInvalidAlpha
			}
			alpha = opts.Alpha
		}
		if opts.MaxCondSet > 0 {
			maxCond = opts.MaxCondSet
		}
		if opts.CITest != nil {
			ci = opts.CITest
		}
		selection = opts.SelectionBias
	}

	// Phase 1: PC-stable skeleton (shared with PCStable).
	adj, sepset, err := pcSkeleton(data, ci, alpha, maxCond, n, p)
	if err != nil {
		return nil, err
	}

	// Represent the skeleton as a mark matrix, every edge o-o.
	m := make([][]Mark, p)
	for i := range m {
		m[i] = make([]Mark, p)
		for j := range m[i] {
			if i != j && adj[i][j] {
				m[i][j] = Circle
			}
		}
	}

	// Phase 2: orient unshielded colliders on the initial skeleton.
	orientCollidersPAG(m, sepset, p)

	// Phase 3: Possible-D-SEP refinement (may delete more edges + record sepsets).
	if err := fciPDSepRefine(data, m, sepset, ci, alpha, maxCond, n, p); err != nil {
		return nil, err
	}

	// Phase 4: reset to o-o and re-orient colliders with the refined information.
	resetToCircles(m, p)
	orientCollidersPAG(m, sepset, p)

	// Phase 5: Zhang's orientation rules to closure.
	applyFCIRules(m, sepset, p, selection)

	return &PAG{names: names, mark: m}, nil
}

// --- endpoint-mark helpers -----------------------------------------------
//
// Throughout, mark[a][b] is the mark AT b on the edge a-b. The predicates read
// "is the mark at `node` (looking from `other`) an X?".

func fciAdjacent(m [][]Mark, i, j int) bool { return m[i][j] != NoMark }
func isArrow(m [][]Mark, node, other int) bool {
	return m[other][node] == Arrow
}
func isTail(m [][]Mark, node, other int) bool   { return m[other][node] == Tail }
func isCircle(m [][]Mark, node, other int) bool { return m[other][node] == Circle }

// isDirected reports the definite direction from → to (tail at from, arrow at to).
func isDirected(m [][]Mark, from, to int) bool {
	return isArrow(m, to, from) && isTail(m, from, to)
}

// potentiallyDirected reports whether the edge from-to can be oriented from → to
// consistently with its current marks: no arrowhead at `from` and no tail at `to`.
func potentiallyDirected(m [][]Mark, from, to int) bool {
	return m[to][from] != Arrow && m[from][to] != Tail
}

// orientEdge sets the endpoint mark at `node` (on the edge node-other) to mk,
// returning whether that actually changed anything. Callers drive the rule loop
// off the returned bool so it terminates at closure.
func orientEdge(m [][]Mark, node, other int, mk Mark) bool {
	if m[other][node] == mk {
		return false
	}
	m[other][node] = mk
	return true
}

// --- collider orientation ------------------------------------------------

// orientCollidersPAG orients every unshielded collider by placing arrowheads into
// the middle node. It is always invoked on an all-circles graph (right after the
// skeleton, and again after the reset), so setting arrowheads never conflicts
// with an existing tail; the far endpoints are left as circles.
func orientCollidersPAG(m [][]Mark, sepset [][][]int, p int) {
	for k := 0; k < p; k++ {
		for i := 0; i < p; i++ {
			if i == k || !fciAdjacent(m, i, k) {
				continue
			}
			for j := i + 1; j < p; j++ {
				if j == k || !fciAdjacent(m, j, k) {
					continue
				}
				if fciAdjacent(m, i, j) { // shielded triple
					continue
				}
				if !contains(sepset[i][j], k) {
					m[i][k] = Arrow // arrowhead at k from i
					m[j][k] = Arrow // arrowhead at k from j
				}
			}
		}
	}
}

// --- Possible-D-SEP refinement -------------------------------------------

// fciPDSepRefine performs the FCI-specific edge thinning: for every still-adjacent
// pair it looks for a separating set inside Possible-D-SEP (which can contain
// nodes NOT adjacent to the pair — the independencies PC's adjacency-only search
// cannot reach). Possible-D-SEP is computed once, up front, on the post-collider
// graph, so deletions during the phase do not change any other pair's candidate
// set: the refinement is order-independent, matching the PC-stable skeleton.
func fciPDSepRefine(data [][]float64, m [][]Mark, sepset [][][]int, ci CITest, alpha float64, maxCond, n, p int) error {
	pds := make([][]bool, p)
	for a := 0; a < p; a++ {
		pds[a] = possibleDSep(m, p, a)
	}
	for i := 0; i < p; i++ {
		for j := i + 1; j < p; j++ {
			if !fciAdjacent(m, i, j) {
				continue
			}
			removed, err := trySeparate(data, m, sepset, ci, alpha, maxCond, n, p, i, j, pds[i])
			if err != nil {
				return err
			}
			if removed {
				continue
			}
			if _, err := trySeparate(data, m, sepset, ci, alpha, maxCond, n, p, i, j, pds[j]); err != nil {
				return err
			}
		}
	}
	return nil
}

// possibleDSep returns a mask over all nodes marking Possible-D-SEP(a) in the
// current graph. A node v is in Possible-D-SEP(a) iff there is a path a … v such
// that every consecutive triple ⟨x, y, z⟩ on it has y a collider on the path
// (x *→ y ←* z) OR ⟨x, y, z⟩ a triangle (x, z adjacent). Every neighbour of a
// qualifies (an empty interior is vacuously fine).
//
// It is computed by a breadth-first search over directed states (prev, cur): the
// state (prev, cur) may advance to (cur, next) exactly when the triple
// ⟨prev, cur, next⟩ satisfies the collider-or-triangle condition. Finite states
// (ordered pairs) make the search terminate.
func possibleDSep(m [][]Mark, p, a int) []bool {
	inPds := make([]bool, p)
	visited := make([][]bool, p)
	for i := range visited {
		visited[i] = make([]bool, p)
	}
	type state struct{ prev, cur int }
	var queue []state
	for x := 0; x < p; x++ {
		if x != a && fciAdjacent(m, a, x) {
			inPds[x] = true
			if !visited[a][x] {
				visited[a][x] = true
				queue = append(queue, state{a, x})
			}
		}
	}
	for len(queue) > 0 {
		s := queue[0]
		queue = queue[1:]
		prev, cur := s.prev, s.cur
		for next := 0; next < p; next++ {
			if next == cur || next == prev || !fciAdjacent(m, cur, next) {
				continue
			}
			// cur is a collider on prev-cur-next: arrowheads at cur from both sides.
			collider := isArrow(m, cur, prev) && isArrow(m, cur, next)
			triangle := fciAdjacent(m, prev, next)
			if !collider && !triangle {
				continue
			}
			inPds[next] = true
			if !visited[cur][next] {
				visited[cur][next] = true
				queue = append(queue, state{cur, next})
			}
		}
	}
	inPds[a] = false
	return inPds
}

// trySeparate searches subsets of the Possible-D-SEP mask (excluding i and j) for
// a set that renders i and j independent. On success it deletes the edge, records
// the separating set on both sides, and returns removed=true. Subset sizes honour
// the same honest small-sample cap as the skeleton (n − |S| − 3 ≥ 1) and
// MaxCondSet.
func trySeparate(data [][]float64, m [][]Mark, sepset [][][]int, ci CITest, alpha float64, maxCond, n, p, i, j int, pdsMask []bool) (bool, error) {
	var cand []int
	for v := 0; v < p; v++ {
		if pdsMask[v] && v != i && v != j {
			cand = append(cand, v)
		}
	}
	levelMax := n - 4 // n - L - 3 >= 1  =>  L <= n-4
	if maxCond > 0 && maxCond < levelMax {
		levelMax = maxCond
	}

	removed := false
	var outErr error
	for level := 0; level <= len(cand) && level <= levelMax && !removed; level++ {
		forEachCombination(len(cand), level, func(sel []int) bool {
			s := make([]int, level)
			for t, x := range sel {
				s[t] = cand[x]
			}
			pval, err := ci(data, i, j, s)
			if err != nil {
				outErr = err
				return false
			}
			if pval > alpha {
				m[i][j] = NoMark
				m[j][i] = NoMark
				cp := append([]int(nil), s...)
				sepset[i][j] = cp
				sepset[j][i] = cp
				removed = true
				return false
			}
			return true
		})
		if outErr != nil {
			return false, outErr
		}
	}
	return removed, nil
}

// resetToCircles restores every surviving edge to o-o, discarding the provisional
// collider arrowheads so the re-orientation phase starts clean.
func resetToCircles(m [][]Mark, p int) {
	for i := 0; i < p; i++ {
		for j := 0; j < p; j++ {
			if i != j && m[i][j] != NoMark {
				m[i][j] = Circle
			}
		}
	}
}

// --- Zhang orientation rules (R1–R4, R8–R10) -----------------------------

// applyFCIRules applies Zhang's orientation rules to closure. R5, R6 and R7 — the
// selection-bias rules, the only ones that produce undirected (—) edges — run only
// when selection is true; with selection false they are omitted by design (the
// no-selection-bias scope) and no — edge is ever produced. Each rule scans every
// applicable tuple and reports whether it changed a mark; the loop repeats until a
// full pass changes nothing.
func applyFCIRules(m [][]Mark, sepset [][][]int, p int, selection bool) {
	for changed := true; changed; {
		changed = false
		if ruleR1(m, p) {
			changed = true
		}
		if ruleR2(m, p) {
			changed = true
		}
		if ruleR3(m, p) {
			changed = true
		}
		if ruleR4(m, sepset, p) {
			changed = true
		}
		if selection {
			if ruleR5(m, p) {
				changed = true
			}
			if ruleR6(m, p) {
				changed = true
			}
			if ruleR7(m, p) {
				changed = true
			}
		}
		if ruleR8(m, p) {
			changed = true
		}
		if ruleR9(m, p) {
			changed = true
		}
		if ruleR10(m, p) {
			changed = true
		}
	}
}

// --- selection-bias mark helpers (R5–R7) ---------------------------------

// isUndirected reports whether a-b is an undirected (—, tail–tail) edge: a tail at
// each end. These arise only under selection bias (R5), and R6/R7 consume them.
func isUndirected(m [][]Mark, a, b int) bool {
	return fciAdjacent(m, a, b) && isTail(m, a, b) && isTail(m, b, a)
}

// isCircleBoth reports whether a-b is a wholly undetermined o—o edge (a circle at
// each end) — the only edges R5 threads its circle path through and the only edge
// it re-orients to undirected.
func isCircleBoth(m [][]Mark, a, b int) bool {
	return isCircle(m, a, b) && isCircle(m, b, a)
}

// setUndirected orients a-b as undirected (—) by placing a tail at both ends,
// returning whether anything changed. Only ever called on an existing o—o edge.
func setUndirected(m [][]Mark, a, b int) bool {
	changed := orientEdge(m, a, b, Tail) // tail at a
	if orientEdge(m, b, a, Tail) {       // tail at b
		changed = true
	}
	return changed
}

// ruleR1: if a *→ b and b o—* c with a, c non-adjacent, orient b → c (the
// arrowhead at b would otherwise create an unshielded collider that the skeleton
// did not license).
func ruleR1(m [][]Mark, p int) bool {
	changed := false
	for a := 0; a < p; a++ {
		for b := 0; b < p; b++ {
			if a == b || !isArrow(m, b, a) { // need a *→ b (arrowhead at b)
				continue
			}
			for c := 0; c < p; c++ {
				if c == a || c == b || !fciAdjacent(m, b, c) {
					continue
				}
				if !isCircle(m, b, c) { // need circle at b on edge b-c
					continue
				}
				if fciAdjacent(m, a, c) {
					continue
				}
				if orientEdge(m, c, b, Arrow) { // arrowhead at c
					changed = true
				}
				if orientEdge(m, b, c, Tail) { // tail at b
					changed = true
				}
			}
		}
	}
	return changed
}

// ruleR2: if a → b *→ c or a *→ b → c, and a *—o c, orient the arrowhead a *→ c
// (else the edge a-c would have to be a tail, forming a directed cycle).
func ruleR2(m [][]Mark, p int) bool {
	changed := false
	for a := 0; a < p; a++ {
		for c := 0; c < p; c++ {
			if a == c || !fciAdjacent(m, a, c) || !isCircle(m, c, a) {
				continue // need a *—o c (circle at c)
			}
			for b := 0; b < p; b++ {
				if b == a || b == c {
					continue
				}
				if (isDirected(m, a, b) && isArrow(m, c, b)) ||
					(isArrow(m, b, a) && isDirected(m, b, c)) {
					if orientEdge(m, c, a, Arrow) { // arrowhead at c
						changed = true
					}
					break
				}
			}
		}
	}
	return changed
}

// ruleR3: if a *→ b ←* c, a *—o d o—* c with a, c non-adjacent, and d *—o b,
// orient the arrowhead d *→ b.
func ruleR3(m [][]Mark, p int) bool {
	changed := false
	for b := 0; b < p; b++ {
		for d := 0; d < p; d++ {
			if d == b || !fciAdjacent(m, d, b) || !isCircle(m, b, d) {
				continue // need d *—o b (circle at b on edge d-b)
			}
			for a := 0; a < p; a++ {
				if a == b || a == d || !isArrow(m, b, a) { // a *→ b
					continue
				}
				if !fciAdjacent(m, a, d) || !isCircle(m, d, a) { // a *—o d
					continue
				}
				for c := 0; c < p; c++ {
					if c == a || c == b || c == d {
						continue
					}
					if !isArrow(m, b, c) { // c *→ b
						continue
					}
					if !fciAdjacent(m, c, d) || !isCircle(m, d, c) { // c *—o d
						continue
					}
					if fciAdjacent(m, a, c) { // a, c must be non-adjacent
						continue
					}
					if orientEdge(m, b, d, Arrow) { // arrowhead at b
						changed = true
					}
				}
			}
		}
	}
	return changed
}

// ruleR4: the discriminating-path rule. If there is a discriminating path
// ⟨θ, …, a, b, c⟩ for b and b o—* c, then orient b → c if b is in the separating
// set of (θ, c), else orient the collider a ↔ b ↔ c.
func ruleR4(m [][]Mark, sepset [][][]int, p int) bool {
	changed := false
	for b := 0; b < p; b++ {
		for c := 0; c < p; c++ {
			if b == c || !fciAdjacent(m, b, c) || !isCircle(m, b, c) {
				continue // need b o—* c (circle at b) to orient
			}
			found, _, theta := findDiscriminatingPath(m, p, b, c)
			if !found {
				continue
			}
			if contains(sepset[theta][c], b) {
				if orientEdge(m, c, b, Arrow) { // b → c
					changed = true
				}
				if orientEdge(m, b, c, Tail) {
					changed = true
				}
			} else {
				if orientEdge(m, c, b, Arrow) { // b ↔ c (a ↔ b already holds)
					changed = true
				}
				if orientEdge(m, b, c, Arrow) {
					changed = true
				}
			}
		}
	}
	return changed
}

// findDiscriminatingPath searches for a discriminating path for b ending at the
// edge b-c. The node adjacent to b on the path (call it a) must satisfy a ↔ b and
// a → c; walking back from a, every interior node must be a collider on the path
// AND a parent of c, until reaching a node theta not adjacent to c. It returns
// (found, a, theta).
func findDiscriminatingPath(m [][]Mark, p, b, c int) (bool, int, int) {
	for a := 0; a < p; a++ {
		if a == b || a == c {
			continue
		}
		if !(isArrow(m, b, a) && isArrow(m, a, b)) { // a ↔ b
			continue
		}
		if !(isArrow(m, c, a) && isTail(m, a, c)) { // a → c
			continue
		}
		visited := make([]bool, p)
		visited[b] = true
		visited[c] = true
		if ok, theta := dpExtend(m, p, a, b, c, visited); ok {
			return true, a, theta
		}
	}
	return false, -1, -1
}

// dpExtend walks the discriminating path backward. cur is the current interior
// node (a collider and a parent of c); succ is its successor toward b. It seeks a
// predecessor d with an arrowhead into cur (so cur is a collider between d and
// succ). If d is not adjacent to c, d is the path start theta and the path is
// discriminating; otherwise d must itself be a parent of c and the walk recurses.
func dpExtend(m [][]Mark, p, cur, succ, c int, visited []bool) (bool, int) {
	if !isArrow(m, cur, succ) { // cur must be a collider: arrowhead at cur from succ
		return false, -1
	}
	visited[cur] = true
	defer func() { visited[cur] = false }()
	for d := 0; d < p; d++ {
		if visited[d] || d == cur {
			continue
		}
		if !fciAdjacent(m, d, cur) || !isArrow(m, cur, d) { // d *→ cur
			continue
		}
		if !fciAdjacent(m, d, c) {
			return true, d // theta found: discriminating path complete
		}
		if isDirected(m, d, c) { // d must be a parent of c to continue
			if ok, theta := dpExtend(m, p, d, cur, c, visited); ok {
				return true, theta
			}
		}
	}
	return false, -1
}

// ruleR5: for every a o—o b, if there is an uncovered CIRCLE path
// ⟨a, γ, …, θ, b⟩ (every edge o—o, every consecutive triple unshielded) with a and
// θ non-adjacent and b and γ non-adjacent, orient a—b and every edge on that path
// as undirected (—, tail at both ends). The two endpoint conditions make the whole
// cycle a—γ—…—θ—b—a uncovered, which is what forces the tails: any arrowhead on it
// would create an unshielded collider the skeleton did not license. This is the
// sole source of tail–tail edges and fires only under selection bias.
func ruleR5(m [][]Mark, p int) bool {
	changed := false
	for a := 0; a < p; a++ {
		for b := a + 1; b < p; b++ {
			if !isCircleBoth(m, a, b) { // need a o—o b
				continue
			}
			path := r5CirclePath(m, p, a, b)
			if path == nil {
				continue
			}
			if setUndirected(m, a, b) {
				changed = true
			}
			for k := 0; k+1 < len(path); k++ {
				if setUndirected(m, path[k], path[k+1]) {
					changed = true
				}
			}
		}
	}
	return changed
}

// r5CirclePath returns an uncovered circle path witnessing R5 for the edge a o—o b,
// as a node slice from a to b, or nil if none exists. Every edge on it is o—o and
// every consecutive triple ⟨x, y, z⟩ is unshielded (x, z non-adjacent); the second
// node is non-adjacent to b and the second-to-last is non-adjacent to a (R5's
// endpoint conditions). The direct a–b edge is never used as a step — the path has
// an interior. Searched depth-first in index order, so the witness is deterministic.
func r5CirclePath(m [][]Mark, p, a, b int) []int {
	visited := make([]bool, p)
	visited[a] = true
	var path []int
	var dfs func(prev, cur int) []int
	dfs = func(prev, cur int) []int {
		for next := 0; next < p; next++ {
			if visited[next] || next == cur || next == prev {
				continue
			}
			if !isCircleBoth(m, cur, next) {
				continue
			}
			if fciAdjacent(m, prev, next) { // uncovered: prev and next non-adjacent
				continue
			}
			if next == b {
				if fciAdjacent(m, a, cur) { // θ = cur must be non-adjacent to a
					continue
				}
				return append(append([]int(nil), path...), b)
			}
			visited[next] = true
			path = append(path, next)
			if res := dfs(cur, next); res != nil {
				return res
			}
			path = path[:len(path)-1]
			visited[next] = false
		}
		return nil
	}
	// First step a → γ: γ must be non-adjacent to b (R5's other endpoint condition).
	for g := 0; g < p; g++ {
		if g == a || g == b || !isCircleBoth(m, a, g) {
			continue
		}
		if fciAdjacent(m, b, g) {
			continue
		}
		visited[g] = true
		path = []int{a, g}
		if res := dfs(a, g); res != nil {
			return res
		}
		visited[g] = false
	}
	return nil
}

// ruleR6: if a — b o—* c (a—b undirected), the mark at b on b-c cannot be an
// arrowhead, so orient b —* c (place a tail at b). c may or may not be adjacent to
// a.
func ruleR6(m [][]Mark, p int) bool {
	changed := false
	for b := 0; b < p; b++ {
		for c := 0; c < p; c++ {
			if b == c || !fciAdjacent(m, b, c) || !isCircle(m, b, c) {
				continue // need b o—* c (circle at b)
			}
			for a := 0; a < p; a++ {
				if a == b || a == c {
					continue
				}
				if isUndirected(m, a, b) { // a — b
					if orientEdge(m, b, c, Tail) { // tail at b: b —* c
						changed = true
					}
					break
				}
			}
		}
	}
	return changed
}

// ruleR7: if a —o b o—* c with a and c non-adjacent, orient b —* c (place a tail at
// b). "a —o b" is a tail at a and a circle at b.
func ruleR7(m [][]Mark, p int) bool {
	changed := false
	for b := 0; b < p; b++ {
		for c := 0; c < p; c++ {
			if b == c || !fciAdjacent(m, b, c) || !isCircle(m, b, c) {
				continue // need b o—* c (circle at b)
			}
			for a := 0; a < p; a++ {
				if a == b || a == c {
					continue
				}
				if !fciAdjacent(m, a, b) || !isTail(m, a, b) || !isCircle(m, b, a) {
					continue // need a —o b (tail at a, circle at b)
				}
				if fciAdjacent(m, a, c) { // a and c must be non-adjacent
					continue
				}
				if orientEdge(m, b, c, Tail) { // tail at b: b —* c
					changed = true
				}
				break
			}
		}
	}
	return changed
}

// ruleR8: if a → b → c or a —o b → c, and a o→ c, orient a → c (add the tail at
// a; the arrowhead at c already holds).
func ruleR8(m [][]Mark, p int) bool {
	changed := false
	for a := 0; a < p; a++ {
		for c := 0; c < p; c++ {
			if a == c || !fciAdjacent(m, a, c) {
				continue
			}
			if !(isArrow(m, c, a) && isCircle(m, a, c)) { // a o→ c
				continue
			}
			for b := 0; b < p; b++ {
				if b == a || b == c {
					continue
				}
				if !isDirected(m, b, c) { // b → c
					continue
				}
				aToB := isDirected(m, a, b)                      // a → b
				aCircToB := isTail(m, a, b) && isCircle(m, b, a) // a —o b
				if aToB || aCircToB {
					if orientEdge(m, a, c, Tail) { // tail at a: a → c
						changed = true
					}
					break
				}
			}
		}
	}
	return changed
}

// ruleR9: if a o→ c and there is an uncovered potentially-directed path from a to
// c whose second node is not adjacent to c, orient a → c (add the tail at a).
func ruleR9(m [][]Mark, p int) bool {
	changed := false
	for a := 0; a < p; a++ {
		for c := 0; c < p; c++ {
			if a == c || !fciAdjacent(m, a, c) {
				continue
			}
			if !(isArrow(m, c, a) && isCircle(m, a, c)) { // a o→ c
				continue
			}
			if r9HasPath(m, p, a, c) {
				if orientEdge(m, a, c, Tail) {
					changed = true
				}
			}
		}
	}
	return changed
}

// r9HasPath reports whether some neighbour mu of a (non-adjacent to c, reachable
// as a potentially-directed first step) begins an uncovered potentially-directed
// path from a to c.
func r9HasPath(m [][]Mark, p, a, c int) bool {
	for mu := 0; mu < p; mu++ {
		if mu == a || mu == c {
			continue
		}
		if !fciAdjacent(m, a, mu) || !potentiallyDirected(m, a, mu) {
			continue
		}
		if fciAdjacent(m, mu, c) { // R9 needs the second node non-adjacent to c
			continue
		}
		if uncoveredPDReachable(m, p, a, mu, c) {
			return true
		}
	}
	return false
}

// ruleR10: if a o→ c, with distinct parents b, d of c (b → c ← d), and there are
// two uncovered potentially-directed paths out of a — one reaching b, one
// reaching d — whose first nodes are distinct and non-adjacent, orient a → c.
func ruleR10(m [][]Mark, p int) bool {
	changed := false
	for a := 0; a < p; a++ {
		for c := 0; c < p; c++ {
			if a == c || !fciAdjacent(m, a, c) {
				continue
			}
			if !(isArrow(m, c, a) && isCircle(m, a, c)) { // a o→ c
				continue
			}
			if r10HasPaths(m, p, a, c) {
				if orientEdge(m, a, c, Tail) {
					changed = true
				}
			}
		}
	}
	return changed
}

// r10HasPaths implements R10's antecedent search: over every ordered pair of
// distinct parents (b, d) of c, it collects the first steps of uncovered
// potentially-directed paths from a to b and to d, and reports whether two of
// them are distinct and non-adjacent.
func r10HasPaths(m [][]Mark, p, a, c int) bool {
	var parents []int
	for x := 0; x < p; x++ {
		if x != c && x != a && isDirected(m, x, c) {
			parents = append(parents, x)
		}
	}
	for bi := 0; bi < len(parents); bi++ {
		for di := 0; di < len(parents); di++ {
			b, d := parents[bi], parents[di]
			if b == d {
				continue
			}
			mus := pdFirstSteps(m, p, a, b)
			oms := pdFirstSteps(m, p, a, d)
			for _, mu := range mus {
				for _, om := range oms {
					if mu != om && !fciAdjacent(m, mu, om) {
						return true
					}
				}
			}
		}
	}
	return false
}

// pdFirstSteps returns the neighbours mu of a such that a potentially-directed
// first step a → mu begins an uncovered potentially-directed path from a to
// target.
func pdFirstSteps(m [][]Mark, p, a, target int) []int {
	var res []int
	for mu := 0; mu < p; mu++ {
		if mu == a {
			continue
		}
		if !fciAdjacent(m, a, mu) || !potentiallyDirected(m, a, mu) {
			continue
		}
		if uncoveredPDReachable(m, p, a, mu, target) {
			res = append(res, mu)
		}
	}
	return res
}

// uncoveredPDReachable reports whether there is an uncovered
// potentially-directed path from src to dst whose first vertex after src is
// `first`. "Potentially directed" means every edge can be oriented forward along
// the path; "uncovered" means every consecutive triple ⟨x, y, z⟩ has x and z
// non-adjacent. If first == dst the single potentially-directed edge src → dst
// qualifies.
func uncoveredPDReachable(m [][]Mark, p, src, first, dst int) bool {
	if !fciAdjacent(m, src, first) || !potentiallyDirected(m, src, first) {
		return false
	}
	if first == dst {
		return true
	}
	visited := make([]bool, p)
	visited[src] = true
	var dfs func(prev, cur int) bool
	dfs = func(prev, cur int) bool {
		if cur == dst {
			return true
		}
		visited[cur] = true
		defer func() { visited[cur] = false }()
		for next := 0; next < p; next++ {
			if visited[next] || next == cur || next == prev {
				continue
			}
			if !fciAdjacent(m, cur, next) || !potentiallyDirected(m, cur, next) {
				continue
			}
			if fciAdjacent(m, prev, next) { // uncovered: prev and next must be non-adjacent
				continue
			}
			if dfs(cur, next) {
				return true
			}
		}
		return false
	}
	return dfs(src, first)
}
