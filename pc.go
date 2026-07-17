package causa

import (
	"errors"
	"fmt"
	"strings"
)

// Errors returned by PCStable. ErrNonFinite and ErrSingular are shared with the
// rest of the package; the values below are specific to causal discovery.
var (
	// ErrTooFewVariables is returned when fewer than two variables are supplied:
	// there is nothing to discover a relationship between.
	ErrTooFewVariables = errors.New("causa: need at least two variables for causal discovery")

	// ErrUnequalLengths is returned when the variables do not all share the same
	// number of samples; the data must be a rectangular n×p panel.
	ErrUnequalLengths = errors.New("causa: variables have differing sample lengths")

	// ErrTooFewSamples is returned when there are too few samples to run even the
	// unconditional (level-0) independence test, which needs n − 3 ≥ 1, i.e.
	// n ≥ 4.
	ErrTooFewSamples = errors.New("causa: too few samples for causal discovery (need n >= 4)")

	// ErrNameCount is returned when a non-nil names slice does not have exactly
	// one entry per variable.
	ErrNameCount = errors.New("causa: number of names does not match number of variables")

	// ErrInvalidAlpha is returned when the configured significance level is not
	// strictly between 0 and 1.
	ErrInvalidAlpha = errors.New("causa: alpha must satisfy 0 < alpha < 1")
)

// PCOptions configures PCStable. A nil *PCOptions, or a zero-valued field within
// it, selects the documented default for that setting.
type PCOptions struct {
	// Alpha is the significance level of the conditional-independence test: an
	// edge is deleted when a test reports a p-value greater than Alpha. Zero
	// selects the default of 0.05; any other value must satisfy 0 < Alpha < 1 or
	// PCStable returns ErrInvalidAlpha. Larger Alpha yields sparser graphs.
	Alpha float64

	// MaxCondSet caps the size of the conditioning sets the skeleton search will
	// consider. Zero (the default) means unbounded — the search grows the
	// conditioning set until no adjacency set is large enough or the sample guard
	// stops it. A positive value trades completeness for speed on dense graphs.
	MaxCondSet int

	// CITest is the conditional-independence test used to thin the skeleton. Nil
	// selects FisherZTest (the linear-Gaussian default). See CITest for the
	// contract a custom test must satisfy.
	CITest CITest
}

// Edge is a directed or undirected edge between two variables, identified by
// their indices into the CPDAG's node list. For a directed edge From is the tail
// and To is the arrowhead (From → To). For an undirected edge the two endpoints
// are reported in ascending index order (From < To) and carry no direction.
type Edge struct {
	From int
	To   int
}

// CPDAG is the output of PCStable: a completed partially directed acyclic graph
// representing a Markov equivalence class of DAGs.
//
// It is NOT a single causal DAG. Constraint-based discovery identifies structure
// only up to Markov equivalence: a directed edge i → j is oriented because every
// DAG consistent with the data agrees on that arrowhead (it is compelled), while
// an undirected edge i — j is reversible — some equivalent DAG orients it each
// way. Reading an undirected edge as "no direction known", not "no causal link",
// is essential to using the result honestly.
type CPDAG struct {
	names []string
	// adj[i][j] means there is a mark at j originating from i. An edge exists
	// between i and j iff adj[i][j] || adj[j][i]; it is directed i → j iff
	// adj[i][j] && !adj[j][i], and undirected iff adj[i][j] && adj[j][i].
	adj [][]bool
}

// Order returns the number of variables (nodes) in the graph.
func (g *CPDAG) Order() int { return len(g.names) }

// Nodes returns a copy of the variable names, indexed as in the input data.
func (g *CPDAG) Nodes() []string { return append([]string(nil), g.names...) }

// Adjacent reports whether variables i and j share an edge of either kind.
func (g *CPDAG) Adjacent(i, j int) bool { return g.adj[i][j] || g.adj[j][i] }

// Directed reports whether the edge between i and j is oriented i → j.
func (g *CPDAG) Directed(i, j int) bool { return g.adj[i][j] && !g.adj[j][i] }

// Undirected reports whether i and j share an undirected edge i — j.
func (g *CPDAG) Undirected(i, j int) bool { return g.adj[i][j] && g.adj[j][i] }

// DirectedEdges returns the compelled (directed) edges i → j in ascending
// (From, To) order.
func (g *CPDAG) DirectedEdges() []Edge {
	var edges []Edge
	p := len(g.adj)
	for i := 0; i < p; i++ {
		for j := 0; j < p; j++ {
			if i != j && g.adj[i][j] && !g.adj[j][i] {
				edges = append(edges, Edge{From: i, To: j})
			}
		}
	}
	return edges
}

// UndirectedEdges returns the reversible (undirected) edges, each reported once
// with From < To, in ascending (From, To) order.
func (g *CPDAG) UndirectedEdges() []Edge {
	var edges []Edge
	p := len(g.adj)
	for i := 0; i < p; i++ {
		for j := i + 1; j < p; j++ {
			if g.adj[i][j] && g.adj[j][i] {
				edges = append(edges, Edge{From: i, To: j})
			}
		}
	}
	return edges
}

// String renders the CPDAG as a human-readable, deterministic multi-line summary:
// a header with the node count and names, then the directed edges (a -> b) and
// the undirected edges (a -- b), each group sorted.
func (g *CPDAG) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "CPDAG with %d nodes %v:", len(g.names), g.names)
	for _, e := range g.DirectedEdges() {
		fmt.Fprintf(&b, "\n  %s -> %s", g.names[e.From], g.names[e.To])
	}
	for _, e := range g.UndirectedEdges() {
		fmt.Fprintf(&b, "\n  %s -- %s", g.names[e.From], g.names[e.To])
	}
	return b.String()
}

// PCStable runs the order-independent PC ("PC-stable") algorithm of Colombo &
// Maathuis (2014) — the stable variant of the Spirtes–Glymour PC algorithm — and
// returns the estimated CPDAG.
//
// Input. data holds one variable per outer slice (data[v] is variable v's
// sample); every variable must have the same length n. names, if non-nil, must
// have one entry per variable; when nil, the variables are named V0, V1, … The
// data is treated as continuous linear-Gaussian observations by the default test
// (see PCOptions.CITest to override).
//
// Algorithm. Three phases:
//
//  1. Skeleton. Start from the complete undirected graph. For each level
//     ℓ = 0, 1, 2, …, snapshot every node's adjacency set BEFORE the level, then
//     for every ordered adjacent pair (i, j) test i ⫫ j given each size-ℓ subset
//     of snapshot-adj(i)\{j}; delete the edge and record a separating set on the
//     first independence found. Using the pre-level snapshot for subset
//     enumeration is exactly what makes the result independent of the order in
//     which pairs are visited — within-level deletions never shrink another
//     pair's candidate conditioning sets.
//
//  2. v-structures. For every unshielded triple i − k − j (i and j both adjacent
//     to k but not to each other), orient i → k ← j iff k is not in the
//     separating set recorded for (i, j).
//
//  3. Meek rules R1–R4 applied to closure, propagating the compelled
//     orientations without creating a new unshielded collider or a directed
//     cycle.
//
// Determinism. Every phase iterates variable indices (and enumerates subsets in
// lexicographic index order); there is no dependence on map iteration order. The
// same input always yields the same CPDAG, and — the PC-stable guarantee — the
// skeleton is invariant to the order in which the variables are presented.
//
// Termination and the sample guard. The skeleton stops growing the conditioning
// set once no adjacency set is large enough, once MaxCondSet is reached, or once
// the sample is too small for the next level: levels stop growing as soon as
// n − ℓ − 3 < 1, because Fisher's z-transform then has no residual degrees of
// freedom. This is an honest cap — on small samples the deeper conditional
// independencies simply cannot be tested, so some edges that a larger sample
// would remove may remain.
//
// Conflict policy for v-structures. Orientations are applied in a fixed
// deterministic order, and an arrowhead is never silently reversed: if orienting
// a v-structure would flip an arrowhead already placed by an earlier v-structure,
// that edge is left as it was and the conflicting arrowhead is skipped (the other
// arrowhead of the same v-structure is still applied if it does not conflict).
// Such conflicts arise only when the data violates the model's faithfulness
// assumption; on faithful data every unshielded collider is oriented consistently.
//
// Assumptions. Correctness rests on the standard constraint-based assumptions:
// causal sufficiency (no unobserved common cause of two measured variables),
// faithfulness (every conditional independence in the distribution is entailed by
// the graph), and a correct CITest (the default assumes linear-Gaussian data).
// The result is a Markov equivalence class (see CPDAG), not a unique DAG.
//
// Errors: ErrTooFewVariables (< 2 variables), ErrUnequalLengths (ragged data),
// ErrTooFewSamples (n < 4), ErrNonFinite (NaN/Inf in the data), ErrNameCount
// (names length mismatch), ErrInvalidAlpha (Alpha out of range), and any error
// surfaced by the CITest — for the default, ErrSingular from a rank-deficient
// residualization aborts the run rather than being silently ignored.
//
// Reference: Colombo & Maathuis, "Order-Independent Constraint-Based Causal
// Structure Learning", JMLR 15 (2014) 3741–3782 (PC-stable skeleton); Meek,
// "Causal Inference and Causal Explanation with Background Knowledge", UAI 1995
// (orientation rules R1–R4).
func PCStable(data [][]float64, names []string, opts *PCOptions) (*CPDAG, error) {
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
	}

	adj, sepset, err := pcSkeleton(data, ci, alpha, maxCond, n, p)
	if err != nil {
		return nil, err
	}
	orientVStructures(adj, sepset, p)
	applyMeekRules(adj, p)

	return &CPDAG{names: names, adj: adj}, nil
}

// pcSkeleton estimates the undirected skeleton with the order-independent
// PC-stable rule and returns the adjacency matrix plus the recorded separating
// sets. A CITest error aborts the search and is returned to the caller.
func pcSkeleton(data [][]float64, ci CITest, alpha float64, maxCond, n, p int) ([][]bool, [][][]int, error) {
	adj := make([][]bool, p)
	for i := range adj {
		adj[i] = make([]bool, p)
		for j := range adj[i] {
			adj[i][j] = i != j
		}
	}
	sepset := make([][][]int, p)
	for i := range sepset {
		sepset[i] = make([][]int, p)
	}

	for level := 0; ; level++ {
		if n-level-3 < 1 { // sample guard: no residual df for the transform
			break
		}
		if maxCond > 0 && level > maxCond {
			break
		}

		// Snapshot each node's adjacency BEFORE the level. Subset enumeration
		// draws only from this snapshot, so deletions made during the level do
		// not affect any other pair's candidate conditioning sets.
		snap := make([][]int, p)
		for i := 0; i < p; i++ {
			for j := 0; j < p; j++ {
				if i != j && adj[i][j] {
					snap[i] = append(snap[i], j)
				}
			}
		}

		if !anyEligiblePair(adj, snap, p, level) {
			break
		}

		for i := 0; i < p; i++ {
			for j := 0; j < p; j++ {
				if i == j || !adj[i][j] {
					continue
				}
				cand := make([]int, 0, len(snap[i]))
				for _, s := range snap[i] {
					if s != j {
						cand = append(cand, s)
					}
				}
				if len(cand) < level {
					continue
				}

				var ciErr error
				forEachCombination(len(cand), level, func(sel []int) bool {
					s := make([]int, level)
					for t, x := range sel {
						s[t] = cand[x]
					}
					pval, err := ci(data, i, j, s)
					if err != nil {
						ciErr = err
						return false
					}
					if pval > alpha {
						adj[i][j] = false
						adj[j][i] = false
						cp := append([]int(nil), s...)
						sepset[i][j] = cp
						sepset[j][i] = cp
						return false
					}
					return true
				})
				if ciErr != nil {
					return nil, nil, ciErr
				}
			}
		}
	}
	return adj, sepset, nil
}

// anyEligiblePair reports whether some ordered, currently-adjacent pair (i, j)
// still has at least level candidate conditioning variables in the snapshot,
// i.e. whether the current level can do any work.
func anyEligiblePair(adj [][]bool, snap [][]int, p, level int) bool {
	for i := 0; i < p; i++ {
		for j := 0; j < p; j++ {
			if i == j || !adj[i][j] {
				continue
			}
			cnt := len(snap[i])
			for _, s := range snap[i] {
				if s == j {
					cnt--
					break
				}
			}
			if cnt >= level {
				return true
			}
		}
	}
	return false
}

// orientVStructures orients every unshielded collider. For each unshielded triple
// i − k − j (i < j, both adjacent to k, i and j non-adjacent), it orients
// i → k ← j iff k is not in the separating set of (i, j). Arrowheads are placed
// through orientArrow, which enforces the never-reverse conflict policy.
func orientVStructures(adj [][]bool, sepset [][][]int, p int) {
	for k := 0; k < p; k++ {
		for i := 0; i < p; i++ {
			if i == k || !adjacent(adj, i, k) {
				continue
			}
			for j := i + 1; j < p; j++ {
				if j == k || !adjacent(adj, j, k) {
					continue
				}
				if adjacent(adj, i, j) { // shielded triple: not a v-structure
					continue
				}
				if !contains(sepset[i][j], k) {
					orientArrow(adj, i, k)
					orientArrow(adj, j, k)
				}
			}
		}
	}
}

// applyMeekRules applies Meek's rules R1–R4 to closure, orienting undirected
// edges without introducing a new unshielded collider or a directed cycle.
func applyMeekRules(adj [][]bool, p int) {
	for changed := true; changed; {
		changed = false
		for a := 0; a < p; a++ {
			for b := 0; b < p; b++ {
				if a == b || !undirected(adj, a, b) {
					continue
				}
				if meekR1(adj, a, b, p) || meekR2(adj, a, b, p) ||
					meekR3(adj, a, b, p) || meekR4(adj, a, b, p) {
					adj[a][b] = true
					adj[b][a] = false
					changed = true
				}
			}
		}
	}
}

// meekR1: orient a → b if some c → a with c not adjacent to b (avoids creating a
// new unshielded collider c → a → b).
func meekR1(adj [][]bool, a, b, p int) bool {
	for c := 0; c < p; c++ {
		if c != a && c != b && directed(adj, c, a) && !adjacent(adj, c, b) {
			return true
		}
	}
	return false
}

// meekR2: orient a → b if a → c → b already exists (avoids a directed cycle).
func meekR2(adj [][]bool, a, b, p int) bool {
	for c := 0; c < p; c++ {
		if c != a && c != b && directed(adj, a, c) && directed(adj, c, b) {
			return true
		}
	}
	return false
}

// meekR3: orient a → b if there are c, d with a − c, a − d undirected, c → b,
// d → b, and c, d non-adjacent.
func meekR3(adj [][]bool, a, b, p int) bool {
	for c := 0; c < p; c++ {
		if c == a || c == b || !undirected(adj, a, c) || !directed(adj, c, b) {
			continue
		}
		for d := c + 1; d < p; d++ {
			if d == a || d == b || !undirected(adj, a, d) || !directed(adj, d, b) {
				continue
			}
			if !adjacent(adj, c, d) {
				return true
			}
		}
	}
	return false
}

// meekR4: orient a → b if there are c, d with a − c, a − d undirected, d → c,
// c → b, and b, d non-adjacent.
func meekR4(adj [][]bool, a, b, p int) bool {
	for c := 0; c < p; c++ {
		if c == a || c == b || !undirected(adj, a, c) || !directed(adj, c, b) {
			continue
		}
		for d := 0; d < p; d++ {
			if d == a || d == b || d == c {
				continue
			}
			if undirected(adj, a, d) && directed(adj, d, c) && !adjacent(adj, b, d) {
				return true
			}
		}
	}
	return false
}

// orientArrow places an arrowhead at 'to' for the edge between from and to,
// orienting from → to. It never reverses an existing arrowhead: if the edge is
// already directed to → from, the request conflicts and is skipped, leaving the
// edge unchanged. Orienting an undirected edge, or reasserting an existing
// from → to, both succeed.
func orientArrow(adj [][]bool, from, to int) {
	if !adj[from][to] && !adj[to][from] {
		return // no edge to orient
	}
	if adj[to][from] && !adj[from][to] {
		return // already directed to->from: reversing is forbidden, skip
	}
	adj[from][to] = true
	adj[to][from] = false
}

// adjacent reports whether i and j share an edge of either orientation.
func adjacent(adj [][]bool, i, j int) bool { return adj[i][j] || adj[j][i] }

// directed reports whether the edge is oriented i → j.
func directed(adj [][]bool, i, j int) bool { return adj[i][j] && !adj[j][i] }

// undirected reports whether i and j share an undirected edge.
func undirected(adj [][]bool, i, j int) bool { return adj[i][j] && adj[j][i] }

// contains reports whether the sorted-or-unsorted set s holds value v.
func contains(s []int, v int) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}

// forEachCombination calls fn with each k-element subset of the index set
// {0, 1, …, n−1}, enumerated in lexicographic order, until fn returns false. The
// slice passed to fn is reused between calls and must not be retained. k == 0
// yields a single empty subset.
func forEachCombination(n, k int, fn func(sel []int) bool) {
	if k == 0 {
		fn(nil)
		return
	}
	if k > n {
		return
	}
	sel := make([]int, k)
	for i := range sel {
		sel[i] = i
	}
	for {
		if !fn(sel) {
			return
		}
		i := k - 1
		for i >= 0 && sel[i] == n-k+i {
			i--
		}
		if i < 0 {
			return
		}
		sel[i]++
		for j := i + 1; j < k; j++ {
			sel[j] = sel[j-1] + 1
		}
	}
}
