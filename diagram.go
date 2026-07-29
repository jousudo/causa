package causa

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

// Diagram errors.
var (
	// ErrEdgeOutOfRange is returned when an edge (or a query set) names a vertex
	// index outside 0..n-1.
	ErrEdgeOutOfRange = errors.New("causa: edge references a vertex index out of range")
	// ErrSelfLoop is returned when a directed or bidirected edge connects a vertex
	// to itself.
	ErrSelfLoop = errors.New("causa: self-loop edge")
	// ErrEmptyOutcome is returned by Identify when the outcome set Y is empty.
	ErrEmptyOutcome = errors.New("causa: outcome set Y must be non-empty")
	// ErrOverlappingQuery is returned by Identify when the intervention set X and
	// the outcome set Y share a variable.
	ErrOverlappingQuery = errors.New("causa: intervention set X and outcome set Y overlap")
	// ErrDuplicateQueryVar is returned by Identify when X or Y lists the same
	// variable twice.
	ErrDuplicateQueryVar = errors.New("causa: duplicate variable in a query set")
	// ErrSelectionBiasUnsupported is returned by IdentifyPAG / IdentifyConditionalPAG
	// (and their PAG methods) when the input PAG carries an undirected (—, tail–tail)
	// edge. Such an edge records selection bias, which FCI produces only in its
	// opt-in SelectionBias mode. The IDP/CIDP identification algorithms are sound and
	// complete only under the no-selection-bias scope, so rather than run them on a
	// selection PAG and return a silently wrong estimand, they refuse. Identification
	// of causal effects UNDER selection bias is a separate problem, not yet supported.
	ErrSelectionBiasUnsupported = errors.New("causa: identification over a PAG with selection bias (an undirected — edge) is not supported")
)

// Diagram is a causal diagram — an acyclic directed mixed graph (ADMG), the
// semi-Markovian model the ID algorithm identifies over. It has two edge kinds:
//
//   - a DIRECTED edge i → j (i is a direct cause / parent of j), and
//   - a BIDIRECTED edge i ↔ j, standing for an unobserved common cause (a latent
//     confounder) of i and j.
//
// The directed part must be acyclic. This is the input a user asserts as the
// known structure; unlike a PAG (the output of FCI) it is a single, fully
// specified graph, not an equivalence class — identification is defined relative
// to one causal diagram.
type Diagram struct {
	names []string
	// dir[i][j] == true iff there is a directed edge i → j (i parent of j).
	dir [][]bool
	// bi[i][j] == true iff there is a bidirected edge i ↔ j; kept symmetric.
	bi [][]bool
}

// NewDiagram builds a causal diagram over n vertices from its directed and
// bidirected edges, each given as an index pair. A directed pair {i, j} means
// i → j; a bidirected pair {i, j} means i ↔ j (order irrelevant, stored
// symmetric). names labels the vertices: when non-nil it fixes n = len(names);
// when nil, n is inferred as one past the largest index used by any edge and the
// vertices are named V0, V1, … (a diagram with no edges and nil names is empty
// and returns ErrTooFewVariables).
//
// Errors: ErrEdgeOutOfRange (an index ≥ n or < 0), ErrSelfLoop (i == j),
// ErrNameCount (names given but not length n), ErrCyclic (the directed part has a
// cycle), and ErrTooFewVariables (n == 0).
func NewDiagram(names []string, directed, bidirected [][2]int) (*Diagram, error) {
	n := 0
	if names != nil {
		n = len(names)
	} else {
		for _, e := range append(append([][2]int(nil), directed...), bidirected...) {
			for _, v := range e {
				if v+1 > n {
					n = v + 1
				}
			}
		}
	}
	if n == 0 {
		return nil, ErrTooFewVariables
	}
	nm, err := resolveNames(names, n)
	if err != nil {
		return nil, err
	}

	dir := make([][]bool, n)
	bi := make([][]bool, n)
	for i := range dir {
		dir[i] = make([]bool, n)
		bi[i] = make([]bool, n)
	}
	inRange := func(v int) bool { return v >= 0 && v < n }
	for _, e := range directed {
		if !inRange(e[0]) || !inRange(e[1]) {
			return nil, ErrEdgeOutOfRange
		}
		if e[0] == e[1] {
			return nil, ErrSelfLoop
		}
		dir[e[0]][e[1]] = true
	}
	for _, e := range bidirected {
		if !inRange(e[0]) || !inRange(e[1]) {
			return nil, ErrEdgeOutOfRange
		}
		if e[0] == e[1] {
			return nil, ErrSelfLoop
		}
		bi[e[0]][e[1]] = true
		bi[e[1]][e[0]] = true
	}

	full := make([]bool, n)
	for i := range full {
		full[i] = true
	}
	if _, ok := topoSub(dir, full); !ok {
		return nil, ErrCyclic
	}
	return &Diagram{names: nm, dir: dir, bi: bi}, nil
}

// Order returns the number of vertices.
func (g *Diagram) Order() int { return len(g.names) }

// Nodes returns a copy of the vertex names, indexed as in the diagram.
func (g *Diagram) Nodes() []string { return append([]string(nil), g.names...) }

// HasDirected reports whether there is a directed edge i → j.
func (g *Diagram) HasDirected(i, j int) bool { return g.dir[i][j] }

// HasBidirected reports whether there is a bidirected edge i ↔ j.
func (g *Diagram) HasBidirected(i, j int) bool { return g.bi[i][j] }

// Parents returns the directed parents of j (the i with i → j), ascending.
func (g *Diagram) Parents(j int) []int {
	var ps []int
	for i := 0; i < len(g.names); i++ {
		if g.dir[i][j] {
			ps = append(ps, i)
		}
	}
	return ps
}

// String renders the diagram deterministically: a header, then the directed
// edges (a -> b) and the bidirected edges (a <-> b, each once with a < b), sorted.
func (g *Diagram) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "Diagram with %d nodes %v:", len(g.names), g.names)
	n := len(g.names)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if g.dir[i][j] {
				fmt.Fprintf(&b, "\n  %s -> %s", g.names[i], g.names[j])
			}
		}
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if g.bi[i][j] {
				fmt.Fprintf(&b, "\n  %s <-> %s", g.names[i], g.names[j])
			}
		}
	}
	return b.String()
}

// --- vertex-set helpers (masks over 0..n-1) ------------------------------

// maskFromSlice builds a mask from explicit indices.
func maskFromSlice(n int, idx []int) []bool {
	m := make([]bool, n)
	for _, v := range idx {
		m[v] = true
	}
	return m
}

func maskToSorted(m []bool) []int {
	var out []int
	for i, on := range m {
		if on {
			out = append(out, i)
		}
	}
	return out
}

func maskEmpty(m []bool) bool {
	for _, on := range m {
		if on {
			return false
		}
	}
	return true
}

func maskEqual(a, b []bool) bool {
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// maskProperSubset reports whether a ⊊ b.
func maskProperSubset(a, b []bool) bool {
	sub, strict := true, false
	for i := range a {
		if a[i] && !b[i] {
			sub = false
		}
		if b[i] && !a[i] {
			strict = true
		}
	}
	return sub && strict
}

func maskUnion(a, b []bool) []bool {
	out := make([]bool, len(a))
	for i := range a {
		out[i] = a[i] || b[i]
	}
	return out
}

func maskIntersect(a, b []bool) []bool {
	out := make([]bool, len(a))
	for i := range a {
		out[i] = a[i] && b[i]
	}
	return out
}

// maskDiff returns a \ b.
func maskDiff(a, b []bool) []bool {
	out := make([]bool, len(a))
	for i := range a {
		out[i] = a[i] && !b[i]
	}
	return out
}

// --- graph algebra used by the ID algorithm ------------------------------

// ancestors returns the mask of ancestors of `targets` within the vertex subset
// `sub`, following directed edges and INCLUDING the targets themselves. When
// cutInto is non-nil, edges pointing INTO a vertex marked in cutInto are ignored
// — this realises the manipulated graph G_{\overline X} (all edges into X
// removed) used to compute ancestors "in the do-graph". Only vertices in sub are
// ever traversed or returned.
func (g *Diagram) ancestors(sub, targets, cutInto []bool) []bool {
	n := len(g.names)
	anc := make([]bool, n)
	var stack []int
	for v := 0; v < n; v++ {
		if sub[v] && targets[v] {
			anc[v] = true
			stack = append(stack, v)
		}
	}
	for len(stack) > 0 {
		v := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if cutInto != nil && cutInto[v] {
			continue // edges into v are removed: v has no parents here
		}
		for p := 0; p < n; p++ {
			if sub[p] && g.dir[p][v] && !anc[p] {
				anc[p] = true
				stack = append(stack, p)
			}
		}
	}
	return anc
}

// cComponents partitions the vertex subset `sub` into its confounded components
// (C-components / districts): maximal sets connected by bidirected edges lying
// within sub. A vertex with no bidirected neighbour in sub is its own singleton
// component. Components are returned as masks, ordered by their smallest member
// (deterministic).
func (g *Diagram) cComponents(sub []bool) [][]bool {
	n := len(g.names)
	comp := make([]int, n)
	for i := range comp {
		comp[i] = -1
	}
	var comps [][]bool
	for v := 0; v < n; v++ {
		if !sub[v] || comp[v] != -1 {
			continue
		}
		// BFS the bidirected-connected component of v within sub.
		mask := make([]bool, n)
		var queue []int
		comp[v] = len(comps)
		mask[v] = true
		queue = append(queue, v)
		for len(queue) > 0 {
			u := queue[0]
			queue = queue[1:]
			for w := 0; w < n; w++ {
				if sub[w] && comp[w] == -1 && g.bi[u][w] {
					comp[w] = comp[v]
					mask[w] = true
					queue = append(queue, w)
				}
			}
		}
		comps = append(comps, mask)
	}
	return comps
}

// topoSub returns a deterministic topological order (as vertex indices, most
// exogenous first) of the directed subgraph induced on the vertex subset `sub`,
// ties broken to the lowest index. ok is false iff that subgraph has a directed
// cycle. dir is the directed-adjacency matrix (dir[i][j] == i → j).
func topoSub(dir [][]bool, sub []bool) (order []int, ok bool) {
	n := len(dir)
	indeg := make([]int, n)
	for j := 0; j < n; j++ {
		if !sub[j] {
			continue
		}
		for i := 0; i < n; i++ {
			if sub[i] && dir[i][j] {
				indeg[j]++
			}
		}
	}
	done := make([]bool, n)
	total := 0
	for v := 0; v < n; v++ {
		if sub[v] {
			total++
		}
	}
	for len(order) < total {
		next := -1
		for v := 0; v < n; v++ {
			if sub[v] && !done[v] && indeg[v] == 0 {
				next = v
				break
			}
		}
		if next == -1 {
			return nil, false
		}
		done[next] = true
		order = append(order, next)
		for j := 0; j < n; j++ {
			if sub[j] && !done[j] && dir[next][j] {
				indeg[j]--
			}
		}
	}
	return order, true
}

// sortedEqual reports whether two int slices hold the same set (helper for tests
// and hedge comparison).
func sortedEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	x := append([]int(nil), a...)
	y := append([]int(nil), b...)
	sort.Ints(x)
	sort.Ints(y)
	for i := range x {
		if x[i] != y[i] {
			return false
		}
	}
	return true
}
