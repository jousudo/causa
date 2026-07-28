package causa

import (
	"errors"
	"fmt"
)

// SEM errors. They are sentinel values so callers can branch with errors.Is.
var (
	// ErrNotSquare is returned when a coefficient matrix is not n×n.
	ErrNotSquare = errors.New("causa: coefficient matrix is not square")
	// ErrCyclic is returned when a coefficient matrix encodes a directed cycle,
	// so it is not a structural model of an acyclic system (no causal order
	// exists).
	ErrCyclic = errors.New("causa: coefficient matrix is cyclic (no causal order)")
	// ErrUnknownVariable is returned when an intervention or observation names a
	// variable the model does not contain.
	ErrUnknownVariable = errors.New("causa: unknown variable name")
	// ErrIncompleteObservation is returned by Counterfactual when the observation
	// does not assign a value to every variable (abduction needs the full row).
	ErrIncompleteObservation = errors.New("causa: observation must give a value for every variable")
	// ErrInvalidOrder is returned by FitSEM when order is not a permutation of the
	// variable indices.
	ErrInvalidOrder = errors.New("causa: order is not a permutation of the variable indices")
)

// SEM is a linear structural equation model (a linear structural causal model),
//
//	x = c + B·x + e,
//
// with a coefficient matrix B that is acyclic — B[i][j] is the direct causal
// effect of variable j on variable i, and is non-zero only when j precedes i in
// a causal order — an intercept vector c, and mutually independent disturbances
// e with mean zero. It is the object the do-operator and counterfactual queries
// act on: given the structural equations, Intervene computes an interventional
// expectation E[X | do(·)] and Counterfactual answers a "what would have been"
// query by Pearl's abduction–action–prediction.
//
// A SEM is obtained either from a hand-specified coefficient matrix (NewSEM) or
// by estimating one from data along a known causal order (FitSEM) — the order a
// DirectLiNGAM run recovers is exactly such an input, so discovery and
// intervention compose: `FitSEM(data, r.Nodes(), r.CausalOrder())`.
//
// Scope (stated bluntly, per the library's honesty policy). This is the
// IDENTIFIED linear-SEM case: it assumes you already hold the full structural
// model — every direct effect and the acyclic structure. It does NOT solve the
// general do-calculus IDENTIFICATION problem (recovering an interventional
// distribution from an OBSERVATIONAL distribution plus a partially known graph
// with latent confounders); that — Pearl's do-calculus rules and the ID
// algorithm — remains research. What is delivered is exact for a fully specified
// linear SEM: interventions and counterfactuals reduce to forward substitution
// through the structural equations.
type SEM struct {
	names     []string
	order     []int       // causal order: order[k] is the variable at position k (most exogenous first)
	pos       []int       // pos[v] is variable v's position in order (inverse permutation)
	b         [][]float64 // b[i][j]: direct effect of x_j on x_i (non-zero only if pos[j] < pos[i])
	intercept []float64   // c[i]
}

// NewSEM builds a linear SEM x = c + B·x + e from a coefficient matrix. coef
// must be square (n×n) with coef[i][j] the direct effect of variable j on
// variable i, and must be acyclic: the edge set {j→i : coef[i][j] ≠ 0} has a
// topological order, otherwise ErrCyclic is returned. intercept may be nil (all
// zeros); otherwise it must have length n. names, if non-nil, must have length
// n (else nil defaults to V0, V1, …). Every entry must be finite.
//
// The recovered causal order is deterministic: ties in the topological sort
// break to the lowest variable index.
func NewSEM(names []string, coef [][]float64, intercept []float64) (*SEM, error) {
	n := len(coef)
	if n == 0 {
		return nil, ErrTooFewVariables
	}
	for _, row := range coef {
		if len(row) != n {
			return nil, ErrNotSquare
		}
		for _, v := range row {
			if !isFinite(v) {
				return nil, ErrNonFinite
			}
		}
	}
	nm, err := resolveNames(names, n)
	if err != nil {
		return nil, err
	}
	ic := make([]float64, n)
	if intercept != nil {
		if len(intercept) != n {
			return nil, ErrNameCount
		}
		for i, v := range intercept {
			if !isFinite(v) {
				return nil, ErrNonFinite
			}
			ic[i] = v
		}
	}

	order, ok := topoOrder(coef)
	if !ok {
		return nil, ErrCyclic
	}
	b := make([][]float64, n)
	for i := range coef {
		b[i] = append([]float64(nil), coef[i]...)
	}
	return newSEM(nm, order, b, ic), nil
}

// FitSEM estimates a linear SEM from observational data given a causal order —
// the order[k] variable is regressed by ordinary least squares on an intercept
// plus its predecessors order[0..k-1], yielding the strictly-lower-triangular
// (in the order) coefficient matrix B and the intercepts. It reuses the same
// backward-stable Householder-QR solver as the Granger, PC and LiNGAM paths, and
// returns ErrSingular if a design is rank deficient (a constant or perfectly
// collinear predecessor).
//
// data holds one variable per outer slice (data[v] is variable v's sample), all
// of the same length n; order must be a permutation of 0..len(data)-1 — e.g.
// DirectLiNGAM(...).CausalOrder(). names, if non-nil, must match the variable
// count.
func FitSEM(data [][]float64, names []string, order []int) (*SEM, error) {
	p := len(data)
	if p < 1 {
		return nil, ErrTooFewVariables
	}
	n := len(data[0])
	for _, col := range data {
		if len(col) != n {
			return nil, ErrUnequalLengths
		}
		for _, v := range col {
			if !isFinite(v) {
				return nil, ErrNonFinite
			}
		}
	}
	nm, err := resolveNames(names, p)
	if err != nil {
		return nil, err
	}
	if !isPermutation(order, p) {
		return nil, ErrInvalidOrder
	}

	b := make([][]float64, p)
	for i := range b {
		b[i] = make([]float64, p)
	}
	intercept := make([]float64, p)

	for k, v := range order {
		// Design: [1, x_{order[0]}, …, x_{order[k-1]}] — an intercept plus the
		// predecessors of v in the causal order.
		cols := 1 + k
		x := make([][]float64, n)
		for r := 0; r < n; r++ {
			row := make([]float64, cols)
			row[0] = 1
			for c := 0; c < k; c++ {
				row[c+1] = data[order[c]][r]
			}
			x[r] = row
		}
		fit, err := fitOLS(x, data[v])
		if err != nil {
			return nil, err
		}
		intercept[v] = fit.coef[0]
		for c := 0; c < k; c++ {
			b[v][order[c]] = fit.coef[c+1]
		}
	}

	ord := append([]int(nil), order...)
	return newSEM(nm, ord, b, intercept), nil
}

// newSEM assembles the struct and the inverse-permutation lookup. b, order and
// intercept are taken as owned (callers pass fresh copies).
func newSEM(names []string, order []int, b [][]float64, intercept []float64) *SEM {
	pos := make([]int, len(order))
	for k, v := range order {
		pos[v] = k
	}
	return &SEM{names: names, order: order, pos: pos, b: b, intercept: intercept}
}

// Nodes returns a copy of the variable names, indexed as in the model.
func (s *SEM) Nodes() []string { return append([]string(nil), s.names...) }

// CausalOrder returns a copy of the causal order as variable indices, most
// exogenous first.
func (s *SEM) CausalOrder() []int { return append([]int(nil), s.order...) }

// OrderedNodes returns the variable names in causal order (most exogenous first).
func (s *SEM) OrderedNodes() []string {
	out := make([]string, len(s.order))
	for k, v := range s.order {
		out[k] = s.names[v]
	}
	return out
}

// Weights returns a copy of the coefficient matrix B; B[i][j] is the direct
// causal effect of variable j on variable i.
func (s *SEM) Weights() [][]float64 {
	out := make([][]float64, len(s.b))
	for i, row := range s.b {
		out[i] = append([]float64(nil), row...)
	}
	return out
}

// Intercepts returns a copy of the intercept vector c.
func (s *SEM) Intercepts() []float64 { return append([]float64(nil), s.intercept...) }

// Intervene computes the interventional expectation E[X | do(interventions)]
// under the model's mean-zero noise. do maps a variable NAME to the value the
// do-operator forces it to; every other variable takes its structural mean given
// the already-computed values of its predecessors in the causal order (the
// forward substitution x_i = c_i + Σ_j B[i][j]·x_j, since E[e]=0). An intervened
// variable's own structural equation is discarded — do(X=x) severs X's incoming
// edges — so an intervention on a downstream variable does not propagate
// backward to its causes. A name absent from the model returns ErrUnknownVariable.
//
// The result maps every variable name to its interventional expected value.
func (s *SEM) Intervene(do map[string]float64) (map[string]float64, error) {
	forced, err := s.byIndex(do)
	if err != nil {
		return nil, err
	}
	val := make([]float64, len(s.names))
	for _, v := range s.order {
		if dv, ok := forced[v]; ok {
			val[v] = dv
			continue
		}
		x := s.intercept[v]
		for j := 0; j < len(s.b[v]); j++ {
			x += s.b[v][j] * val[j]
		}
		val[v] = x
	}
	return s.toNamed(val), nil
}

// Counterfactual answers a "what would X have been?" query: given that we
// OBSERVED `observed`, what value would each variable have taken had we
// intervened with `do`? It is Pearl's three-step abduction–action–prediction on
// the linear SEM:
//
//  1. Abduction — infer the disturbance each variable actually realized from the
//     observation: e_i = x_i − c_i − Σ_j B[i][j]·x_j.
//  2. Action — apply do(·): the intervened variables are fixed to their
//     counterfactual values and their incoming edges cut.
//  3. Prediction — re-propagate the structural equations in causal order holding
//     the abducted disturbances fixed: x'_i = c_i + Σ_j B[i][j]·x'_j + e_i (or
//     the forced value for an intervened variable).
//
// observed must assign a value to every variable (abduction needs the full row),
// else ErrIncompleteObservation; an unknown name in either map returns
// ErrUnknownVariable. With an empty `do` the result reproduces `observed` (the
// abducted noise exactly regenerates it) — the consistency check the method
// satisfies by construction.
func (s *SEM) Counterfactual(observed, do map[string]float64) (map[string]float64, error) {
	n := len(s.names)
	obs, err := s.byIndex(observed)
	if err != nil {
		return nil, err
	}
	if len(obs) != n {
		return nil, ErrIncompleteObservation
	}
	forced, err := s.byIndex(do)
	if err != nil {
		return nil, err
	}

	// 1. Abduction: recover each realized disturbance from the observed row.
	x := make([]float64, n)
	for i := 0; i < n; i++ {
		x[i] = obs[i]
	}
	e := make([]float64, n)
	for i := 0; i < n; i++ {
		ei := x[i] - s.intercept[i]
		for j := 0; j < n; j++ {
			ei -= s.b[i][j] * x[j]
		}
		e[i] = ei
	}

	// 2+3. Action + prediction: forward-substitute with the disturbances fixed.
	cf := make([]float64, n)
	for _, v := range s.order {
		if dv, ok := forced[v]; ok {
			cf[v] = dv
			continue
		}
		xv := s.intercept[v] + e[v]
		for j := 0; j < n; j++ {
			xv += s.b[v][j] * cf[j]
		}
		cf[v] = xv
	}
	return s.toNamed(cf), nil
}

// TotalEffect returns the total causal effect on E[to] of a unit do-intervention
// on `from`: the sum over every directed path from → to of the product of its
// edge coefficients (the reduced-form (I−B)⁻¹ entry). It is 0 when `to` is not a
// descendant of `from`. Both names must exist in the model.
func (s *SEM) TotalEffect(from, to string) (float64, error) {
	if _, ok := s.index(from); !ok {
		return 0, fmt.Errorf("%w: %q", ErrUnknownVariable, from)
	}
	if _, ok := s.index(to); !ok {
		return 0, fmt.Errorf("%w: %q", ErrUnknownVariable, to)
	}
	// The effect is linear, so E[to | do(from=1)] − E[to | do(from=0)] isolates it;
	// intercepts and every other root cancel in the difference.
	hi, err := s.Intervene(map[string]float64{from: 1})
	if err != nil {
		return 0, err
	}
	lo, err := s.Intervene(map[string]float64{from: 0})
	if err != nil {
		return 0, err
	}
	return hi[to] - lo[to], nil
}

// --- helpers ---------------------------------------------------------------

func (s *SEM) index(name string) (int, bool) {
	for i, nm := range s.names {
		if nm == name {
			return i, true
		}
	}
	return -1, false
}

// byIndex converts a name→value map to an index→value map, rejecting any name
// the model does not contain.
func (s *SEM) byIndex(m map[string]float64) (map[int]float64, error) {
	out := make(map[int]float64, len(m))
	for name, v := range m {
		i, ok := s.index(name)
		if !ok {
			return nil, fmt.Errorf("%w: %q", ErrUnknownVariable, name)
		}
		if !isFinite(v) {
			return nil, ErrNonFinite
		}
		out[i] = v
	}
	return out, nil
}

func (s *SEM) toNamed(val []float64) map[string]float64 {
	out := make(map[string]float64, len(val))
	for i, v := range val {
		out[s.names[i]] = v
	}
	return out
}

// resolveNames returns a length-n name slice: a copy of names when it matches,
// generated V0,V1,… when names is nil, else ErrNameCount.
func resolveNames(names []string, n int) ([]string, error) {
	if names == nil {
		out := make([]string, n)
		for i := range out {
			out[i] = fmt.Sprintf("V%d", i)
		}
		return out, nil
	}
	if len(names) != n {
		return nil, ErrNameCount
	}
	return append([]string(nil), names...), nil
}

// isPermutation reports whether order is a permutation of 0..n-1.
func isPermutation(order []int, n int) bool {
	if len(order) != n {
		return false
	}
	seen := make([]bool, n)
	for _, v := range order {
		if v < 0 || v >= n || seen[v] {
			return false
		}
		seen[v] = true
	}
	return true
}

// topoOrder returns a deterministic topological order of the variables for the
// edge set {j→i : coef[i][j] ≠ 0} (j is a parent of i), ties broken to the
// lowest index. ok is false when the graph contains a directed cycle.
func topoOrder(coef [][]float64) (order []int, ok bool) {
	n := len(coef)
	indeg := make([]int, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i != j && coef[i][j] != 0 {
				indeg[i]++ // edge j→i
			}
		}
	}
	order = make([]int, 0, n)
	done := make([]bool, n)
	for len(order) < n {
		// Kahn's algorithm, picking the lowest-index available node for determinism.
		next := -1
		for v := 0; v < n; v++ {
			if !done[v] && indeg[v] == 0 {
				next = v
				break
			}
		}
		if next == -1 {
			return nil, false // a cycle remains
		}
		done[next] = true
		order = append(order, next)
		for i := 0; i < n; i++ {
			if !done[i] && i != next && coef[i][next] != 0 {
				indeg[i]--
			}
		}
	}
	return order, true
}
