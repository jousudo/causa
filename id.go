package causa

import (
	"fmt"
	"sort"
	"strings"
)

// IDResult is the outcome of Identify: either an identifiable interventional
// distribution with its estimand, or a proof that the effect is NOT identifiable
// from observational data given the diagram (a hedge).
//
// Non-identifiability is a legitimate mathematical answer, not an error: it means
// no estimand exists, because two causal models can agree on every observable
// probability P(V) yet disagree on P(y | do(x)). Identifiable reports which case
// holds; Estimand is valid only when Identifiable is true.
type IDResult struct {
	names []string

	// Identifiable reports whether P(y | do(x)) is identifiable from P(V) in the
	// diagram. When true, Estimand holds the formula; when false, Hedge reports a
	// witnessing vertex set.
	Identifiable bool

	// Estimand is the interventional distribution written over the observational
	// P(V); valid only when Identifiable. Evaluate it on discrete data with
	// Estimand.Evaluate.
	Estimand *Expr

	hedgeR, hedgeS []int
}

// Hedge returns the witnessing vertex sets (F ⊇ F') of the hedge that blocks
// identification, valid only when Identifiable is false: the effect is not
// identifiable because of the confounding structure spanning these vertices. The
// two slices are variable indices, ascending. When Identifiable is true it
// returns nil, nil.
func (r *IDResult) Hedge() (larger, smaller []int) {
	if r.Identifiable {
		return nil, nil
	}
	return append([]int(nil), r.hedgeR...), append([]int(nil), r.hedgeS...)
}

// String renders the result: the estimand (with the diagram's variable names)
// when identifiable, else a "not identifiable" line naming the hedge vertices.
func (r *IDResult) String() string {
	if r.Identifiable {
		return renderExpr(r.Estimand, r.names)
	}
	nm := func(idx []int) string {
		parts := make([]string, len(idx))
		for k, v := range idx {
			parts[k] = r.names[v]
		}
		return strings.Join(parts, ",")
	}
	return fmt.Sprintf("not identifiable (hedge over {%s} ⊇ {%s})", nm(r.hedgeR), nm(r.hedgeS))
}

// hedge signals a failed (non-identifiable) branch of the recursion, carrying the
// witnessing C-forest vertex sets.
type hedge struct {
	r, s []int
}

// Identify runs the Shpitser–Pearl ID algorithm and decides whether the
// interventional distribution P(y | do(x)) — the effect of setting the variables
// x to a value on the distribution of the variables y — is identifiable from the
// observational distribution P(V) in the causal diagram g, returning its estimand
// when it is.
//
// y and x are variable-index sets: y (the outcome) must be non-empty, x (the
// intervention) may be empty (then the query is the plain observational marginal
// P(y)), and the two must be disjoint with no repeats. The diagram admits latent
// confounders (bidirected edges), so this solves the causal-sufficiency-free
// identification problem the linear SEM path (Intervene) does not: it recovers
// the interventional distribution from the observational one plus the graph,
// rather than from a fully specified structural model.
//
// The result is symbolic — an Expr over P(V). To turn it into numbers on discrete
// data, evaluate it against an observational joint with Estimand.Evaluate. When
// the effect is not identifiable, Identifiable is false and Hedge reports the
// obstructing vertices; this is a valid answer, not an error.
//
// Errors (bad input only): ErrEdgeOutOfRange (a query index out of range),
// ErrEmptyOutcome (empty y), ErrOverlappingQuery (x and y share a variable),
// ErrDuplicateQueryVar (a set repeats a variable).
//
// Scope. This is the ID algorithm for a single known causal diagram without
// selection bias; it decides identifiability soundly and completely for that
// setting. It does not take a PAG (an equivalence class) as input — identifying
// an effect over a PAG is a different algorithm (IDP) and out of scope.
//
// Reference: Shpitser & Pearl, "Identification of Joint Interventional
// Distributions in Recursive Semi-Markovian Causal Models", AAAI 2006; Tian &
// Pearl, "A General Identification Condition for Causal Effects", AAAI 2002
// (C-component factorization).
func Identify(g *Diagram, y, x []int) (*IDResult, error) {
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

	est, hd := g.id(ymask, xmask, jointExpr(allVars), full)
	if hd != nil {
		return &IDResult{names: g.names, Identifiable: false, hedgeR: hd.r, hedgeS: hd.s}, nil
	}
	return &IDResult{names: g.names, Identifiable: true, Estimand: est}, nil
}

func validateQuery(n int, y, x []int) error {
	if len(y) == 0 {
		return ErrEmptyOutcome
	}
	// Range-check and reject repeats within each set.
	distinct := func(set []int) ([]bool, error) {
		mark := make([]bool, n)
		for _, v := range set {
			if v < 0 || v >= n {
				return nil, ErrEdgeOutOfRange
			}
			if mark[v] {
				return nil, ErrDuplicateQueryVar
			}
			mark[v] = true
		}
		return mark, nil
	}
	ymark, err := distinct(y)
	if err != nil {
		return err
	}
	xmark, err := distinct(x)
	if err != nil {
		return err
	}
	for v := 0; v < n; v++ {
		if ymark[v] && xmark[v] {
			return ErrOverlappingQuery
		}
	}
	return nil
}

// id is the recursive ID algorithm operating on the vertex subset `sub` (the
// current graph G's vertices), the outcome y and intervention x (masks), and the
// current distribution expression P (a distribution over at least `sub`). It
// returns the estimand for P(y | do(x)) in G[sub], or a non-nil hedge when the
// effect is not identifiable.
//
// The line numbering follows Shpitser & Pearl (2006), Algorithm 1.
func (g *Diagram) id(y, x []bool, P *Expr, sub []bool) (*Expr, *hedge) {
	// Line 1: no intervention left — the query is a plain marginal of P.
	if maskEmpty(x) {
		return marginalExpr(maskToSorted(maskDiff(sub, y)), P), nil
	}

	// Line 2: discard non-ancestors of Y (they cannot affect it).
	anY := g.ancestors(sub, y, nil)
	if maskProperSubset(anY, sub) {
		newP := marginalExpr(maskToSorted(maskDiff(sub, anY)), P)
		return g.id(y, maskIntersect(x, anY), newP, anY)
	}

	// Line 3: force into X the vertices W that are non-ancestors of Y once the
	// arrows into X are cut — intervening on them changes nothing but sharpens
	// the c-component structure.
	anYdo := g.ancestors(sub, y, x) // An(Y) in G_{\overline X}
	w := maskDiff(maskDiff(sub, x), anYdo)
	if !maskEmpty(w) {
		return g.id(y, maskUnion(x, w), P, sub)
	}

	// Lines 4–7 hinge on the c-components of G[sub \ X].
	comps := g.cComponents(maskDiff(sub, x))
	if len(comps) > 1 {
		// Line 4: factorize over the c-components, then sum out everything but Y∪X.
		factors := make([]*Expr, 0, len(comps))
		for _, si := range comps {
			est, hd := g.id(si, maskDiff(sub, si), P, sub)
			if hd != nil {
				return nil, hd
			}
			factors = append(factors, est)
		}
		over := maskToSorted(maskDiff(sub, maskUnion(y, x)))
		return marginalExpr(over, productExpr(factors)), nil
	}

	// Single c-component S of G[sub \ X].
	s := comps[0]
	d := g.cComponents(sub) // c-components of the whole current graph

	// Line 5: the whole graph is one c-component — a hedge, not identifiable.
	if len(d) == 1 && maskEqual(d[0], sub) {
		return nil, &hedge{r: maskToSorted(sub), s: maskToSorted(s)}
	}

	order, _ := topoSub(g.dir, sub)

	// Line 6: S is itself a c-component of G — its c-factor is directly identified.
	for _, comp := range d {
		if maskEqual(comp, s) {
			cf := g.cFactor(P, sub, order, s)
			return marginalExpr(maskToSorted(maskDiff(s, y)), cf), nil
		}
	}

	// Line 7: S sits strictly inside a larger c-component S'; recurse on G[S']
	// with S's enclosing c-factor as the new distribution.
	for _, sp := range d {
		if maskProperSubset(s, sp) {
			newP := g.cFactor(P, sub, order, sp)
			return g.id(y, maskIntersect(x, sp), newP, sp)
		}
	}

	// Unreachable for a well-formed diagram (S lies in exactly one c-component).
	return nil, &hedge{r: maskToSorted(sub), s: maskToSorted(s)}
}

// cFactor builds the c-factor Q[S] = Π_{Vᵢ ∈ S} P(vᵢ | v^{(i-1)}), where v^{(i-1)}
// are the predecessors of Vᵢ in the topological order `order` of the current
// vertex set `sub`, and each conditional is taken of the current distribution P.
func (g *Diagram) cFactor(P *Expr, sub []bool, order []int, s []bool) *Expr {
	universe := maskToSorted(sub)
	var factors []*Expr
	for pos, vi := range order {
		if !s[vi] {
			continue
		}
		preds := append([]int(nil), order[:pos]...)
		sort.Ints(preds)
		factors = append(factors, condition(P, []int{vi}, preds, universe))
	}
	return productExpr(factors)
}
