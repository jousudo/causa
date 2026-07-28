package causa

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

// Distribution errors.
var (
	// ErrBadDistribution is returned when a distribution's probability table does
	// not match the declared cardinalities, or holds a negative/non-finite entry.
	ErrBadDistribution = errors.New("causa: probability table length or values invalid")
	// ErrAssignment is returned by ProbAt when an assignment does not give an
	// in-range state for exactly the distribution's variables.
	ErrAssignment = errors.New("causa: assignment does not match the distribution's variables")
)

// Distribution is a discrete joint distribution over a set of variables, held as
// a dense table. vars are the variable indices it ranges over (ascending); card
// gives their cardinalities (card[k] states for vars[k]); prob is the flat table
// in mixed-radix order with vars[0] most significant, so
// len(prob) == card[0]·card[1]·…. It is both the observational input to an
// estimand's Evaluate and the interventional table Evaluate returns; the same
// type doubles as the intermediate factor in the evaluator.
type Distribution struct {
	vars []int
	card []int
	prob []float64
}

// NewDistribution builds a full joint distribution over variables 0..len(card)-1.
// card[v] is the number of states of variable v; prob is the dense table in
// mixed-radix order (variable 0 most significant), so len(prob) must equal the
// product of the cardinalities. Entries must be finite and non-negative; the
// table is NOT required to sum to one (so it can also carry an unnormalized
// factor), but a genuine observational input normally does.
func NewDistribution(card []int, prob []float64) (*Distribution, error) {
	total := 1
	for _, c := range card {
		if c < 1 {
			return nil, ErrBadDistribution
		}
		total *= c
	}
	if len(prob) != total {
		return nil, ErrBadDistribution
	}
	for _, p := range prob {
		if !isFinite(p) || p < 0 {
			return nil, ErrBadDistribution
		}
	}
	vars := make([]int, len(card))
	for i := range vars {
		vars[i] = i
	}
	return &Distribution{
		vars: vars,
		card: append([]int(nil), card...),
		prob: append([]float64(nil), prob...),
	}, nil
}

// Vars returns a copy of the variable indices the distribution ranges over,
// ascending.
func (d *Distribution) Vars() []int { return append([]int(nil), d.vars...) }

// Card returns a copy of the cardinalities, aligned to Vars.
func (d *Distribution) Card() []int { return append([]int(nil), d.card...) }

// ProbAt returns the (possibly unnormalized) probability of an assignment. The
// map must give an in-range state for exactly the variables in Vars, else
// ErrAssignment.
func (d *Distribution) ProbAt(assign map[int]int) (float64, error) {
	if len(assign) != len(d.vars) {
		return 0, ErrAssignment
	}
	idx := 0
	for k, v := range d.vars {
		s, ok := assign[v]
		if !ok || s < 0 || s >= d.card[k] {
			return 0, ErrAssignment
		}
		idx = idx*d.card[k] + s
	}
	return d.prob[idx], nil
}

// --- dense factor operations (over a global cardinality vector) ----------
//
// gcard is indexed by variable id: gcard[v] is v's cardinality. Every factor
// keeps its vars ascending, and flatIndex builds the mixed-radix offset with the
// first (lowest-id) variable most significant — the same convention throughout,
// so tables built by one operation are read correctly by the next.

func flatIndex(vars, gcard, assign []int) int {
	idx := 0
	for _, v := range vars {
		idx = idx*gcard[v] + assign[v]
	}
	return idx
}

func factorSize(vars, gcard []int) int {
	total := 1
	for _, v := range vars {
		total *= gcard[v]
	}
	return total
}

func cardsFor(vars, gcard []int) []int {
	cs := make([]int, len(vars))
	for k, v := range vars {
		cs[k] = gcard[v]
	}
	return cs
}

// forEachAssign enumerates every joint state of `vars`, invoking fn with a
// global-length assignment vector (entries for vars set, the rest left zero).
func forEachAssign(vars, gcard []int, fn func(assign []int)) {
	assign := make([]int, len(gcard))
	var rec func(k int)
	rec = func(k int) {
		if k == len(vars) {
			fn(assign)
			return
		}
		v := vars[k]
		for s := 0; s < gcard[v]; s++ {
			assign[v] = s
			rec(k + 1)
		}
	}
	rec(0)
}

// factorMarginalize sums a factor out over the variables in `over`, returning a
// factor on the remaining variables.
func factorMarginalize(f *Distribution, over map[int]bool, gcard []int) *Distribution {
	var keep []int
	for _, v := range f.vars {
		if !over[v] {
			keep = append(keep, v)
		}
	}
	res := make([]float64, factorSize(keep, gcard))
	forEachAssign(f.vars, gcard, func(assign []int) {
		res[flatIndex(keep, gcard, assign)] += f.prob[flatIndex(f.vars, gcard, assign)]
	})
	return &Distribution{vars: keep, card: cardsFor(keep, gcard), prob: res}
}

// factorMul multiplies two factors, broadcasting over the union of their
// variables.
func factorMul(a, b *Distribution, gcard []int) *Distribution {
	union := sortedUnionInts(a.vars, b.vars)
	res := make([]float64, factorSize(union, gcard))
	forEachAssign(union, gcard, func(assign []int) {
		res[flatIndex(union, gcard, assign)] =
			a.prob[flatIndex(a.vars, gcard, assign)] * b.prob[flatIndex(b.vars, gcard, assign)]
	})
	return &Distribution{vars: union, card: cardsFor(union, gcard), prob: res}
}

// factorDivide divides num by den (den's variables must be a subset of num's),
// broadcasting den over the extra variables. 0/0 is defined as 0.
func factorDivide(num, den *Distribution, gcard []int) *Distribution {
	res := make([]float64, len(num.prob))
	forEachAssign(num.vars, gcard, func(assign []int) {
		d := den.prob[flatIndex(den.vars, gcard, assign)]
		i := flatIndex(num.vars, gcard, assign)
		if d != 0 {
			res[i] = num.prob[i] / d
		}
	})
	return &Distribution{vars: append([]int(nil), num.vars...), card: append([]int(nil), num.card...), prob: res}
}

func sortedUnionInts(a, b []int) []int {
	seen := map[int]bool{}
	for _, v := range a {
		seen[v] = true
	}
	for _, v := range b {
		seen[v] = true
	}
	out := make([]int, 0, len(seen))
	for v := range seen {
		out = append(out, v)
	}
	sort.Ints(out)
	return out
}

// --- the estimand expression ---------------------------------------------

type exprKind uint8

const (
	exprJoint       exprKind = iota // the base observational joint P over `vars`
	exprConditional                 // P(head | given), a conditional of `base` within `universe`
	exprProduct                     // a product of factors
	exprMarginal                    // a summation over `over` of `child`
)

// Expr is a symbolic estimand: the interventional distribution the ID algorithm
// identifies, written as an expression over the observational joint P(V). It is
// built from three operations — marginalization (Σ), products, and conditionals
// P(A | B) (each a ratio of marginals) — over the base joint. It is the exact,
// but NOT algebraically simplified, Tian–Pearl form; equal to a hand-derived
// textbook estimand in value (verify with Evaluate) though not necessarily
// character-for-character.
type Expr struct {
	kind exprKind

	// exprJoint: the full observed variable set (for rendering).
	vars []int

	// exprConditional: base distribution, the head/given variables, and the
	// universe (the vertex set the conditional's marginals range over).
	base                  *Expr
	head, given, universe []int

	// exprProduct.
	factors []*Expr

	// exprMarginal.
	over  []int
	child *Expr
}

func jointExpr(v []int) *Expr {
	return &Expr{kind: exprJoint, vars: append([]int(nil), v...)}
}

// condition builds P(head | given), a conditional of base whose marginals range
// over `universe`.
func condition(base *Expr, head, given, universe []int) *Expr {
	return &Expr{
		kind: exprConditional, base: base,
		head:     append([]int(nil), head...),
		given:    append([]int(nil), given...),
		universe: append([]int(nil), universe...),
	}
}

// productExpr builds a product, collapsing a single factor.
func productExpr(fs []*Expr) *Expr {
	if len(fs) == 1 {
		return fs[0]
	}
	return &Expr{kind: exprProduct, factors: fs}
}

// marginalExpr builds Σ_over child, collapsing an empty sum.
func marginalExpr(over []int, child *Expr) *Expr {
	if len(over) == 0 {
		return child
	}
	return &Expr{kind: exprMarginal, over: append([]int(nil), over...), child: child}
}

// Evaluate computes the estimand numerically against an observational joint
// distribution `joint` over the whole variable set (as from NewDistribution),
// returning the interventional distribution as a table over the query's free
// variables (Y ∪ X). This is the discrete-data companion to symbolic
// identification: given P(V) it yields the numbers of P(y | do(x)).
//
// joint must be a full joint over 0..n-1 (its Vars is 0,1,…). Errors:
// ErrBadDistribution if it is not.
func (e *Expr) Evaluate(joint *Distribution) (*Distribution, error) {
	n := len(joint.vars)
	for i, v := range joint.vars {
		if v != i {
			return nil, ErrBadDistribution // must be a full joint over 0..n-1
		}
	}
	gcard := make([]int, n)
	copy(gcard, joint.card)
	return evalExpr(e, joint, gcard), nil
}

func evalExpr(e *Expr, joint *Distribution, gcard []int) *Distribution {
	switch e.kind {
	case exprJoint:
		return joint
	case exprMarginal:
		child := evalExpr(e.child, joint, gcard)
		over := map[int]bool{}
		for _, v := range e.over {
			over[v] = true
		}
		return factorMarginalize(child, over, gcard)
	case exprProduct:
		var acc *Distribution
		for _, f := range e.factors {
			t := evalExpr(f, joint, gcard)
			if acc == nil {
				acc = t
			} else {
				acc = factorMul(acc, t, gcard)
			}
		}
		if acc == nil { // empty product = scalar 1 over no variables
			return &Distribution{vars: nil, card: nil, prob: []float64{1}}
		}
		return acc
	case exprConditional:
		b := evalExpr(e.base, joint, gcard)
		hg := map[int]bool{}
		for _, v := range e.head {
			hg[v] = true
		}
		for _, v := range e.given {
			hg[v] = true
		}
		g := map[int]bool{}
		for _, v := range e.given {
			g[v] = true
		}
		// num = Σ_{universe \ (head∪given)} base ; den = Σ_{universe \ given} base.
		numOver := map[int]bool{}
		denOver := map[int]bool{}
		for _, v := range e.universe {
			if !hg[v] {
				numOver[v] = true
			}
			if !g[v] {
				denOver[v] = true
			}
		}
		num := factorMarginalize(b, numOver, gcard)
		den := factorMarginalize(b, denOver, gcard)
		return factorDivide(num, den, gcard)
	}
	return nil
}

// --- rendering -----------------------------------------------------------

// String renders the estimand with generic variable names V0, V1, … For names
// from a specific diagram, use IDResult.String.
func (e *Expr) String() string { return renderExpr(e, nil) }

func renderExpr(e *Expr, names []string) string {
	name := func(i int) string {
		if names != nil {
			return names[i]
		}
		return fmt.Sprintf("V%d", i)
	}
	list := func(idx []int) string {
		parts := make([]string, len(idx))
		for k, v := range idx {
			parts[k] = name(v)
		}
		return strings.Join(parts, ",")
	}
	switch e.kind {
	case exprJoint:
		return "P(" + list(e.vars) + ")"
	case exprConditional:
		head := list(e.head)
		var inner string
		if len(e.given) == 0 {
			inner = head
		} else {
			inner = head + "|" + list(e.given)
		}
		if e.base.kind == exprJoint {
			return "P(" + inner + ")"
		}
		return "[" + renderExpr(e.base, names) + "](" + inner + ")"
	case exprProduct:
		parts := make([]string, len(e.factors))
		for k, f := range e.factors {
			parts[k] = renderExpr(f, names)
		}
		return strings.Join(parts, "·")
	case exprMarginal:
		return "Σ_{" + list(e.over) + "}[" + renderExpr(e.child, names) + "]"
	}
	return "?"
}
