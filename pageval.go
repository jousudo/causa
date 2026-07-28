package causa

import "errors"

// ErrNotIdentifiable is returned by (*PAGIDResult).Evaluate when the effect is not
// identifiable from the PAG — there is no estimand to evaluate.
var ErrNotIdentifiable = errors.New("causa: effect is not identifiable from the PAG")

// Evaluate turns an identified PAG estimand into numbers. Given a discrete
// observational joint over all variables (as from NewDistribution), it returns the
// interventional table P(y | do(x)) — or P(y | do(x), z) for a conditional query —
// over the query's Y, X (and Z) variables, read with ProbAt exactly as an Identify
// estimand's Evaluate output is: fix the intervention X (and context Z) to their
// values and read the outcome Y.
//
// Why this is not just Estimand.Evaluate. The raw IDP/CIDP estimand is a c-factor
// expression that, once evaluated, carries "spectator" variables outside Y ∪ X ∪ Z
// — the non-ancestors of the outcome that the c-factor form leaves in. The
// interventional distribution does not depend on them, so they come out constant;
// Evaluate reduces them away (slices them at 0), returning a clean table over
// Y ∪ X ∪ Z. The correctness of this reduction — that the effect really is
// identifiable and the spectators really are constant — is validated against
// brute-force truth on random latent SCMs (see the PAG-evaluation tests), the same
// parameterization-independent standard used for Identify.
//
// This is the discrete-data companion to the symbolic decision: it works on a
// discrete observational joint, the same scope as Expr.Evaluate. Returns
// ErrNotIdentifiable when the effect is not identifiable, or the underlying
// evaluator's error (ErrBadDistribution) when joint is not a full joint over
// 0..n-1.
func (r *PAGIDResult) Evaluate(joint *Distribution) (*Distribution, error) {
	if !r.Identifiable || r.Estimand == nil {
		return nil, ErrNotIdentifiable
	}
	raw, err := r.Estimand.Evaluate(joint)
	if err != nil {
		return nil, err
	}
	keep := map[int]bool{}
	for _, v := range r.y {
		keep[v] = true
	}
	for _, v := range r.x {
		keep[v] = true
	}
	for _, v := range r.z {
		keep[v] = true
	}
	return sliceSpectators(raw, keep), nil
}

// sliceSpectators reduces a distribution to the variables in keep by fixing every
// other variable to state 0. It is exact only when those other variables are
// constant in the table (the interventional-estimand case Evaluate relies on);
// otherwise it returns the 0-slice.
func sliceSpectators(d *Distribution, keep map[int]bool) *Distribution {
	var kv, kc []int
	pos := make(map[int]int, len(d.vars)) // variable -> its position in d.vars
	for k, v := range d.vars {
		pos[v] = k
		if keep[v] {
			kv = append(kv, v)
			kc = append(kc, d.card[k])
		}
	}
	total := 1
	for _, c := range kc {
		total *= c
	}
	out := make([]float64, total)
	assign := make([]int, len(d.vars)) // positions of spectators stay 0
	var rec func(ki, outIdx int)
	rec = func(ki, outIdx int) {
		if ki == len(kv) {
			di := 0
			for k := range d.vars {
				di = di*d.card[k] + assign[k]
			}
			out[outIdx] = d.prob[di]
			return
		}
		p := pos[kv[ki]]
		base := outIdx * kc[ki]
		for s := 0; s < kc[ki]; s++ {
			assign[p] = s
			rec(ki+1, base+s)
		}
		assign[p] = 0
	}
	rec(0, 0)
	return &Distribution{vars: kv, card: kc, prob: out}
}
