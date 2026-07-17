package causa

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

// Errors returned by DirectLiNGAM. ErrTooFewVariables, ErrUnequalLengths,
// ErrTooFewSamples, ErrNonFinite, ErrNameCount and ErrSingular are shared with
// the rest of the package; the value below is specific to LiNGAM.
var (
	// ErrInvalidThreshold is returned when the configured pruning threshold is
	// negative. A threshold is an absolute lower bound on |coefficient| and must
	// be zero (the default, meaning "no pruning") or positive.
	ErrInvalidThreshold = errors.New("causa: prune threshold must be non-negative")
)

// LiNGAMOptions configures DirectLiNGAM. A nil *LiNGAMOptions, or a zero-valued
// field within it, selects the documented default for that setting.
type LiNGAMOptions struct {
	// PruneThreshold zeroes every estimated connection strength whose absolute
	// value is strictly below it, after the coefficients are fitted. Zero (the
	// default) disables pruning and returns the raw least-squares estimates; a
	// negative value returns ErrInvalidThreshold.
	//
	// This is a deliberately simple absolute-magnitude filter, NOT the
	// adaptive-lasso pruning used by the paper's reference implementations. That
	// technique needs an L1-regularized optimizer, which pulls in machinery this
	// stdlib-only library does not carry; the honest, documented divergence is a
	// fixed threshold the caller sets from domain knowledge or a first
	// unthresholded run.
	PruneThreshold float64
}

// WeightedEdge is a directed edge From → To carrying the estimated linear
// connection strength Weight: the coefficient of variable From in the structural
// equation of variable To (the direct causal effect of From on To).
type WeightedEdge struct {
	From   int
	To     int
	Weight float64
}

// LiNGAMResult is the output of DirectLiNGAM: a full causal ordering of the
// variables together with the weighted, strictly lower-triangular (in that
// order) coefficient matrix B of the linear structural equation model
//
//	x = B·x + e,
//
// where e is a vector of independent non-Gaussian noise terms. Unlike the CPDAG
// returned by PCStable, this is a single fully directed model, not a Markov
// equivalence class — the non-Gaussianity of the noise is exactly what makes the
// direction identifiable (see DirectLiNGAM for the assumptions that buys it).
type LiNGAMResult struct {
	names []string
	// order holds the recovered causal order as original variable indices,
	// order[0] the most exogenous (a root cause), order[len-1] the most
	// downstream. Every structural parent of a variable precedes it here.
	order []int
	// b is the p×p coefficient matrix; b[i][j] is the direct effect of variable
	// j on variable i (edge j → i). It is strictly lower-triangular when the rows
	// and columns are permuted into causal order: b[i][j] is non-zero only if j
	// precedes i in order.
	b [][]float64
}

// Nodes returns a copy of the variable names, indexed as in the input data.
func (r *LiNGAMResult) Nodes() []string { return append([]string(nil), r.names...) }

// CausalOrder returns a copy of the recovered causal order as original variable
// indices, most exogenous first. Every variable's structural parents appear
// before it.
func (r *LiNGAMResult) CausalOrder() []int { return append([]int(nil), r.order...) }

// OrderedNodes returns the variable names in recovered causal order (most
// exogenous first) — CausalOrder mapped through the names.
func (r *LiNGAMResult) OrderedNodes() []string {
	out := make([]string, len(r.order))
	for i, v := range r.order {
		out[i] = r.names[v]
	}
	return out
}

// Coefficient returns the estimated direct causal effect of variable from on
// variable to: the coefficient of x_from in the structural equation of x_to.
// It is zero unless from precedes to in the causal order.
func (r *LiNGAMResult) Coefficient(from, to int) float64 { return r.b[to][from] }

// Weights returns a copy of the coefficient matrix B; B[i][j] is the direct
// effect of variable j on variable i (edge j → i).
func (r *LiNGAMResult) Weights() [][]float64 {
	out := make([][]float64, len(r.b))
	for i := range r.b {
		out[i] = append([]float64(nil), r.b[i]...)
	}
	return out
}

// Edges returns the non-zero directed edges of the recovered model as
// From → To with the estimated Weight (the direct effect of From on To), in
// ascending (From, To) order. A coefficient left at exactly zero — either never
// fitted because From does not precede To, or removed by PruneThreshold — is not
// reported.
func (r *LiNGAMResult) Edges() []WeightedEdge {
	var edges []WeightedEdge
	p := len(r.b)
	for from := 0; from < p; from++ {
		for to := 0; to < p; to++ {
			if r.b[to][from] != 0 {
				edges = append(edges, WeightedEdge{From: from, To: to, Weight: r.b[to][from]})
			}
		}
	}
	return edges
}

// String renders the result as a deterministic multi-line summary: a header with
// the node count and names, the recovered causal order, then the weighted edges
// (a -> b: w), each group in a fixed order.
func (r *LiNGAMResult) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "DirectLiNGAM with %d nodes %v:", len(r.names), r.names)
	fmt.Fprintf(&b, "\n  causal order: %v", r.OrderedNodes())
	for _, e := range r.Edges() {
		fmt.Fprintf(&b, "\n  %s -> %s: %.6g", r.names[e.From], r.names[e.To], e.Weight)
	}
	return b.String()
}

// DirectLiNGAM estimates a Linear Non-Gaussian Acyclic Model with the
// deterministic DirectLiNGAM method of Shimizu, Inazumi, Sogawa, Hyvärinen,
// Kawahara, Washio, Hoyer & Bollen, "DirectLiNGAM: A Direct Method for Learning
// a Linear Non-Gaussian Structural Equation Model", JMLR 12 (2011) 1225–1248.
//
// It recovers a full causal ordering of the variables and the weighted
// coefficient matrix B of the structural equation model x = B·x + e. This is the
// direct, fixed-point-free algorithm of the paper, NOT ICA-LiNGAM: there is no
// random ICA initialization, so — like the rest of causa — the same input always
// yields the same output (see "Determinism" below).
//
// Input. data holds one variable per outer slice (data[v] is variable v's
// sample); every variable must have the same length n. names, if non-nil, must
// have one entry per variable; when nil, the variables are named V0, V1, …
//
// Algorithm (paper §3, Algorithm 1). On standardized working copies of the data:
//
//  1. Find the most exogenous variable. For each remaining candidate x_i, and
//     each other remaining x_j, regress x_i on x_j and x_j on x_i by ordinary
//     least squares (the same Householder-QR solver as the Granger and PC paths,
//     fitOLS), forming residuals r_i^{(j)} and r_j^{(i)}. Score each candidate by
//     the paper's entropy-based pairwise independence measure (§4.1.2): the sum
//     over j of min(0, T)², where T is the difference of pairwise mutual-
//     information estimates between the two directions (see the "Independence
//     measure" note). The candidate minimizing the total is the most exogenous
//     and is appended to the causal order.
//  2. Residualize. Regress every remaining variable on the chosen one and replace
//     it by the residual (paper §3, step 2(d)); recurse on the residuals until a
//     single variable remains, which closes the order.
//  3. Coefficients. Estimate B by regressing each variable on its predecessors in
//     the recovered order, by ordinary least squares on the ORIGINAL
//     (unstandardized) data (paper §3, step 3). B[i][j] is the direct effect of
//     x_j on x_i and is non-zero only if j precedes i in the order, so B is
//     strictly lower-triangular in causal order.
//
// Independence measure (paper §4.1.2). For two standardized variables u and w the
// differential entropy is estimated by the maximum-entropy approximation of
// Hyvärinen (1998),
//
//	H(u) ≈ H(ν) − k₁·(E[log cosh u] − γ)² − k₂·(E[u·exp(−u²/2)])²,
//
// with H(ν) = (1 + log 2π)/2 the entropy of a standard Gaussian, k₁ = 79.047,
// k₂ = 7.4129 and γ = E[log cosh ν] = 0.37457. The two moments capture,
// respectively, the even (heavy-/light-tailed) and odd (skew) departures from
// Gaussianity that make the causal direction identifiable. The directional
// statistic for candidate i against j is
//
//	T = (H(x_j) + H(r_i^{(j)})) − (H(x_i) + H(r_j^{(i)})),
//
// the residuals standardized before their entropy is taken; only its negative
// part is penalized (min(0, T)²), summed over j. This is the entropy variant of
// the pairwise likelihood ratio of Hyvärinen & Smith, "Pairwise Likelihood Ratios
// for Estimation of Non-Gaussian Structural Equation Models", JMLR 14 (2013)
// 111–152, which the paper's reference implementation adopts.
//
// Normalization choice. Standardization, the pairwise correlation and every
// residual variance use the population (÷n) convention consistently. The widely
// used reference implementation mixes ÷n and ÷(n−1) in one place; the difference
// is inconsequential for order recovery and this library's self-consistent choice
// is the one cross-checked by the numeric oracle test.
//
// Pruning. By default the raw least-squares coefficients are returned.
// LiNGAMOptions.PruneThreshold applies a simple absolute-magnitude filter; the
// adaptive-lasso pruning of the reference implementations is consciously not
// reproduced (it needs an L1 optimizer this stdlib-only library does not carry),
// a divergence documented on LiNGAMOptions.
//
// Determinism. Every phase iterates variable indices; there is no dependence on
// map iteration order, and no randomness anywhere. Ties in the exogeneity score
// are broken by the lowest variable index. The same input always yields the same
// result (locked by a determinism test).
//
// Assumptions — stated bluntly because violating them silently returns a
// plausible-looking but wrong model:
//
//   - Linearity. Each variable is a linear function of its parents plus noise.
//   - Acyclicity. The true structure is a DAG.
//   - Non-Gaussian, independent noise. The disturbances e_i are mutually
//     independent and at most one is Gaussian. This is the load-bearing
//     assumption: it is what makes the direction identifiable from observational
//     data at all.
//   - Causal sufficiency. No unobserved common cause of two measured variables.
//
// On GAUSSIAN noise the model is fundamentally UNIDENTIFIABLE — a linear-Gaussian
// SEM and its reverse fit the data equally well — and the returned order is
// arbitrary (it depends on sampling noise, not structure). DirectLiNGAM does not
// and cannot detect this; it returns a fully oriented model regardless. Treat a
// LiNGAM result as meaningful only when the non-Gaussianity assumption holds. This
// failure mode is pinned by an honest-failure test.
//
// Errors: ErrTooFewVariables (< 2 variables), ErrUnequalLengths (ragged data),
// ErrTooFewSamples (n < 3), ErrNonFinite (NaN/Inf in the data), ErrNameCount
// (names length mismatch), ErrInvalidThreshold (negative PruneThreshold), and
// ErrSingular when a variable is constant (cannot be standardized) or a
// regression design is rank deficient (collinear variables, or fewer samples than
// a variable has predecessors).
func DirectLiNGAM(data [][]float64, names []string, opts *LiNGAMOptions) (*LiNGAMResult, error) {
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
	if n < 3 {
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

	threshold := 0.0
	if opts != nil {
		if opts.PruneThreshold < 0 {
			return nil, ErrInvalidThreshold
		}
		threshold = opts.PruneThreshold
	}

	// Working copies get residualized in place as the order is peeled off; the
	// caller's slices and the ORIGINAL data (used for the final coefficients) are
	// left untouched.
	work := make([][]float64, p)
	for i := range data {
		work[i] = append([]float64(nil), data[i]...)
	}

	order := make([]int, 0, p)
	remaining := make([]int, p)
	for i := range remaining {
		remaining[i] = i
	}

	for len(remaining) > 1 {
		m, err := mostExogenous(work, remaining)
		if err != nil {
			return nil, err
		}
		order = append(order, m)
		if err := residualizeOn(work, remaining, m); err != nil {
			return nil, err
		}
		remaining = removeIndex(remaining, m)
	}
	order = append(order, remaining[0])

	b, err := lingamCoefficients(data, order, n, p)
	if err != nil {
		return nil, err
	}
	if threshold > 0 {
		pruneCoefficients(b, threshold)
	}

	return &LiNGAMResult{names: names, order: order, b: b}, nil
}

// mostExogenous returns the remaining variable index that minimizes the paper's
// entropy-based dependence score — the sum over the other remaining variables of
// min(0, T)², where T is the directional mutual-information difference. Ties are
// broken by the lowest index (remaining is kept in ascending order and scanned in
// order, so the first minimizer wins). It returns ErrSingular when a working
// column is constant or a pair is perfectly collinear.
func mostExogenous(work [][]float64, remaining []int) (int, error) {
	// Standardize each remaining working column once for this round.
	std := make([][]float64, len(remaining))
	for idx, v := range remaining {
		s, ok := standardize(work[v])
		if !ok {
			return 0, ErrSingular
		}
		std[idx] = s
	}

	bestIdx := -1
	bestScore := math.Inf(1)
	for a := range remaining {
		var score float64
		for b := range remaining {
			if a == b {
				continue
			}
			t, err := lingamDiffMI(std[a], std[b])
			if err != nil {
				return 0, err
			}
			if t < 0 {
				score += t * t
			}
		}
		if score < bestScore {
			bestScore = score
			bestIdx = a
		}
	}
	if bestIdx < 0 {
		return 0, ErrSingular
	}
	return remaining[bestIdx], nil
}

// lingamDiffMI returns the directional statistic T of DirectLiNGAM §4.1.2 for a
// candidate cause xi against xj (both already standardized to zero mean and unit
// variance):
//
//	T = (H(xj) + H(r_i^{(j)})) − (H(xi) + H(r_j^{(i)})),
//
// where r_i^{(j)} is the OLS residual of xi on xj and r_j^{(i)} that of xj on xi,
// each standardized before its entropy is taken. T is the log-likelihood ratio of
// the two directions: a POSITIVE T favors xi → xj (xi exogenous), a NEGATIVE T is
// evidence against it (the reverse direction fits better) — which is exactly why
// the caller penalizes only min(0, T)²: a true root scores ~zero. Returns
// ErrSingular when a residual collapses to zero (perfect collinearity).
func lingamDiffMI(xi, xj []float64) (float64, error) {
	rij, err := lingamPairResidual(xi, xj)
	if err != nil {
		return 0, err
	}
	rji, err := lingamPairResidual(xj, xi)
	if err != nil {
		return 0, err
	}
	sij, ok := standardize(rij)
	if !ok {
		return 0, ErrSingular
	}
	sji, ok := standardize(rji)
	if !ok {
		return 0, ErrSingular
	}
	return (lingamEntropy(xj) + lingamEntropy(sij)) - (lingamEntropy(xi) + lingamEntropy(sji)), nil
}

// lingamPairResidual regresses target on an intercept and regressor by ordinary
// least squares (reusing the Householder-QR solver via residualize) and returns
// the residual vector target − fitted. ErrSingular propagates when the design is
// rank deficient (a constant regressor).
func lingamPairResidual(target, regressor []float64) ([]float64, error) {
	n := len(target)
	x := make([][]float64, n)
	for r := 0; r < n; r++ {
		x[r] = []float64{1, regressor[r]}
	}
	res, _, err := residualize(x, target)
	return res, err
}

// residualizeOn replaces every remaining variable other than m by its OLS
// residual after regressing it on x_m (paper §3, step 2(d)). It works on the raw
// working data, not standardized copies, so the recursion peels the influence of
// each chosen exogenous variable off the rest.
func residualizeOn(work [][]float64, remaining []int, m int) error {
	for _, v := range remaining {
		if v == m {
			continue
		}
		res, err := lingamPairResidual(work[v], work[m])
		if err != nil {
			return err
		}
		work[v] = res
	}
	return nil
}

// lingamEntropy estimates the differential entropy of a standardized variable u
// (zero mean, unit variance assumed) via the maximum-entropy approximation of
// Hyvärinen (1998) used by DirectLiNGAM §4.1.2:
//
//	H(u) ≈ (1 + log 2π)/2 − k₁·(mean(log cosh u) − γ)² − k₂·(mean(u·exp(−u²/2)))².
//
// The log cosh term is evaluated through logCosh for numerical stability on
// heavy-tailed standardized values.
func lingamEntropy(u []float64) float64 {
	const (
		k1    = 79.047
		k2    = 7.4129
		gamma = 0.37457
	)
	n := float64(len(u))
	var s1, s2 float64
	for _, x := range u {
		s1 += logCosh(x)
		s2 += x * math.Exp(-x*x/2)
	}
	m1 := s1/n - gamma
	m2 := s2 / n
	return 0.5*(1+math.Log(2*math.Pi)) - k1*m1*m1 - k2*m2*m2
}

// logCosh returns log(cosh(x)) computed as |x| + log1p(exp(−2|x|)) − ln 2, which
// stays finite and accurate for large |x| where cosh(x) itself overflows.
func logCosh(x float64) float64 {
	a := math.Abs(x)
	return a + math.Log1p(math.Exp(-2*a)) - math.Ln2
}

// standardize returns a zero-mean, unit-variance copy of v using the population
// (÷n) variance, and ok=false when v is (numerically) constant so that no valid
// standardization exists.
func standardize(v []float64) ([]float64, bool) {
	n := float64(len(v))
	var mean float64
	for _, x := range v {
		mean += x
	}
	mean /= n
	var ss float64
	for _, x := range v {
		d := x - mean
		ss += d * d
	}
	variance := ss / n
	if variance <= 0 || !isFinite(variance) {
		return nil, false
	}
	sd := math.Sqrt(variance)
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = (x - mean) / sd
	}
	return out, true
}

// lingamCoefficients estimates the strictly lower-triangular (in causal order)
// coefficient matrix B by regressing each ordered variable on an intercept plus
// its predecessors, by ordinary least squares on the ORIGINAL data. B[i][j] is
// the direct effect of variable j on variable i.
func lingamCoefficients(data [][]float64, order []int, n, p int) ([][]float64, error) {
	b := make([][]float64, p)
	for i := range b {
		b[i] = make([]float64, p)
	}
	for idx := 1; idx < len(order); idx++ {
		target := order[idx]
		preds := order[:idx]
		x := make([][]float64, n)
		for r := 0; r < n; r++ {
			row := make([]float64, len(preds)+1)
			row[0] = 1
			for c, pv := range preds {
				row[c+1] = data[pv][r]
			}
			x[r] = row
		}
		fit, err := fitOLS(x, data[target])
		if err != nil {
			return nil, err
		}
		for c, pv := range preds {
			b[target][pv] = fit.coef[c+1] // skip the intercept at column 0
		}
	}
	return b, nil
}

// pruneCoefficients zeroes every entry of b whose absolute value is strictly
// below threshold (the simple absolute-magnitude pruning of
// LiNGAMOptions.PruneThreshold).
func pruneCoefficients(b [][]float64, threshold float64) {
	for i := range b {
		for j := range b[i] {
			if math.Abs(b[i][j]) < threshold {
				b[i][j] = 0
			}
		}
	}
}

// removeIndex returns a new slice with the first occurrence of value removed,
// preserving order. The recovered-order peeling relies on the remaining set
// staying in ascending index order for the lowest-index tie-break.
func removeIndex(s []int, value int) []int {
	out := make([]int, 0, len(s)-1)
	for _, v := range s {
		if v != value {
			out = append(out, v)
		}
	}
	return out
}
