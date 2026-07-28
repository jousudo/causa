package causa

import (
	"errors"
	"math"
)

// Gaussian-evaluation errors.
var (
	// ErrBadGaussian is returned when a Gaussian's mean/covariance shapes disagree,
	// hold a non-finite entry, the covariance is not symmetric, or a Condition
	// assignment names a variable outside the factor.
	ErrBadGaussian = errors.New("causa: mean/covariance shape, symmetry or assignment invalid")
	// ErrNotPositiveDefinite is returned when a matrix that must be positive
	// definite is not — an input covariance, or a precision block a marginalization
	// or conditioning has to invert during evaluation. For a well-posed
	// interventional estimand on a proper Gaussian joint it does not arise; it
	// signals a degenerate (rank-deficient) input or intermediate factor.
	ErrNotPositiveDefinite = errors.New("causa: matrix is not positive definite")
)

const log2Pi = 1.8378770664093454835606594728112 // log(2π)

// --- symmetric positive-definite linear algebra ---------------------------
//
// The Gaussian factor algebra needs three operations on symmetric
// positive-definite matrices: a solve (marginalization eliminates a variable
// block), an inverse (canonical → moment form), and a log-determinant (the
// scalar normalizer). All three are read off one Cholesky factorization A = L·Lᵀ,
// which also doubles as the positive-definiteness test — it fails exactly when A
// is not PD, the same rank-deficiency signal fitOLS raises for a collinear design.

// cholesky returns the lower-triangular L with A = L·Lᵀ for a symmetric PD matrix
// A, or ok=false when A is not positive definite (a non-positive pivot).
func cholesky(a [][]float64) (l [][]float64, ok bool) {
	n := len(a)
	l = make([][]float64, n)
	for i := range l {
		l[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			sum := a[i][j]
			for k := 0; k < j; k++ {
				sum -= l[i][k] * l[j][k]
			}
			if i == j {
				if sum <= 0 || !isFinite(sum) {
					return nil, false
				}
				l[i][j] = math.Sqrt(sum)
			} else {
				l[i][j] = sum / l[j][j]
			}
		}
	}
	return l, true
}

// cholLogDet returns log det A = 2·Σ log L[i][i] from a Cholesky factor L.
func cholLogDet(l [][]float64) float64 {
	var s float64
	for i := range l {
		s += math.Log(l[i][i])
	}
	return 2 * s
}

// cholSolveVec solves A·x = b for x given A's Cholesky factor L (A = L·Lᵀ), by a
// forward then a back substitution.
func cholSolveVec(l [][]float64, b []float64) []float64 {
	n := len(l)
	y := make([]float64, n)
	for i := 0; i < n; i++ { // L·y = b
		s := b[i]
		for k := 0; k < i; k++ {
			s -= l[i][k] * y[k]
		}
		y[i] = s / l[i][i]
	}
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- { // Lᵀ·x = y
		s := y[i]
		for k := i + 1; k < n; k++ {
			s -= l[k][i] * x[k]
		}
		x[i] = s / l[i][i]
	}
	return x
}

// cholSolveMat solves A·X = B column by column (B is n×m, row-major).
func cholSolveMat(l [][]float64, b [][]float64) [][]float64 {
	n := len(l)
	if n == 0 {
		return nil
	}
	m := len(b[0])
	x := make([][]float64, n)
	for i := range x {
		x[i] = make([]float64, m)
	}
	col := make([]float64, n)
	for c := 0; c < m; c++ {
		for i := 0; i < n; i++ {
			col[i] = b[i][c]
		}
		sol := cholSolveVec(l, col)
		for i := 0; i < n; i++ {
			x[i][c] = sol[i]
		}
	}
	return x
}

// spdInverse returns A⁻¹ (and log det A) for a symmetric PD A, or ok=false when A
// is not positive definite.
func spdInverse(a [][]float64) (inv [][]float64, logDet float64, ok bool) {
	l, ok := cholesky(a)
	if !ok {
		return nil, 0, false
	}
	n := len(a)
	id := make([][]float64, n)
	for i := range id {
		id[i] = make([]float64, n)
		id[i][i] = 1
	}
	return cholSolveMat(l, id), cholLogDet(l), true
}

// --- moment-form Gaussian (the caller-facing distribution) ----------------

// GaussianDistribution is a proper multivariate normal in moment form —
// mean vector μ and covariance Σ — over a set of variable indices. It is both the
// observational input to Expr.EvaluateGaussian (a full joint over 0..n-1, from
// NewGaussian) and the output of GaussianFactor.Condition (the interventional
// distribution of the outcome once the intervention is fixed). It is the
// continuous, linear-Gaussian companion to the discrete Distribution.
type GaussianDistribution struct {
	vars []int
	mean []float64
	cov  [][]float64
}

// NewGaussian builds a full joint normal N(mean, cov) over variables 0..n-1.
// mean has length n; cov is an n×n symmetric positive-definite covariance
// (row-major). Every entry must be finite. Errors: ErrBadGaussian (shape,
// non-finite or non-symmetric), ErrNotPositiveDefinite (cov not PD).
func NewGaussian(mean []float64, cov [][]float64) (*GaussianDistribution, error) {
	n := len(mean)
	if n == 0 || len(cov) != n {
		return nil, ErrBadGaussian
	}
	for i := 0; i < n; i++ {
		if len(cov[i]) != n || !isFinite(mean[i]) {
			return nil, ErrBadGaussian
		}
		for j := 0; j < n; j++ {
			if !isFinite(cov[i][j]) {
				return nil, ErrBadGaussian
			}
			if math.Abs(cov[i][j]-cov[j][i]) > 1e-9*(1+math.Abs(cov[i][j])) {
				return nil, ErrBadGaussian
			}
		}
	}
	if _, ok := cholesky(cov); !ok {
		return nil, ErrNotPositiveDefinite
	}
	vars := make([]int, n)
	for i := range vars {
		vars[i] = i
	}
	return &GaussianDistribution{vars: vars, mean: cloneVec(mean), cov: cloneMat(cov)}, nil
}

// Vars returns a copy of the variable indices, ascending.
func (d *GaussianDistribution) Vars() []int { return append([]int(nil), d.vars...) }

// Mean returns a copy of the mean vector, aligned to Vars.
func (d *GaussianDistribution) Mean() []float64 { return cloneVec(d.mean) }

// Cov returns a copy of the covariance matrix, aligned to Vars.
func (d *GaussianDistribution) Cov() [][]float64 { return cloneMat(d.cov) }

// MeanAt returns the mean of variable v, or ErrBadGaussian if v is not in Vars.
func (d *GaussianDistribution) MeanAt(v int) (float64, error) {
	for k, u := range d.vars {
		if u == v {
			return d.mean[k], nil
		}
	}
	return 0, ErrBadGaussian
}

// --- canonical-form Gaussian factor (the evaluation intermediate) ---------
//
// A Gaussian factor in canonical (information) form over the ordered variable set
// `vars` is φ(x) = exp(−½·xᵀ·K·x + hᵀ·x + g). This form — not the moment (μ, Σ)
// form — is what closes the estimand algebra: products add (K, h, g), division
// subtracts them, and marginalizing a variable block out is a Schur complement.
// Those are exactly the three operations Expr is built from, so EvaluateGaussian
// walks the same AST as the discrete Evaluate. A proper Gaussian N(μ, Σ) is the
// special case K = Σ⁻¹, h = K·μ; an interventional estimand generally lands on a
// factor that is proper in the outcome but flat in the intervention (improper in
// x) — which is correct: P(y | do(x)) is a distribution over y for each fixed x,
// recovered with Condition.

// GaussianFactor is the canonical-form result of Expr.EvaluateGaussian: the
// interventional estimand P(y | do(x)) as a Gaussian factor over the query's free
// variables (Y ∪ X). Fix the intervention (and any context) with Condition to read
// the outcome's distribution, the continuous analogue of ProbAt on a Distribution.
type GaussianFactor struct {
	vars []int
	k    [][]float64 // precision (information) matrix, aligned to vars
	h    []float64   // potential vector, aligned to vars
	g    float64     // log-normalizer
}

// Vars returns a copy of the variable indices the factor ranges over, ascending.
func (f *GaussianFactor) Vars() []int { return append([]int(nil), f.vars...) }

// Condition fixes the variables in `given` to their values and returns the proper
// Gaussian distribution over the remaining variables — for an estimand factor over
// Y ∪ X, fixing X yields P(y | do(x)). Every key in `given` must be a variable of
// the factor (else ErrBadGaussian); the remaining precision block must be positive
// definite (else ErrNotPositiveDefinite). An empty `given` just converts the factor
// to moment form.
func (f *GaussianFactor) Condition(given map[int]float64) (*GaussianDistribution, error) {
	pos := make(map[int]int, len(f.vars))
	for i, v := range f.vars {
		pos[v] = i
	}
	for v, val := range given {
		if _, ok := pos[v]; !ok || !isFinite(val) {
			return nil, ErrBadGaussian
		}
	}
	var freePos []int
	var freeVars []int
	for i, v := range f.vars {
		if _, fixed := given[v]; !fixed {
			freePos = append(freePos, i)
			freeVars = append(freeVars, v)
		}
	}
	// Canonical of the conditional over the free block: precision is the free
	// sub-block; the potential absorbs the fixed values through the cross terms,
	// h'_i = h_i − Σ_{j fixed} K[i][j]·x_j. No inversion is needed to condition.
	kff := subMatrix(f.k, freePos, freePos)
	hf := make([]float64, len(freePos))
	for a, i := range freePos {
		s := f.h[i]
		for v, val := range given {
			s -= f.k[i][pos[v]] * val
		}
		hf[a] = s
	}
	if len(freePos) == 0 {
		return &GaussianDistribution{}, nil
	}
	cov, _, ok := spdInverse(kff)
	if !ok {
		return nil, ErrNotPositiveDefinite
	}
	mean := matVec(cov, hf)
	return &GaussianDistribution{vars: freeVars, mean: mean, cov: cov}, nil
}

// canonicalFromMoment converts a proper Gaussian N(mean, cov) over `vars` into a
// canonical factor. cov must be PD (validated by the caller).
func canonicalFromMoment(vars []int, mean []float64, cov [][]float64) (*gaussCanon, error) {
	k, logDetCov, ok := spdInverse(cov)
	if !ok {
		return nil, ErrNotPositiveDefinite
	}
	h := matVec(k, mean)
	n := len(vars)
	// g = −½·μᵀ·K·μ − ½·n·log(2π) − ½·log det Σ, so exp(g)·exp(−½xᵀKx+hᵀx) integrates to 1.
	g := -0.5*dot(mean, h) - 0.5*float64(n)*log2Pi - 0.5*logDetCov
	return &gaussCanon{vars: append([]int(nil), vars...), k: k, h: h, g: g}, nil
}

// gaussCanon is the internal working factor the evaluator threads through the AST;
// GaussianFactor is its exported wrapper (the query result).
type gaussCanon struct {
	vars []int
	k    [][]float64
	h    []float64
	g    float64
}

// gaussMul multiplies two canonical factors: over the union of their scopes the
// precision matrices, potentials and log-normalizers add.
func gaussMul(a, b *gaussCanon) *gaussCanon {
	union := sortedUnionInts(a.vars, b.vars)
	pos := posOf(union)
	k := zeros(len(union))
	h := make([]float64, len(union))
	addInto(k, h, a, pos)
	addInto(k, h, b, pos)
	return &gaussCanon{vars: union, k: k, h: h, g: a.g + b.g}
}

// gaussDivide divides num by den (den's scope must be a subset of num's): over
// num's scope the precision, potential and log-normalizer are subtracted. This is
// the ratio P(A, B) / P(B) that a conditional reduces to.
func gaussDivide(num, den *gaussCanon) *gaussCanon {
	pos := posOf(num.vars)
	k := cloneMat(num.k)
	h := cloneVec(num.h)
	dp := make([]int, len(den.vars))
	for a, v := range den.vars {
		dp[a] = pos[v]
	}
	for a := range den.vars {
		h[dp[a]] -= den.h[a]
		for b := range den.vars {
			k[dp[a]][dp[b]] -= den.k[a][b]
		}
	}
	return &gaussCanon{vars: append([]int(nil), num.vars...), k: k, h: h, g: num.g - den.g}
}

// gaussMarginalize integrates the variables in `over` out of f, returning a factor
// on the remaining variables. Partitioning K into the kept block (x) and the
// summed-out block (y), the marginal precision is the Schur complement
// K/Kyy = Kxx − Kxy·Kyy⁻¹·Kyx, the potential is hx − Kxy·Kyy⁻¹·hy, and the scalar
// picks up ½·(|y|·log 2π − log det Kyy + hyᵀ·Kyy⁻¹·hy). Kyy must be positive
// definite (the block being integrated out is a genuine, normalizable Gaussian);
// otherwise ErrNotPositiveDefinite.
func gaussMarginalize(f *gaussCanon, over map[int]bool) (*gaussCanon, error) {
	var keepPos, overPos []int
	var keepVars []int
	for i, v := range f.vars {
		if over[v] {
			overPos = append(overPos, i)
		} else {
			keepPos = append(keepPos, i)
			keepVars = append(keepVars, v)
		}
	}
	if len(overPos) == 0 {
		return &gaussCanon{vars: keepVars, k: cloneMat(f.k), h: cloneVec(f.h), g: f.g}, nil
	}
	kyy := subMatrix(f.k, overPos, overPos)
	l, ok := cholesky(kyy)
	if !ok {
		return nil, ErrNotPositiveDefinite
	}
	hy := subVec(f.h, overPos)
	kxx := subMatrix(f.k, keepPos, keepPos)
	kxy := subMatrix(f.k, keepPos, overPos) // |keep| × |over|
	kyx := subMatrix(f.k, overPos, keepPos) // |over| × |keep|
	hx := subVec(f.h, keepPos)

	// scalar and potential corrections through Kyy⁻¹
	u := cholSolveVec(l, hy) // Kyy⁻¹·hy
	m := len(over)
	g := f.g + 0.5*(float64(m)*log2Pi-cholLogDet(l)+dot(hy, u))
	h := make([]float64, len(keepPos))
	for i := range keepPos {
		h[i] = hx[i] - dot(kxy[i], u)
	}
	// precision correction Kxy·Kyy⁻¹·Kyx
	var k [][]float64
	if len(keepPos) == 0 {
		k = zeros(0)
	} else {
		w := cholSolveMat(l, kyx) // Kyy⁻¹·Kyx, |over| × |keep|
		k = make([][]float64, len(keepPos))
		for i := range keepPos {
			k[i] = make([]float64, len(keepPos))
			for j := range keepPos {
				var c float64
				for t := 0; t < m; t++ {
					c += kxy[i][t] * w[t][j]
				}
				k[i][j] = kxx[i][j] - c
			}
		}
	}
	return &gaussCanon{vars: keepVars, k: k, h: h, g: g}, nil
}

// --- the AST evaluator ----------------------------------------------------

// EvaluateGaussian computes the estimand numerically against a linear-Gaussian
// observational joint `joint` over the whole variable set (from NewGaussian),
// returning the interventional distribution P(y | do(x)) as a Gaussian factor over
// the query's free variables (Y ∪ X). It is the continuous companion to Evaluate:
// where Evaluate reads a discrete joint and returns a probability table, this reads
// a normal joint and returns a Gaussian factor, exact for a linear-Gaussian model.
// Fix the intervention with GaussianFactor.Condition to obtain P(y | do(x)) as a
// proper Gaussian over the outcome.
//
// The estimand is the same symbolic Expr the ID/IDC algorithms produce; only the
// numeric backend differs (a canonical-form Gaussian factor algebra instead of
// dense probability tables). joint must be a full joint over 0..n-1 (its Vars is
// 0,1,…), else ErrBadGaussian. A degenerate joint or intermediate factor whose
// eliminated block is not positive definite yields ErrNotPositiveDefinite.
func (e *Expr) EvaluateGaussian(joint *GaussianDistribution) (*GaussianFactor, error) {
	for i, v := range joint.vars {
		if v != i {
			return nil, ErrBadGaussian // must be a full joint over 0..n-1
		}
	}
	base, err := canonicalFromMoment(joint.vars, joint.mean, joint.cov)
	if err != nil {
		return nil, err
	}
	res, err := evalGauss(e, base)
	if err != nil {
		return nil, err
	}
	return &GaussianFactor{vars: res.vars, k: res.k, h: res.h, g: res.g}, nil
}

func evalGauss(e *Expr, base *gaussCanon) (*gaussCanon, error) {
	switch e.kind {
	case exprJoint:
		return base, nil
	case exprMarginal:
		child, err := evalGauss(e.child, base)
		if err != nil {
			return nil, err
		}
		over := map[int]bool{}
		for _, v := range e.over {
			over[v] = true
		}
		return gaussMarginalize(child, over)
	case exprProduct:
		var acc *gaussCanon
		for _, f := range e.factors {
			t, err := evalGauss(f, base)
			if err != nil {
				return nil, err
			}
			if acc == nil {
				acc = t
			} else {
				acc = gaussMul(acc, t)
			}
		}
		if acc == nil { // empty product = the scalar 1 over no variables
			return &gaussCanon{}, nil
		}
		return acc, nil
	case exprConditional:
		b, err := evalGauss(e.base, base)
		if err != nil {
			return nil, err
		}
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
		num, err := gaussMarginalize(b, numOver)
		if err != nil {
			return nil, err
		}
		den, err := gaussMarginalize(b, denOver)
		if err != nil {
			return nil, err
		}
		return gaussDivide(num, den), nil
	case exprRatio:
		num, err := evalGauss(e.num, base)
		if err != nil {
			return nil, err
		}
		den, err := evalGauss(e.den, base)
		if err != nil {
			return nil, err
		}
		return gaussDivide(num, den), nil
	}
	return nil, ErrBadGaussian
}

// --- small dense helpers --------------------------------------------------

func posOf(vars []int) map[int]int {
	pos := make(map[int]int, len(vars))
	for i, v := range vars {
		pos[v] = i
	}
	return pos
}

// addInto accumulates a factor's precision and potential into a target laid out on
// `pos` (variable id → row/col in the target).
func addInto(k [][]float64, h []float64, f *gaussCanon, pos map[int]int) {
	idx := make([]int, len(f.vars))
	for a, v := range f.vars {
		idx[a] = pos[v]
	}
	for a := range f.vars {
		h[idx[a]] += f.h[a]
		for b := range f.vars {
			k[idx[a]][idx[b]] += f.k[a][b]
		}
	}
}

func subMatrix(m [][]float64, rows, cols []int) [][]float64 {
	out := make([][]float64, len(rows))
	for a, r := range rows {
		out[a] = make([]float64, len(cols))
		for b, c := range cols {
			out[a][b] = m[r][c]
		}
	}
	return out
}

func subVec(v []float64, idx []int) []float64 {
	out := make([]float64, len(idx))
	for a, i := range idx {
		out[a] = v[i]
	}
	return out
}

func matVec(m [][]float64, v []float64) []float64 {
	out := make([]float64, len(m))
	for i := range m {
		var s float64
		for j := range v {
			s += m[i][j] * v[j]
		}
		out[i] = s
	}
	return out
}

func dot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func zeros(n int) [][]float64 {
	m := make([][]float64, n)
	for i := range m {
		m[i] = make([]float64, n)
	}
	return m
}

func cloneMat(m [][]float64) [][]float64 {
	out := make([][]float64, len(m))
	for i := range m {
		out[i] = append([]float64(nil), m[i]...)
	}
	return out
}

func cloneVec(v []float64) []float64 { return append([]float64(nil), v...) }
