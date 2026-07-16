package causa

import "math"

// olsResult is the outcome of an ordinary-least-squares fit: the estimated
// coefficients and the residual sum of squares (RSS) of the fitted model.
type olsResult struct {
	coef []float64
	rss  float64
}

// fitOLS solves the linear least-squares problem
//
//	min_b ||X·b − y||₂
//
// for a design matrix x (n rows, k columns, row-major) and response y (length
// n), returning the coefficient vector and the residual sum of squares.
//
// The solver uses an in-place Householder QR factorization of X rather than
// the normal equations (XᵀX)b = Xᵀy. Forming XᵀX squares the condition number
// of the problem and loses precision on collinear regressors — exactly the
// situation a causality test provokes when it stacks a series and its own
// lags. Householder reflections are backward stable, and they expose rank
// deficiency directly: if a pivot sub-column collapses to (near) zero after
// elimination by the earlier reflections, the design is numerically singular.
//
// fitOLS returns ErrSingular in that case — which is what a constant or
// perfectly collinear regressor produces — instead of returning garbage
// coefficients from an ill-posed back-substitution.
//
// The RSS is read straight off the transformed response: after the k
// reflections, QᵀX has zeros below the diagonal in its first k rows, and the
// squared norm of the trailing (n−k) entries of Qᵀy is the minimized residual
// norm. This is both cheaper and more accurate than forming residuals from the
// coefficients.
//
// x is not modified; fitOLS works on internal copies.
func fitOLS(x [][]float64, y []float64) (olsResult, error) {
	n := len(x)
	if n == 0 {
		return olsResult{}, ErrSingular
	}
	k := len(x[0])
	if k == 0 || k > n {
		return olsResult{}, ErrSingular
	}

	// Work on copies so the caller's slices are untouched. r starts as X and is
	// transformed in place into R (upper triangle) plus the Householder vectors
	// (lower triangle, transient). qty accumulates Qᵀy.
	r := make([][]float64, n)
	for i := range x {
		r[i] = append([]float64(nil), x[i]...)
	}
	qty := append([]float64(nil), y...)

	// Original column norms give a relative tolerance for rank deficiency: a
	// pivot is singular when the remaining sub-column is negligible compared to
	// where that column started, independent of the data's absolute scale.
	colScale := make([]float64, k)
	for j := 0; j < k; j++ {
		var s float64
		for i := 0; i < n; i++ {
			s += r[i][j] * r[i][j]
		}
		colScale[j] = math.Sqrt(s)
	}

	const tol = 1e-11
	for j := 0; j < k; j++ {
		// Norm of the sub-column r[j:n][j] that this reflection zeroes below the
		// diagonal.
		var norm float64
		for i := j; i < n; i++ {
			norm += r[i][j] * r[i][j]
		}
		norm = math.Sqrt(norm)

		scale := colScale[j]
		if scale == 0 {
			scale = 1
		}
		if norm <= tol*scale {
			return olsResult{}, ErrSingular
		}

		// Reflect the sub-column onto ±norm·e₁. Choosing the sign opposite to
		// r[j][j] avoids cancellation when forming the Householder vector.
		alpha := -norm
		if r[j][j] < 0 {
			alpha = norm
		}
		vjj := r[j][j] - alpha
		vnorm2 := vjj * vjj
		for i := j + 1; i < n; i++ {
			vnorm2 += r[i][j] * r[i][j]
		}
		if vnorm2 == 0 {
			return olsResult{}, ErrSingular
		}

		// Apply H = I − 2·v·vᵀ / (vᵀv) to the remaining columns and to qty.
		// v is (vjj, r[j+1..n-1][j]).
		for c := j + 1; c < k; c++ {
			dot := vjj * r[j][c]
			for i := j + 1; i < n; i++ {
				dot += r[i][j] * r[i][c]
			}
			f := 2 * dot / vnorm2
			r[j][c] -= f * vjj
			for i := j + 1; i < n; i++ {
				r[i][c] -= f * r[i][j]
			}
		}
		dot := vjj * qty[j]
		for i := j + 1; i < n; i++ {
			dot += r[i][j] * qty[i]
		}
		f := 2 * dot / vnorm2
		qty[j] -= f * vjj
		for i := j + 1; i < n; i++ {
			qty[i] -= f * r[i][j]
		}

		// Commit the R diagonal; entries below are done being used as v.
		r[j][j] = alpha
	}

	// Back-substitution R·b = (Qᵀy)[0:k].
	coef := make([]float64, k)
	for i := k - 1; i >= 0; i-- {
		s := qty[i]
		for j := i + 1; j < k; j++ {
			s -= r[i][j] * coef[j]
		}
		coef[i] = s / r[i][i]
	}

	// RSS = ||(Qᵀy)[k:n]||².
	var rss float64
	for i := k; i < n; i++ {
		rss += qty[i] * qty[i]
	}
	return olsResult{coef: coef, rss: rss}, nil
}
