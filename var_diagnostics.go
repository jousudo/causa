package causa

import (
	"errors"
	"math"
	"math/cmplx"
	"sort"
)

const (
	// DefaultVARStabilityTolerance is the exclusion band around the unit circle.
	// A fitted root inside this band is reported as indeterminate rather than
	// being classified from numerically insignificant distance alone.
	DefaultVARStabilityTolerance = 1e-8
	// DefaultMaxVARCompanionCells bounds the complex companion matrix used by
	// the stability eigensolver. The default admits dimensions through 64;
	// callers must explicitly review CPU and memory budgets before raising it.
	DefaultMaxVARCompanionCells = 1 << 12
	// DefaultMaxVAREigenIterations bounds shifted QR iterations after the
	// companion matrix has been reduced to upper-Hessenberg form.
	DefaultMaxVAREigenIterations = 10_000
)

var (
	// ErrInvalidVARStabilityOptions is returned for a non-finite/negative
	// boundary tolerance or a negative iteration budget.
	ErrInvalidVARStabilityOptions = errors.New("causa: invalid VAR stability options")
	// ErrVARCompanionTooLarge is returned before an oversized companion matrix
	// is allocated.
	ErrVARCompanionTooLarge = errors.New("causa: VAR companion matrix exceeds the configured cell budget")
	// ErrVAREigenNoConvergence is returned when the shifted-QR eigensolver does
	// not converge within the configured iteration budget.
	ErrVAREigenNoConvergence = errors.New("causa: VAR companion eigensolver did not converge")
	// ErrInvalidWhitenessLag is returned unless the tested maximum lag is larger
	// than the fitted VAR order and shorter than the residual series.
	ErrInvalidWhitenessLag = errors.New("causa: whiteness maximum lag must exceed VAR order and be shorter than residual series")
)

// VARStabilityStatus is a three-way numerical classification of a fitted VAR.
// Indeterminate is intentional: a root within the configured unit-circle band
// must not be promoted into a reliable stable/unstable claim.
type VARStabilityStatus uint8

const (
	VARStabilityIndeterminate VARStabilityStatus = iota
	VARStable
	VARUnstable
)

func (s VARStabilityStatus) String() string {
	switch s {
	case VARStable:
		return "stable"
	case VARUnstable:
		return "unstable"
	case VARStabilityIndeterminate:
		return "indeterminate"
	default:
		return "unknown"
	}
}

// VARStabilityOptions configures the companion-matrix stability calculation.
// The zero value selects conservative defaults.
type VARStabilityOptions struct {
	// BoundaryTolerance defines an exclusion band [1-tol,1+tol] around the unit
	// circle. Zero selects DefaultVARStabilityTolerance.
	BoundaryTolerance float64
	// MaxCompanionCells bounds dimension squared before allocation. Zero selects
	// DefaultMaxVARCompanionCells; a negative value removes this explicit cap.
	MaxCompanionCells int
	// MaxIterations bounds shifted QR iterations. Zero selects
	// DefaultMaxVAREigenIterations. Negative values are invalid.
	MaxIterations int
}

// VARStabilityResult describes the roots of the fitted VAR(1) companion
// representation. A covariance-stationary VAR requires every companion
// eigenvalue strictly inside the unit circle.
type VARStabilityResult struct {
	Status         VARStabilityStatus
	SpectralRadius float64
	// Margin is 1-SpectralRadius. Positive is inside the unit circle.
	Margin      float64
	Tolerance   float64
	Iterations  int
	eigenvalues []complex128
}

// Eigenvalues returns a defensive copy sorted by decreasing modulus, then real
// and imaginary parts for deterministic reporting.
func (r *VARStabilityResult) Eigenvalues() []complex128 {
	return append([]complex128(nil), r.eigenvalues...)
}

// Stability computes the eigenvalues of the VAR(1) companion representation
// with a stdlib-only complex shifted-QR eigensolver. The result distinguishes a
// stable system, an unstable system, and a numerically indeterminate unit-root
// boundary band.
//
// This is a diagnostic of the fitted linear dynamics, not a unit-root test on
// the data-generating process and not evidence of structural causality.
func (m *VARModel) Stability(opts *VARStabilityOptions) (*VARStabilityResult, error) {
	tolerance := DefaultVARStabilityTolerance
	maxCells := DefaultMaxVARCompanionCells
	maxIterations := DefaultMaxVAREigenIterations
	if opts != nil {
		if opts.BoundaryTolerance != 0 {
			tolerance = opts.BoundaryTolerance
		}
		if opts.MaxCompanionCells != 0 {
			maxCells = opts.MaxCompanionCells
		}
		if opts.MaxIterations != 0 {
			maxIterations = opts.MaxIterations
		}
	}
	if !isFinite(tolerance) || tolerance < 0 || maxIterations < 0 {
		return nil, ErrInvalidVARStabilityOptions
	}
	variables := m.Variables()
	lags := m.Lags()
	maxInt := int(^uint(0) >> 1)
	if variables < 1 || lags < 1 || variables > maxInt/lags {
		return nil, ErrVARCompanionTooLarge
	}
	dimension := variables * lags
	if dimension > maxInt/dimension {
		return nil, ErrVARCompanionTooLarge
	}
	cells := dimension * dimension
	if maxCells > 0 && cells > maxCells {
		return nil, ErrVARCompanionTooLarge
	}
	companion := make([][]complex128, dimension)
	for row := range companion {
		companion[row] = make([]complex128, dimension)
	}
	k := m.Variables()
	for lag := 0; lag < m.Lags(); lag++ {
		for response := 0; response < k; response++ {
			for predictor := 0; predictor < k; predictor++ {
				companion[response][lag*k+predictor] = complex(m.coef[lag][response][predictor], 0)
			}
		}
	}
	for block := 1; block < m.Lags(); block++ {
		for variable := 0; variable < k; variable++ {
			companion[block*k+variable][(block-1)*k+variable] = 1
		}
	}
	eigenvalues, iterations, err := shiftedQREigenvalues(companion, maxIterations)
	if err != nil {
		return nil, err
	}
	sort.SliceStable(eigenvalues, func(i, j int) bool {
		mi, mj := cmplx.Abs(eigenvalues[i]), cmplx.Abs(eigenvalues[j])
		if mi != mj {
			return mi > mj
		}
		if real(eigenvalues[i]) != real(eigenvalues[j]) {
			return real(eigenvalues[i]) > real(eigenvalues[j])
		}
		return imag(eigenvalues[i]) > imag(eigenvalues[j])
	})
	radius := 0.0
	for _, eigenvalue := range eigenvalues {
		if modulus := cmplx.Abs(eigenvalue); modulus > radius {
			radius = modulus
		}
	}
	margin := 1 - radius
	status := VARStabilityIndeterminate
	switch {
	case margin > tolerance:
		status = VARStable
	case margin < -tolerance:
		status = VARUnstable
	}
	return &VARStabilityResult{
		Status: status, SpectralRadius: radius, Margin: margin,
		Tolerance: tolerance, Iterations: iterations,
		eigenvalues: append([]complex128(nil), eigenvalues...),
	}, nil
}

// VARWhitenessResult holds the multivariate Portmanteau test of the fitted
// innovations. The null hypothesis is zero residual autocovariance through
// MaxLag. A small PValue rejects residual whiteness and warns that lagged
// predictive tests from this fit are misspecified.
type VARWhitenessResult struct {
	Statistic           float64
	PValue              float64
	DegreesOfFreedom    int
	MaxLag              int
	VARLags             int
	Observations        int
	Variables           int
	SmallSampleAdjusted bool
}

// Reject reports whether residual whiteness is rejected at alpha.
func (r *VARWhitenessResult) Reject(alpha float64) (bool, error) {
	if !isFinite(alpha) || alpha <= 0 || alpha >= 1 {
		return false, ErrInvalidAlpha
	}
	return r.PValue < alpha, nil
}

// WhitenessTest performs the multivariate Portmanteau residual test through
// maxLag. maxLag must exceed the fitted VAR order because the asymptotic chi-
// square degrees of freedom are k²(maxLag-p). When adjusted is true, each lag
// term is divided by T-lag and the total is scaled by T², the Lütkepohl small-
// sample adjustment.
//
// Failure to reject does not prove independence or correct model structure; it
// only means this finite-lag autocovariance test did not detect remaining linear
// serial correlation.
func (m *VARModel) WhitenessTest(maxLag int, adjusted bool) (*VARWhitenessResult, error) {
	if len(m.residuals) == 0 || maxLag <= m.Lags() || maxLag >= m.Observations() {
		return nil, ErrInvalidWhitenessLag
	}
	k := m.Variables()
	t := m.Observations()
	centered := cloneMatrix(m.residuals)
	for variable := 0; variable < k; variable++ {
		var mean float64
		for _, value := range centered[variable] {
			mean += value
		}
		mean /= float64(t)
		for observation := range centered[variable] {
			centered[variable][observation] -= mean
		}
	}
	covariance0 := residualAutocovariance(centered, 0)
	inverse, _, ok := spdInverse(covariance0)
	if !ok {
		return nil, ErrSingular
	}
	statistic := 0.0
	for lag := 1; lag <= maxLag; lag++ {
		covariance := residualAutocovariance(centered, lag)
		standardized := multiplyRealMatrices(multiplyRealMatrices(inverse, covariance), inverse)
		term := 0.0
		for i := 0; i < k; i++ {
			for j := 0; j < k; j++ {
				term += covariance[i][j] * standardized[i][j]
			}
		}
		if term < 0 && term > -1e-12 {
			term = 0
		}
		if !isFinite(term) || term < 0 {
			return nil, ErrSingular
		}
		if adjusted {
			term /= float64(t - lag)
		}
		statistic += term
	}
	if adjusted {
		statistic *= float64(t) * float64(t)
	} else {
		statistic *= float64(t)
	}
	lagDifference := maxLag - m.Lags()
	maxInt := int(^uint(0) >> 1)
	if k > maxInt/k || k*k > maxInt/lagDifference {
		return nil, ErrVARDesignTooLarge
	}
	df := k * k * lagDifference
	pValue := chiSquareUpperTail(statistic, float64(df))
	if !isFinite(pValue) {
		return nil, ErrSingular
	}
	return &VARWhitenessResult{
		Statistic: statistic, PValue: pValue, DegreesOfFreedom: df,
		MaxLag: maxLag, VARLags: m.Lags(), Observations: t, Variables: k,
		SmallSampleAdjusted: adjusted,
	}, nil
}

func residualAutocovariance(residuals [][]float64, lag int) [][]float64 {
	k := len(residuals)
	t := len(residuals[0])
	out := make([][]float64, k)
	for i := 0; i < k; i++ {
		out[i] = make([]float64, k)
		for j := 0; j < k; j++ {
			var sum float64
			for observation := lag; observation < t; observation++ {
				sum += residuals[i][observation] * residuals[j][observation-lag]
			}
			out[i][j] = sum / float64(t)
		}
	}
	return out
}

func multiplyRealMatrices(a, b [][]float64) [][]float64 {
	n := len(a)
	m := len(b[0])
	inner := len(b)
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		out[i] = make([]float64, m)
		for k := 0; k < inner; k++ {
			value := a[i][k]
			for j := 0; j < m; j++ {
				out[i][j] += value * b[k][j]
			}
		}
	}
	return out
}

func shiftedQREigenvalues(matrix [][]complex128, maxIterations int) ([]complex128, int, error) {
	n := len(matrix)
	if n == 1 {
		return []complex128{matrix[0][0]}, 0, nil
	}
	h := cloneComplexMatrix(matrix)
	reduceToUpperHessenberg(h)
	eigenvalues := make([]complex128, n)
	active := n
	iterations := 0
	sinceDeflation := 0
	for active > 0 {
		if active == 1 {
			eigenvalues[0] = h[0][0]
			break
		}
		if active == 2 {
			first, second := eigenvalues2x2(h[0][0], h[0][1], h[1][0], h[1][1])
			eigenvalues[0], eigenvalues[1] = first, second
			break
		}
		scale := cmplx.Abs(h[active-2][active-2]) + cmplx.Abs(h[active-1][active-1])
		if scale == 0 {
			scale = complexMatrixNorm(h, active)
		}
		if cmplx.Abs(h[active-1][active-2]) <= 1e-13*math.Max(1, scale) {
			h[active-1][active-2] = 0
			eigenvalues[active-1] = h[active-1][active-1]
			active--
			sinceDeflation = 0
			continue
		}
		if iterations >= maxIterations {
			return nil, iterations, ErrVAREigenNoConvergence
		}
		first, second := eigenvalues2x2(
			h[active-2][active-2], h[active-2][active-1],
			h[active-1][active-2], h[active-1][active-1],
		)
		var shift complex128
		if cmplx.Abs(first-h[active-1][active-1]) < cmplx.Abs(second-h[active-1][active-1]) {
			shift = first
		} else {
			shift = second
		}
		sinceDeflation++
		if sinceDeflation%50 == 0 {
			norm := complexMatrixNorm(h, active)
			shift += complex(0.071*norm, 0.053*norm)
		}
		block := make([][]complex128, active)
		for i := 0; i < active; i++ {
			block[i] = append([]complex128(nil), h[i][:active]...)
			block[i][i] -= shift
		}
		q, r := complexQR(block)
		next := multiplyComplexMatrices(r, q)
		for i := 0; i < active; i++ {
			next[i][i] += shift
			for j := 0; j < active; j++ {
				h[i][j] = next[i][j]
			}
		}
		for i := 2; i < active; i++ {
			for j := 0; j < i-1; j++ {
				h[i][j] = 0
			}
		}
		iterations++
	}
	for _, value := range eigenvalues {
		if !isFinite(real(value)) || !isFinite(imag(value)) {
			return nil, iterations, ErrVAREigenNoConvergence
		}
	}
	return eigenvalues, iterations, nil
}

func reduceToUpperHessenberg(a [][]complex128) {
	n := len(a)
	for column := 0; column < n-2; column++ {
		v, ok := complexHouseholderVector(a, column+1, column)
		if !ok {
			continue
		}
		start := column + 1
		for j := column; j < n; j++ {
			var dot complex128
			for i := range v {
				dot += cmplx.Conj(v[i]) * a[start+i][j]
			}
			for i := range v {
				a[start+i][j] -= 2 * v[i] * dot
			}
		}
		for i := 0; i < n; i++ {
			var dot complex128
			for j := range v {
				dot += a[i][start+j] * v[j]
			}
			for j := range v {
				a[i][start+j] -= 2 * dot * cmplx.Conj(v[j])
			}
		}
		for row := column + 2; row < n; row++ {
			a[row][column] = 0
		}
	}
}

func complexHouseholderVector(a [][]complex128, rowStart, column int) ([]complex128, bool) {
	v := make([]complex128, len(a)-rowStart)
	normSquared := 0.0
	for i := range v {
		v[i] = a[rowStart+i][column]
		modulus := cmplx.Abs(v[i])
		normSquared += modulus * modulus
	}
	norm := math.Sqrt(normSquared)
	if norm == 0 {
		return nil, false
	}
	phase := complex(1, 0)
	if cmplx.Abs(v[0]) != 0 {
		phase = v[0] / complex(cmplx.Abs(v[0]), 0)
	}
	v[0] += phase * complex(norm, 0)
	vNormSquared := 0.0
	for _, value := range v {
		modulus := cmplx.Abs(value)
		vNormSquared += modulus * modulus
	}
	vNorm := math.Sqrt(vNormSquared)
	for i := range v {
		v[i] /= complex(vNorm, 0)
	}
	return v, true
}

func complexQR(a [][]complex128) (q, r [][]complex128) {
	n := len(a)
	r = cloneComplexMatrix(a)
	q = make([][]complex128, n)
	for i := range q {
		q[i] = make([]complex128, n)
		q[i][i] = 1
	}
	for column := 0; column < n; column++ {
		v, ok := complexHouseholderVector(r, column, column)
		if !ok {
			continue
		}
		for j := column; j < n; j++ {
			var dot complex128
			for i := range v {
				dot += cmplx.Conj(v[i]) * r[column+i][j]
			}
			for i := range v {
				r[column+i][j] -= 2 * v[i] * dot
			}
		}
		for row := 0; row < n; row++ {
			var dot complex128
			for j := range v {
				dot += q[row][column+j] * v[j]
			}
			for j := range v {
				q[row][column+j] -= 2 * dot * cmplx.Conj(v[j])
			}
		}
		for row := column + 1; row < n; row++ {
			r[row][column] = 0
		}
	}
	return q, r
}

func eigenvalues2x2(a, b, c, d complex128) (complex128, complex128) {
	trace := a + d
	determinant := a*d - b*c
	discriminant := cmplx.Sqrt(trace*trace - 4*determinant)
	first := (trace + discriminant) / 2
	second := (trace - discriminant) / 2
	// Recover the smaller root from the determinant. This avoids cancellation
	// when the two eigenvalue magnitudes differ by many orders.
	if cmplx.Abs(first) >= cmplx.Abs(second) && first != 0 {
		second = determinant / first
	} else if second != 0 {
		first = determinant / second
	}
	return first, second
}

func multiplyComplexMatrices(a, b [][]complex128) [][]complex128 {
	n := len(a)
	out := make([][]complex128, n)
	for i := 0; i < n; i++ {
		out[i] = make([]complex128, n)
		for k := 0; k < n; k++ {
			value := a[i][k]
			for j := 0; j < n; j++ {
				out[i][j] += value * b[k][j]
			}
		}
	}
	return out
}

func cloneComplexMatrix(a [][]complex128) [][]complex128 {
	out := make([][]complex128, len(a))
	for i := range a {
		out[i] = append([]complex128(nil), a[i]...)
	}
	return out
}

func complexMatrixNorm(a [][]complex128, active int) float64 {
	maximum := 0.0
	for i := 0; i < active; i++ {
		rowSum := 0.0
		for j := 0; j < active; j++ {
			rowSum += cmplx.Abs(a[i][j])
		}
		if rowSum > maximum {
			maximum = rowSum
		}
	}
	return maximum
}

func chiSquareUpperTail(statistic, degreesOfFreedom float64) float64 {
	if statistic < 0 || degreesOfFreedom <= 0 || !isFinite(statistic) || !isFinite(degreesOfFreedom) {
		return math.NaN()
	}
	if statistic == 0 {
		return 1
	}
	return regularizedGammaQ(degreesOfFreedom/2, statistic/2)
}

func regularizedGammaQ(a, x float64) float64 {
	if a <= 0 || x < 0 || !isFinite(a) || !isFinite(x) {
		return math.NaN()
	}
	logGamma, _ := math.Lgamma(a)
	const epsilon = 1e-14
	if x < a+1 {
		ap := a
		delta := 1 / a
		sum := delta
		for iteration := 0; iteration < 100_000; iteration++ {
			ap++
			delta *= x / ap
			sum += delta
			if math.Abs(delta) <= math.Abs(sum)*epsilon {
				p := sum * math.Exp(-x+a*math.Log(x)-logGamma)
				return math.Max(0, math.Min(1, 1-p))
			}
		}
		return math.NaN()
	}
	b := x + 1 - a
	const floor = 1e-300
	if math.Abs(b) < floor {
		b = floor
	}
	c := 1 / floor
	d := 1 / b
	h := d
	for iteration := 1; iteration <= 100_000; iteration++ {
		fi := float64(iteration)
		an := -fi * (fi - a)
		b += 2
		d = an*d + b
		if math.Abs(d) < floor {
			d = floor
		}
		c = b + an/c
		if math.Abs(c) < floor {
			c = floor
		}
		d = 1 / d
		delta := d * c
		h *= delta
		if math.Abs(delta-1) <= epsilon {
			q := math.Exp(-x+a*math.Log(x)-logGamma) * h
			return math.Max(0, math.Min(1, q))
		}
	}
	return math.NaN()
}
