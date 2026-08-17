package causa

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

// independentADFStat recomputes the ADF t-statistic through a fully independent
// route: it forms the normal equations XᵀX and Xᵀy and solves them with an
// explicit Gauss-Jordan inverse, then reads ρ̂ and its standard error straight
// off the coefficient covariance σ̂²·(XᵀX)⁻¹. ADFTest instead uses the package's
// Householder-QR RSS route, so agreement between the two is an oracle on the
// statistic that shares no linear-algebra code with the implementation.
func independentADFStat(t *testing.T, y []float64, lag int, trend ADFTrend) float64 {
	t.Helper()
	det := 0
	switch trend {
	case ADFConstant:
		det = 1
	case ADFConstantTrend:
		det = 2
	}
	k := det + 1 + lag
	rhoCol := det
	rows := len(y) - lag - 1

	x := make([][]float64, rows)
	yr := make([]float64, rows)
	for r := 0; r < rows; r++ {
		tt := r + lag + 1
		yr[r] = y[tt] - y[tt-1]
		row := make([]float64, k)
		c := 0
		if trend == ADFConstant || trend == ADFConstantTrend {
			row[c] = 1
			c++
		}
		if trend == ADFConstantTrend {
			row[c] = float64(r + 1)
			c++
		}
		row[c] = y[tt-1]
		c++
		for i := 1; i <= lag; i++ {
			row[c] = y[tt-i] - y[tt-i-1]
			c++
		}
		x[r] = row
	}

	xtx := make([][]float64, k)
	xty := make([]float64, k)
	for a := 0; a < k; a++ {
		xtx[a] = make([]float64, k)
		for b := 0; b < k; b++ {
			var s float64
			for r := 0; r < rows; r++ {
				s += x[r][a] * x[r][b]
			}
			xtx[a][b] = s
		}
		var s float64
		for r := 0; r < rows; r++ {
			s += x[r][a] * yr[r]
		}
		xty[a] = s
	}
	inv := invert(t, xtx)

	beta := make([]float64, k)
	for a := 0; a < k; a++ {
		var s float64
		for b := 0; b < k; b++ {
			s += inv[a][b] * xty[b]
		}
		beta[a] = s
	}
	var rss float64
	for r := 0; r < rows; r++ {
		var pred float64
		for a := 0; a < k; a++ {
			pred += x[r][a] * beta[a]
		}
		e := yr[r] - pred
		rss += e * e
	}
	sigma2 := rss / float64(rows-k)
	se := math.Sqrt(sigma2 * inv[rhoCol][rhoCol])
	return beta[rhoCol] / se
}

// invert returns the inverse of the square matrix m via Gauss-Jordan elimination
// with partial pivoting. Test-only; m is small (k <= a handful of columns).
func invert(t *testing.T, m [][]float64) [][]float64 {
	t.Helper()
	n := len(m)
	a := make([][]float64, n)
	inv := make([][]float64, n)
	for i := range m {
		a[i] = append([]float64(nil), m[i]...)
		inv[i] = make([]float64, n)
		inv[i][i] = 1
	}
	for col := 0; col < n; col++ {
		piv := col
		for r := col + 1; r < n; r++ {
			if math.Abs(a[r][col]) > math.Abs(a[piv][col]) {
				piv = r
			}
		}
		a[col], a[piv] = a[piv], a[col]
		inv[col], inv[piv] = inv[piv], inv[col]
		d := a[col][col]
		if d == 0 {
			t.Fatalf("singular matrix in test inverse")
		}
		for j := 0; j < n; j++ {
			a[col][j] /= d
			inv[col][j] /= d
		}
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := a[r][col]
			for j := 0; j < n; j++ {
				a[r][j] -= f * a[col][j]
				inv[r][j] -= f * inv[col][j]
			}
		}
	}
	return inv
}

// randomWalk and stationaryAR1 build reproducible test series.
func randomWalk(seed int64, n int) []float64 {
	rng := rand.New(rand.NewSource(seed))
	y := make([]float64, n)
	for i := 1; i < n; i++ {
		y[i] = y[i-1] + rng.NormFloat64()
	}
	return y
}

func stationaryAR1(seed int64, n int, phi float64) []float64 {
	rng := rand.New(rand.NewSource(seed))
	y := make([]float64, n)
	for i := 1; i < n; i++ {
		y[i] = phi*y[i-1] + rng.NormFloat64()
	}
	return y
}

// TestADFMatchesIndependentNormalEquations validates the statistic against the
// independent normal-equations oracle across all three trend specifications and
// several lags, on a fixed stationary series.
func TestADFMatchesIndependentNormalEquations(t *testing.T) {
	y := stationaryAR1(11, 200, 0.4)
	for _, trend := range []ADFTrend{ADFNone, ADFConstant, ADFConstantTrend} {
		for _, lag := range []int{0, 1, 3} {
			res, err := ADFTest(y, lag, trend)
			if err != nil {
				t.Fatalf("trend=%d lag=%d: %v", trend, lag, err)
			}
			want := independentADFStat(t, y, lag, trend)
			if math.Abs(res.Statistic-want) > 1e-6*math.Max(1, math.Abs(want)) {
				t.Fatalf("trend=%d lag=%d: statistic=%.12g, independent oracle=%.12g", trend, lag, res.Statistic, want)
			}
		}
	}
}

// TestADFDecisionsOnKnownProcesses locks the qualitative behaviour: a stationary
// AR(1) rejects the unit root, an independent random walk does not.
func TestADFDecisionsOnKnownProcesses(t *testing.T) {
	ar := stationaryAR1(7, 300, 0.2)
	res, err := ADFTest(ar, 1, ADFConstant)
	if err != nil {
		t.Fatalf("AR(1): %v", err)
	}
	if rej, _ := res.Reject(0.05); !rej {
		t.Fatalf("stationary AR(1): want reject unit root, statistic=%.3f", res.Statistic)
	}

	rw := randomWalk(7, 300)
	res, err = ADFTest(rw, 1, ADFConstant)
	if err != nil {
		t.Fatalf("random walk: %v", err)
	}
	if rej, _ := res.Reject(0.05); rej {
		t.Fatalf("random walk: want DO NOT reject unit root, statistic=%.3f", res.Statistic)
	}
}

func TestADFErrors(t *testing.T) {
	good := stationaryAR1(1, 50, 0.3)
	if _, err := ADFTest(good, -1, ADFConstant); !errors.Is(err, ErrADFInvalidLag) {
		t.Fatalf("negative lag: %v", err)
	}
	bad := append([]float64(nil), good...)
	bad[10] = math.NaN()
	if _, err := ADFTest(bad, 1, ADFConstant); !errors.Is(err, ErrNonFinite) {
		t.Fatalf("NaN: %v", err)
	}
	if _, err := ADFTest(good[:4], 2, ADFConstant); !errors.Is(err, ErrADFTooShort) {
		t.Fatalf("too short: %v", err)
	}
	constant := make([]float64, 50) // all zeros → rank-deficient design
	if _, err := ADFTest(constant, 1, ADFConstant); !errors.Is(err, ErrSingular) {
		t.Fatalf("constant series: %v", err)
	}
	res, err := ADFTest(good, 1, ADFConstant)
	if err != nil {
		t.Fatalf("good: %v", err)
	}
	if _, err := res.Reject(0.2); !errors.Is(err, ErrADFAlpha) {
		t.Fatalf("bad alpha: %v", err)
	}
}
