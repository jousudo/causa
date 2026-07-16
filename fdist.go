package causa

import "math"

// betacf evaluates the continued fraction for the incomplete beta function
// using the modified Lentz algorithm. It is a helper for regIncBeta and is
// only accurate in the region where that continued fraction converges quickly
// (x < (a+1)/(a+b+2)); regIncBeta routes arguments to the correct branch.
//
// Reference: Press et al., "Numerical Recipes", 3rd ed., §6.4 (betacf); Lentz,
// Appl. Opt. 15 (1976) 668 for the continued-fraction evaluation scheme.
func betacf(a, b, x float64) float64 {
	const (
		maxIter = 300
		eps     = 3e-16
		fpMin   = 1e-300
	)
	qab := a + b
	qap := a + 1
	qam := a - 1
	c := 1.0
	d := 1 - qab*x/qap
	if math.Abs(d) < fpMin {
		d = fpMin
	}
	d = 1 / d
	h := d
	for m := 1; m <= maxIter; m++ {
		mf := float64(m)
		m2 := 2 * mf
		// Even step.
		aa := mf * (b - mf) * x / ((qam + m2) * (a + m2))
		d = 1 + aa*d
		if math.Abs(d) < fpMin {
			d = fpMin
		}
		c = 1 + aa/c
		if math.Abs(c) < fpMin {
			c = fpMin
		}
		d = 1 / d
		h *= d * c
		// Odd step.
		aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2))
		d = 1 + aa*d
		if math.Abs(d) < fpMin {
			d = fpMin
		}
		c = 1 + aa/c
		if math.Abs(c) < fpMin {
			c = fpMin
		}
		d = 1 / d
		del := d * c
		h *= del
		if math.Abs(del-1) < eps {
			break
		}
	}
	return h
}

// regIncBeta returns the regularized incomplete beta function I_x(a, b), i.e.
// the incomplete beta integral from 0 to x divided by the complete beta
// function B(a, b), for a, b > 0 and 0 ≤ x ≤ 1. It is the building block for
// the F-distribution tail below.
//
// The value is computed from the continued fraction (betacf) applied to
// whichever of x or 1−x lands in the fast-converging region, using the
// symmetry I_x(a, b) = 1 − I_{1−x}(b, a). The prefactor is evaluated through
// math.Lgamma to stay finite for large parameters.
//
// Reference: Press et al., "Numerical Recipes", 3rd ed., §6.4 (betai).
func regIncBeta(a, b, x float64) float64 {
	if x <= 0 {
		return 0
	}
	if x >= 1 {
		return 1
	}
	la, _ := math.Lgamma(a + b)
	lb, _ := math.Lgamma(a)
	lc, _ := math.Lgamma(b)
	logPrefactor := la - lb - lc + a*math.Log(x) + b*math.Log1p(-x)
	prefactor := math.Exp(logPrefactor)
	if x < (a+1)/(a+b+2) {
		return prefactor * betacf(a, b, x) / a
	}
	return 1 - prefactor*betacf(b, a, 1-x)/b
}

// fUpperTail returns the upper-tail probability P(F > f) of an
// F-distribution with df1 numerator and df2 denominator degrees of freedom
// (df1, df2 > 0). This is the survival function, so for an F-statistic it is
// directly the p-value.
//
// It uses the standard identity relating the F CDF to the regularized
// incomplete beta function. Writing the survival branch directly (rather than
// 1 − CDF) keeps precision in the small-p tail that matters for hypothesis
// testing:
//
//	P(F > f) = I_{df2/(df2+df1·f)}(df2/2, df1/2).
//
// Reference: Abramowitz & Stegun, "Handbook of Mathematical Functions",
// §26.6.2 (relation of the F distribution to the incomplete beta function).
func fUpperTail(f, df1, df2 float64) float64 {
	if f <= 0 {
		return 1
	}
	x := df2 / (df2 + df1*f)
	return regIncBeta(df2/2, df1/2, x)
}
