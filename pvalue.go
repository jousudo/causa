package causa

import (
	"errors"
	"sort"
)

var (
	// ErrInvalidPValue is returned when a p-value is NaN, infinite, or outside
	// the closed interval [0,1].
	ErrInvalidPValue = errors.New("causa: p-values must be finite and in [0,1]")
	// ErrInvalidPValueAdjustment is returned for an unknown adjustment method.
	ErrInvalidPValueAdjustment = errors.New("causa: unknown p-value adjustment")
)

// PValueAdjustment selects how a family of simultaneous hypotheses is
// corrected. The zero value is Holm, the conservative default that controls
// family-wise error without an independence assumption.
type PValueAdjustment uint8

const (
	PValueHolm PValueAdjustment = iota
	PValueBenjaminiHochberg
	PValueNone
)

func (method PValueAdjustment) String() string {
	switch method {
	case PValueHolm:
		return "holm"
	case PValueBenjaminiHochberg:
		return "benjamini-hochberg"
	case PValueNone:
		return "none"
	default:
		return "unknown"
	}
}

// AdjustPValues adjusts one hypothesis family and preserves the original
// ordering. Holm controls the family-wise error rate under arbitrary dependence.
// Benjamini-Hochberg controls false discovery rate under independence or its
// standard positive-dependence conditions; it is not a universal guarantee for
// arbitrary dependent tests. PValueNone returns a validated copy unchanged.
func AdjustPValues(pValues []float64, method PValueAdjustment) ([]float64, error) {
	if !validPValueAdjustment(method) {
		return nil, ErrInvalidPValueAdjustment
	}
	adjusted := make([]float64, len(pValues))
	copy(adjusted, pValues)
	for _, pValue := range adjusted {
		if !isFinite(pValue) || pValue < 0 || pValue > 1 {
			return nil, ErrInvalidPValue
		}
	}
	if method == PValueNone || len(adjusted) < 2 {
		return adjusted, nil
	}
	type rankedPValue struct {
		value float64
		index int
	}
	ranked := make([]rankedPValue, len(adjusted))
	for index, value := range adjusted {
		ranked[index] = rankedPValue{value: value, index: index}
	}
	sort.SliceStable(ranked, func(i, j int) bool {
		if ranked[i].value != ranked[j].value {
			return ranked[i].value < ranked[j].value
		}
		return ranked[i].index < ranked[j].index
	})
	m := len(ranked)
	sortedAdjusted := make([]float64, m)
	switch method {
	case PValueHolm:
		runningMaximum := 0.0
		for rank, item := range ranked {
			value := float64(m-rank) * item.value
			if value > 1 {
				value = 1
			}
			if value < runningMaximum {
				value = runningMaximum
			}
			runningMaximum = value
			sortedAdjusted[rank] = value
		}
	case PValueBenjaminiHochberg:
		runningMinimum := 1.0
		for rank := m - 1; rank >= 0; rank-- {
			value := float64(m) * ranked[rank].value / float64(rank+1)
			if value > 1 {
				value = 1
			}
			if value > runningMinimum {
				value = runningMinimum
			}
			runningMinimum = value
			sortedAdjusted[rank] = value
		}
	}
	for rank, item := range ranked {
		adjusted[item.index] = sortedAdjusted[rank]
	}
	return adjusted, nil
}

func validPValueAdjustment(method PValueAdjustment) bool {
	return method == PValueHolm || method == PValueBenjaminiHochberg || method == PValueNone
}
