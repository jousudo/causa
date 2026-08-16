package causa

import "errors"

const (
	// DefaultMaxDenseCells is the default maximum number of float64 cells in a
	// dense discrete distribution or intermediate factor. It limits one table to
	// 128 MiB. Callers with a reviewed memory budget may override it through
	// DenseOptions.
	DefaultMaxDenseCells = 1 << 24
)

// ErrStateSpaceTooLarge is returned before allocation when the product of the
// declared cardinalities overflows int or exceeds the configured dense-table
// cell budget.
var ErrStateSpaceTooLarge = errors.New("causa: discrete state space exceeds dense-table budget")

// DenseOptions configures dense discrete distributions and estimand evaluation.
// The zero value is safe and uses DefaultMaxDenseCells.
type DenseOptions struct {
	// MaxCells is the maximum number of float64 cells in any dense table. Zero
	// selects DefaultMaxDenseCells. A negative value removes the configured cap,
	// but integer overflow is always rejected. Positive overrides should be
	// chosen from an explicit process memory budget.
	MaxCells int
}

func denseCellLimit(opts *DenseOptions) int {
	if opts == nil || opts.MaxCells == 0 {
		return DefaultMaxDenseCells
	}
	return opts.MaxCells
}

func checkedDenseProduct(card []int, maxCells int) (int, error) {
	maxInt := int(^uint(0) >> 1)
	total := 1
	for _, c := range card {
		if c < 1 {
			return 0, ErrBadDistribution
		}
		if total > maxInt/c {
			return 0, ErrStateSpaceTooLarge
		}
		total *= c
		if maxCells > 0 && total > maxCells {
			return 0, ErrStateSpaceTooLarge
		}
	}
	return total, nil
}
