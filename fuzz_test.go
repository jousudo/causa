package causa

import (
	"errors"
	"testing"
)

func FuzzDenseConstructors(f *testing.F) {
	f.Add([]byte{2, 2})
	f.Add([]byte{0, 255, 3})
	f.Add([]byte{254, 254, 254, 254})

	f.Fuzz(func(t *testing.T, raw []byte) {
		if len(raw) > 8 {
			raw = raw[:8]
		}
		card := make([]int, len(raw))
		for i, b := range raw {
			switch b {
			case 0:
				card[i] = 0
			case 255:
				card[i] = -1
			case 254:
				card[i] = int(^uint(0) >> 1)
			default:
				card[i] = int(b%8) + 1
			}
		}

		_, err := NewDistribution(card, nil)
		if err != nil && !errors.Is(err, ErrBadDistribution) && !errors.Is(err, ErrStateSpaceTooLarge) {
			t.Fatalf("unexpected error: %v", err)
		}

		total, sizeErr := checkedDenseProduct(card, 1024)
		if sizeErr == nil && len(card) > 0 {
			data := make([][]int, len(card))
			for i := range data {
				data[i] = []int{0}
			}
			_, err := SampleDistributionWithOptions(data, card, &DenseOptions{MaxCells: 1024})
			if err != nil {
				t.Fatalf("small valid state space (%d cells): %v", total, err)
			}
		}
	})
}
