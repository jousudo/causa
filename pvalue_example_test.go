package causa_test

import (
	"fmt"

	"github.com/jousudo/causa"
)

func ExampleAdjustPValues() {
	adjusted, _ := causa.AdjustPValues(
		[]float64{0.01, 0.04, 0.03, 0.002, 0.8},
		causa.PValueHolm,
	)
	fmt.Println(adjusted)

	// Output:
	// [0.04 0.09 0.09 0.01 0.8]
}
