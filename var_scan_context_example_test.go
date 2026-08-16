package causa_test

import (
	"context"
	"fmt"

	"github.com/jousudo/causa"
)

func ExampleVARGrangerScanContext() {
	data, names := validityExampleData()
	scan, _ := causa.VARGrangerScanContext(context.Background(), data, names, 1, nil)
	fmt.Println("complete ordered family:", scan.Tests)
	fmt.Println("adjustment:", scan.Adjustment)

	// Output:
	// complete ordered family: 6
	// adjustment: holm
}
