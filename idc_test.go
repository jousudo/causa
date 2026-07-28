package causa

import "testing"

// --- m-separation unit tests ---------------------------------------------

func mSep(t *testing.T, g *Diagram, a, b, c []int) bool {
	t.Helper()
	n := g.Order()
	return mSeparated(g.dir, g.bi, n, maskFromSlice(n, a), maskFromSlice(n, b), maskFromSlice(n, c))
}

func TestMSeparationChain(t *testing.T) {
	// X→M→Y: X and Y are m-separated GIVEN M, dependent given nothing.
	g, _ := NewDiagram([]string{"X", "M", "Y"}, [][2]int{{0, 1}, {1, 2}}, nil)
	if !mSep(t, g, []int{0}, []int{2}, []int{1}) {
		t.Error("X ⊥ Y | M should hold on a chain")
	}
	if mSep(t, g, []int{0}, []int{2}, nil) {
		t.Error("X ⊥ Y (unconditional) must NOT hold on a chain")
	}
}

func TestMSeparationCollider(t *testing.T) {
	// X→C←Y: X,Y independent marginally, DEPENDENT given the collider C.
	g, _ := NewDiagram([]string{"X", "Y", "C"}, [][2]int{{0, 2}, {1, 2}}, nil)
	if !mSep(t, g, []int{0}, []int{1}, nil) {
		t.Error("X ⊥ Y should hold marginally at a collider")
	}
	if mSep(t, g, []int{0}, []int{1}, []int{2}) {
		t.Error("conditioning on the collider must OPEN the path")
	}
}

func TestMSeparationBidirected(t *testing.T) {
	// X↔Y: a latent common cause makes them dependent, unconditionally.
	g, _ := NewDiagram([]string{"X", "Y"}, nil, [][2]int{{0, 1}})
	if mSep(t, g, []int{0}, []int{1}, nil) {
		t.Error("X ↔ Y must not be m-separated")
	}
}

func TestMSeparationBidirectedCollider(t *testing.T) {
	// A↔M↔B: M is a (bidirected) collider — A,B separated marginally, connected
	// once M is conditioned on.
	g, _ := NewDiagram([]string{"A", "M", "B"}, nil, [][2]int{{0, 1}, {1, 2}})
	if !mSep(t, g, []int{0}, []int{2}, nil) {
		t.Error("A ⊥ B should hold at a bidirected collider")
	}
	if mSep(t, g, []int{0}, []int{2}, []int{1}) {
		t.Error("conditioning on the bidirected collider must open the path")
	}
}

// --- IDC numeric validation ----------------------------------------------

// validateConditional asserts identifiability and that the conditional estimand,
// evaluated on the observational joint, matches the true P(y | do(x), z) computed
// by brute force on random latent SCMs.
func validateConditional(t *testing.T, g *Diagram, y, x, z []int) {
	t.Helper()
	res, err := IdentifyConditional(g, y, x, z)
	if err != nil {
		t.Fatal(err)
	}
	if !res.Identifiable {
		t.Fatalf("expected identifiable, got %s", res)
	}
	n := g.Order()
	obsCard := make([]int, n)
	for i := range obsCard {
		obsCard[i] = 2
	}
	yset := map[int]bool{}
	for _, v := range y {
		yset[v] = true
	}
	yz := append(append([]int(nil), y...), z...)
	for _, seed := range []int64{1, 2, 3, 20260728} {
		nodes, gcardFull := buildSCM(g, obsCard, 2, seed)
		joint := observationalJoint(nodes, gcardFull, obsCard, n)
		got, err := res.Estimand.Evaluate(joint)
		if err != nil {
			t.Fatal(err)
		}
		// truth = P_x(y,z) / P_x(z).
		pxyz := trueInterventional(nodes, gcardFull, n, yz, x)
		pxz := factorMarginalize(pxyz, yset, gcardFull)
		truth := factorDivide(pxyz, pxz, gcardFull)
		if !sortedEqual(got.Vars(), truth.vars) {
			t.Fatalf("seed %d: vars = %v, want %v\nestimand: %s", seed, got.Vars(), truth.vars, res)
		}
		for i := range truth.prob {
			if diff := got.prob[i] - truth.prob[i]; diff > 1e-9 || diff < -1e-9 {
				t.Fatalf("seed %d: estimand ≠ truth at %d: got %g want %g\nestimand: %s",
					seed, i, got.prob[i], truth.prob[i], res)
			}
		}
	}
}

func TestIDCContextBackdoor(t *testing.T) {
	// Z→X→Y with Z→Y: effect of X on Y WITHIN context Z=z. Rule 2 moves Z into the
	// intervention (it acts like one here). X=0, Y=1, Z=2.
	g, err := NewDiagram([]string{"X", "Y", "Z"},
		[][2]int{{2, 0}, {0, 1}, {2, 1}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	validateConditional(t, g, []int{1}, []int{0}, []int{2})
}

func TestIDCDescendantConditioning(t *testing.T) {
	// X→Y→Z, conditioning on the downstream Z. Rule 2 does not fire (Y and Z stay
	// dependent), so the estimand is the ratio P_x(y,z)/P_x(z). X=0, Y=1, Z=2.
	g, err := NewDiagram([]string{"X", "Y", "Z"},
		[][2]int{{0, 1}, {1, 2}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	validateConditional(t, g, []int{1}, []int{0}, []int{2})
}

func TestIDCLatentContext(t *testing.T) {
	// Front-door with a context: X→M→Y, X↔Y, and an extra observed context W→Y.
	// Effect of X on Y given W. X=0, M=1, Y=2, W=3.
	g, err := NewDiagram([]string{"X", "M", "Y", "W"},
		[][2]int{{0, 1}, {1, 2}, {3, 2}}, [][2]int{{0, 2}})
	if err != nil {
		t.Fatal(err)
	}
	validateConditional(t, g, []int{2}, []int{0}, []int{3})
}

func TestIDCReducesToIdentify(t *testing.T) {
	// With empty z, IDC is exactly ID: the front-door effect, validated.
	g, err := NewDiagram([]string{"X", "M", "Y"},
		[][2]int{{0, 1}, {1, 2}}, [][2]int{{0, 2}})
	if err != nil {
		t.Fatal(err)
	}
	validateConditional(t, g, []int{2}, []int{0}, nil)
}

func TestIDCNotIdentifiable(t *testing.T) {
	// Bow arc, asked as a conditional (empty z): still not identifiable.
	g, err := NewDiagram([]string{"X", "Y"}, [][2]int{{0, 1}}, [][2]int{{0, 1}})
	if err != nil {
		t.Fatal(err)
	}
	res, err := IdentifyConditional(g, []int{1}, []int{0}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Identifiable {
		t.Errorf("conditional bow-arc effect must be non-identifiable, got %s", res)
	}
}

func TestIdentifyConditionalErrors(t *testing.T) {
	g, _ := NewDiagram([]string{"X", "Y", "Z"}, [][2]int{{0, 1}}, nil)
	tests := []struct {
		name    string
		y, x, z []int
		want    error
	}{
		{"empty y", nil, []int{0}, []int{2}, ErrEmptyOutcome},
		{"y/z overlap", []int{1}, nil, []int{1}, ErrOverlappingQuery},
		{"x/z overlap", []int{1}, []int{0}, []int{0}, ErrOverlappingQuery},
		{"dup in z", []int{1}, nil, []int{2, 2}, ErrDuplicateQueryVar},
		{"out of range", []int{1}, []int{0}, []int{9}, ErrEdgeOutOfRange},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := IdentifyConditional(g, tc.y, tc.x, tc.z); !isErr(err, tc.want) {
				t.Errorf("got %v, want %v", err, tc.want)
			}
		})
	}
}
