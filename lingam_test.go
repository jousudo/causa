package causa

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

// --- Non-Gaussian ground-truth SEM generator ----------------------------
//
// genLiNGAMData draws n i.i.d. samples from a linear acyclic SEM whose weighted
// adjacency is w (w[parent][child] is the structural coefficient, 0 = no edge)
// and topological order topo. Each variable is the weighted sum of its parents
// plus an INDEPENDENT non-Gaussian disturbance drawn by noise. Non-Gaussian noise
// is what makes the causal direction identifiable — the whole point of LiNGAM —
// so unlike genSEM (Gaussian, for the PC tests) this takes a noise generator. The
// draw is seeded so the suite is deterministic.
func genLiNGAMData(seed int64, n, p int, w [][]float64, topo []int, noise func(*rand.Rand) float64) [][]float64 {
	rng := rand.New(rand.NewSource(seed))
	data := make([][]float64, p)
	for i := range data {
		data[i] = make([]float64, n)
	}
	for t := 0; t < n; t++ {
		for _, node := range topo {
			v := noise(rng)
			for parent := 0; parent < p; parent++ {
				if w[parent][node] != 0 {
					v += w[parent][node] * data[parent][t]
				}
			}
			data[node][t] = v
		}
	}
	return data
}

// uniformNoise draws from U(−1, 1): zero mean, platykurtic (negative excess
// kurtosis) — non-Gaussian, hence identifiable by LiNGAM.
func uniformNoise(rng *rand.Rand) float64 { return rng.Float64()*2 - 1 }

// laplaceNoise draws from a standard Laplace (mean 0, scale 1) by inverse-CDF:
// leptokurtic (heavy tails), strongly non-Gaussian.
func laplaceNoise(rng *rand.Rand) float64 {
	u := rng.Float64() - 0.5
	if u < 0 {
		return math.Log(1 + 2*u)
	}
	return -math.Log(1 - 2*u)
}

// positionInOrder returns the position of variable v in a causal order, or −1.
func positionInOrder(order []int, v int) int {
	for pos, x := range order {
		if x == v {
			return pos
		}
	}
	return -1
}

// --- Direction recovery on known small DAGs -----------------------------

func TestLiNGAM2VarDirection(t *testing.T) {
	// x0 -> x1 with coefficient 1.5, uniform noise. The exogenous root is x0, so
	// the recovered order must be [0, 1] and the edge 0 -> 1 recovered.
	w := zeroW(2)
	w[0][1] = 1.5
	data := genLiNGAMData(1, 3000, 2, w, []int{0, 1}, uniformNoise)
	res, err := DirectLiNGAM(data, []string{"X0", "X1"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := res.CausalOrder(); got[0] != 0 || got[1] != 1 {
		t.Fatalf("causal order = %v, want [0 1]\n%s", got, res)
	}
	if c := res.Coefficient(0, 1); math.Abs(c-1.5) > 0.1 {
		t.Errorf("coefficient 0->1 = %.4f, want ~1.5", c)
	}
	if c := res.Coefficient(1, 0); c != 0 {
		t.Errorf("coefficient 1->0 = %.4f, want 0 (1 is downstream)", c)
	}
}

func TestLiNGAM2VarReverseDirection(t *testing.T) {
	// The mirror image: x1 -> x0. The recovered order must flip to [1, 0], proving
	// the method recovers direction, not just an arbitrary ordering.
	w := zeroW(2)
	w[1][0] = 1.5
	data := genLiNGAMData(2, 3000, 2, w, []int{1, 0}, uniformNoise)
	res, err := DirectLiNGAM(data, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := res.CausalOrder(); got[0] != 1 || got[1] != 0 {
		t.Fatalf("causal order = %v, want [1 0]\n%s", got, res)
	}
	if c := res.Coefficient(1, 0); math.Abs(c-1.5) > 0.1 {
		t.Errorf("coefficient 1->0 = %.4f, want ~1.5", c)
	}
}

func TestLiNGAM3VarChain(t *testing.T) {
	// Chain x0 -> x1 -> x2. Unlike the PC algorithm (which returns this undirected,
	// as a Markov equivalence class), LiNGAM orients the whole chain from the
	// non-Gaussian noise: order [0, 1, 2] and edges 0->1, 1->2.
	w := zeroW(3)
	w[0][1] = 0.9
	w[1][2] = 0.9
	data := genLiNGAMData(3, 3000, 3, w, []int{0, 1, 2}, laplaceNoise)
	res, err := DirectLiNGAM(data, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	order := res.CausalOrder()
	if positionInOrder(order, 0) > positionInOrder(order, 1) ||
		positionInOrder(order, 1) > positionInOrder(order, 2) {
		t.Fatalf("causal order %v inconsistent with chain 0->1->2\n%s", order, res)
	}
	if c := res.Coefficient(0, 1); math.Abs(c-0.9) > 0.1 {
		t.Errorf("coefficient 0->1 = %.4f, want ~0.9", c)
	}
	if c := res.Coefficient(1, 2); math.Abs(c-0.9) > 0.1 {
		t.Errorf("coefficient 1->2 = %.4f, want ~0.9", c)
	}
}

func TestLiNGAM3VarCollider(t *testing.T) {
	// Collider x0 -> x2 <- x1: two independent roots into a common effect. Both x0
	// and x1 must precede x2 in the recovered order, and both edges recovered.
	w := zeroW(3)
	w[0][2] = 0.9
	w[1][2] = 0.9
	data := genLiNGAMData(4, 3000, 3, w, []int{0, 1, 2}, laplaceNoise)
	res, err := DirectLiNGAM(data, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	order := res.CausalOrder()
	if positionInOrder(order, 2) != 2 {
		t.Fatalf("x2 (the collider) should be last in the order, got %v\n%s", order, res)
	}
	if c := res.Coefficient(0, 2); math.Abs(c-0.9) > 0.1 {
		t.Errorf("coefficient 0->2 = %.4f, want ~0.9", c)
	}
	if c := res.Coefficient(1, 2); math.Abs(c-0.9) > 0.1 {
		t.Errorf("coefficient 1->2 = %.4f, want ~0.9", c)
	}
}

// TestLiNGAMRandomDAG builds a random 6-node DAG (edges only from lower to higher
// index, so ascending index is a valid topological order) with Laplace noise, and
// checks (a) the recovered order is consistent with the true topology — no true
// edge points backwards — and (b) the coefficients on true edges are recovered
// within tolerance.
func TestLiNGAMRandomDAG(t *testing.T) {
	const p = 6
	rng := rand.New(rand.NewSource(20260717))
	w := zeroW(p)
	type edge struct{ parent, child int }
	var trueEdges []edge
	for i := 0; i < p; i++ {
		for j := i + 1; j < p; j++ {
			if rng.Float64() < 0.4 {
				c := 0.8
				if rng.Float64() < 0.5 {
					c = -0.8
				}
				w[i][j] = c
				trueEdges = append(trueEdges, edge{i, j})
			}
		}
	}
	topo := []int{0, 1, 2, 3, 4, 5}
	data := genLiNGAMData(777, 4000, p, w, topo, laplaceNoise)

	res, err := DirectLiNGAM(data, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	order := res.CausalOrder()

	// (a) No true edge parent->child may be reversed in the recovered order.
	for _, e := range trueEdges {
		if positionInOrder(order, e.parent) > positionInOrder(order, e.child) {
			t.Errorf("true edge %d->%d reversed in recovered order %v", e.parent, e.child, order)
		}
	}
	// (b) Coefficients on true edges recovered within tolerance.
	for _, e := range trueEdges {
		got := res.Coefficient(e.parent, e.child)
		if math.Abs(got-w[e.parent][e.child]) > 0.12 {
			t.Errorf("coefficient %d->%d = %.4f, want %.4f", e.parent, e.child, got, w[e.parent][e.child])
		}
	}
}

// --- Gaussian honest-failure --------------------------------------------

// TestLiNGAMGaussianUnidentifiable locks the documented failure mode: on Gaussian
// noise the linear model is unidentifiable (a Gaussian SEM and its reverse fit
// identically), so the recovered order is arbitrary and carries no meaning.
// DirectLiNGAM cannot detect this and returns a fully oriented model regardless.
// We therefore assert ONLY that it runs and returns a complete, well-formed order
// — NOT that the order is correct — on two Gaussian datasets from the same
// structure, and log whatever orders come back: whether they agree or differ,
// neither carries the causal meaning it would under non-Gaussian noise.
func TestLiNGAMGaussianUnidentifiable(t *testing.T) {
	w := zeroW(2)
	w[0][1] = 0.9 // true structure 0 -> 1, but Gaussian noise hides the direction

	assertWellFormed := func(res *LiNGAMResult) {
		t.Helper()
		order := res.CausalOrder()
		if len(order) != 2 {
			t.Fatalf("expected a complete order of length 2, got %v", order)
		}
		seen := map[int]bool{}
		for _, v := range order {
			if v < 0 || v >= 2 || seen[v] {
				t.Fatalf("malformed order %v", order)
			}
			seen[v] = true
		}
	}

	// genSEM uses Gaussian (NormFloat64) noise; reused from the PC tests.
	a := genSEM(101, 3000, 2, w, []int{0, 1})
	resA, err := DirectLiNGAM(a, nil, nil)
	if err != nil {
		t.Fatalf("Gaussian input must still run without error, got %v", err)
	}
	assertWellFormed(resA)

	b := genSEM(202, 3000, 2, w, []int{0, 1})
	resB, err := DirectLiNGAM(b, nil, nil)
	if err != nil {
		t.Fatalf("Gaussian input must still run without error, got %v", err)
	}
	assertWellFormed(resB)

	// No assertion on which order is "right": under Gaussian noise there is no
	// right answer to recover. We only log what came back.
	t.Logf("Gaussian noise (unidentifiable): dataset A order=%v, dataset B order=%v",
		resA.CausalOrder(), resB.CausalOrder())
}

// --- Determinism --------------------------------------------------------

func TestLiNGAMDeterministic(t *testing.T) {
	w := zeroW(4)
	w[0][1] = 0.8
	w[0][2] = -0.6
	w[1][3] = 0.7
	w[2][3] = 0.5
	data := genLiNGAMData(11, 2000, 4, w, []int{0, 1, 2, 3}, laplaceNoise)
	r1, err := DirectLiNGAM(data, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := DirectLiNGAM(data, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if r1.String() != r2.String() {
		t.Errorf("non-deterministic output:\n%s\nvs\n%s", r1, r2)
	}
}

// --- Numeric oracle -----------------------------------------------------

// TestLiNGAMOracle cross-checks the entropy-based dependence measure, the
// recovered causal order and the coefficient matrix B against an INDEPENDENT
// pure-Python (stdlib-only) reimplementation that shares no code with the Go
// package: it solves every OLS by the normal equations with Gaussian elimination
// (Go uses Householder QR) and evaluates the same closed-form entropy
// approximation. Agreement to ≥10 significant digits validates the numerics
// end-to-end, standing in for a reference-package oracle unavailable in this
// offline, dependency-free build.
//
// Provenance. The dataset literal and the locked constants below are produced by
// scripts/lingam_oracle.py, run from the repository root as:
//
//	python3 scripts/lingam_oracle.py
//
// on a fixed dataset (random.Random(20260717), n=14, SEM x0 exogenous,
// x1 = 1.5·x0 + e1, x2 = −0.8·x0 + 0.6·x1 + e2, uniform noise). The script and
// this test implement the same formulas by two independent numerical paths.
func TestLiNGAMOracle(t *testing.T) {
	data := [][]float64{
		{-0.2958605991625505, -0.6376752182470251, 0.8490236582937243, -0.5461364967933764, 0.1748823775821804, 0.463124072713615, -0.13670103466909822, 0.8081512840366207, -0.786413639387838, -0.19436437402827966, 0.3662133265212191, -0.42972274970282154, 0.05045975931686297, 0.5822524794180368},
		{-0.9878302423091678, -0.164194025177141, 0.42312123333447427, 0.014450557683166165, 0.32611949359133696, 1.4446377063510463, -0.7670384226460678, 2.152830675468084, -1.8340573409584415, -0.4939404654044821, 1.4989524164240433, 0.29615570733962837, -0.09972175903225311, 0.4947322320992924},
		{-0.6463539141526573, -0.16019803192879967, -0.6599521216669759, -0.48051544750213043, 0.4114839965898249, 1.2069976102879618, -0.9790332898400275, 1.5282111810718306, -1.4650159934521363, 0.8457389503468282, -0.30366921103974676, 0.9246572302325052, 0.4371665018409193, -0.5473950689682465},
	}

	// (1) Entropy-based dependence measure: the directional diff-MI on the
	// standardized original columns, locked to the oracle.
	s0, ok0 := standardize(data[0])
	s1, ok1 := standardize(data[1])
	s2, ok2 := standardize(data[2])
	if !ok0 || !ok1 || !ok2 {
		t.Fatal("standardize unexpectedly failed on the oracle data")
	}
	type mi struct {
		name string
		a, b []float64
		want float64
	}
	for _, c := range []mi{
		{"diffMI(0,1)", s0, s1, 0.21665029674578085},
		{"diffMI(0,2)", s0, s2, 0.05461464368515401},
		{"diffMI(1,2)", s1, s2, -0.04532970386179791},
	} {
		got, err := lingamDiffMI(c.a, c.b)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if rel(got, c.want) > 1e-10 {
			t.Errorf("%s = %.15g, oracle %.15g (rel %.2e)", c.name, got, c.want, rel(got, c.want))
		}
	}

	// (2) Full recovered order and coefficient matrix, locked to the oracle.
	res, err := DirectLiNGAM(data, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := res.CausalOrder(); got[0] != 0 || got[1] != 1 || got[2] != 2 {
		t.Fatalf("recovered order = %v, oracle [0 1 2]", got)
	}
	for _, c := range []struct {
		from, to int
		want     float64
	}{
		{0, 1, 1.4748006800239122},
		{0, 2, -0.556368510163533},
		{1, 2, 0.771530016306644},
	} {
		got := res.Coefficient(c.from, c.to)
		if rel(got, c.want) > 1e-10 {
			t.Errorf("B coefficient %d->%d = %.15g, oracle %.15g (rel %.2e)",
				c.from, c.to, got, c.want, rel(got, c.want))
		}
	}
}

// --- Input validation ---------------------------------------------------

func TestDirectLiNGAMErrors(t *testing.T) {
	good := genLiNGAMData(1, 200, 3, func() [][]float64 {
		w := zeroW(3)
		w[0][1] = 0.9
		w[1][2] = 0.9
		return w
	}(), []int{0, 1, 2}, laplaceNoise)

	ragged := [][]float64{{1, 2, 3, 4}, {1, 2, 3}}
	withNaNData := [][]float64{{1, 2, 3, 4}, {1, 2, math.NaN(), 4}}
	constVar := [][]float64{{1, 1, 1, 1, 1}, {0.3, -0.2, 0.5, 0.1, -0.4}}

	tests := []struct {
		name  string
		data  [][]float64
		names []string
		opts  *LiNGAMOptions
		want  error
	}{
		{"one variable", [][]float64{{1, 2, 3, 4}}, nil, nil, ErrTooFewVariables},
		{"ragged", ragged, nil, nil, ErrUnequalLengths},
		{"too few samples", [][]float64{{1, 2}, {3, 4}}, nil, nil, ErrTooFewSamples},
		{"nan", withNaNData, nil, nil, ErrNonFinite},
		{"name count", good, []string{"only-one"}, nil, ErrNameCount},
		{"negative threshold", good, nil, &LiNGAMOptions{PruneThreshold: -0.1}, ErrInvalidThreshold},
		{"constant variable", constVar, nil, nil, ErrSingular},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := DirectLiNGAM(tc.data, tc.names, tc.opts)
			if !errors.Is(err, tc.want) {
				t.Errorf("got err=%v, want %v", err, tc.want)
			}
		})
	}
}

// --- Pruning ------------------------------------------------------------

// TestLiNGAMPruneThreshold checks the absolute-magnitude pruning option zeroes
// small coefficients while leaving strong edges intact.
func TestLiNGAMPruneThreshold(t *testing.T) {
	// x0 -> x2 strong, x1 -> x2 weak. A threshold between the two drops the weak
	// edge only.
	w := zeroW(3)
	w[0][2] = 1.2
	w[1][2] = 0.05
	data := genLiNGAMData(5, 4000, 3, w, []int{0, 1, 2}, laplaceNoise)

	raw, err := DirectLiNGAM(data, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if raw.Coefficient(1, 2) == 0 {
		t.Fatal("precondition: unpruned weak edge 1->2 should be non-zero")
	}

	pruned, err := DirectLiNGAM(data, nil, &LiNGAMOptions{PruneThreshold: 0.3})
	if err != nil {
		t.Fatal(err)
	}
	if c := pruned.Coefficient(1, 2); c != 0 {
		t.Errorf("weak edge 1->2 = %.4f, want pruned to 0", c)
	}
	if c := pruned.Coefficient(0, 2); math.Abs(c-1.2) > 0.1 {
		t.Errorf("strong edge 0->2 = %.4f, want ~1.2 (not pruned)", c)
	}
}

// --- Result accessors ---------------------------------------------------

func TestLiNGAMResultAccessors(t *testing.T) {
	w := zeroW(2)
	w[0][1] = 1.5
	data := genLiNGAMData(1, 2000, 2, w, []int{0, 1}, uniformNoise)
	res, err := DirectLiNGAM(data, []string{"A", "B"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if nodes := res.Nodes(); len(nodes) != 2 || nodes[0] != "A" || nodes[1] != "B" {
		t.Errorf("Nodes() = %v, want [A B]", nodes)
	}
	if on := res.OrderedNodes(); on[0] != "A" || on[1] != "B" {
		t.Errorf("OrderedNodes() = %v, want [A B]", on)
	}
	edges := res.Edges()
	if len(edges) != 1 || edges[0].From != 0 || edges[0].To != 1 {
		t.Fatalf("Edges() = %+v, want a single 0->1 edge", edges)
	}
	if math.Abs(edges[0].Weight-1.5) > 0.1 {
		t.Errorf("edge weight = %.4f, want ~1.5", edges[0].Weight)
	}
	// Weights() must be a copy; mutating it must not affect the result.
	ws := res.Weights()
	ws[1][0] = 999
	if res.Coefficient(0, 1) == 999 {
		t.Error("Weights() leaked internal state (not a copy)")
	}
	if res.String() == "" {
		t.Error("String() returned empty")
	}
}
