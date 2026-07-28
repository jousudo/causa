package causa_test

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/jousudo/causa"
)

// ExampleGrangerTest demonstrates a one-directional Granger causality test on
// synthetic data where a "cause" series genuinely drives an "effect" series one
// step ahead. The random draw is seeded so the output is deterministic.
func ExampleGrangerTest() {
	rng := rand.New(rand.NewSource(42))
	const n = 400
	cause := make([]float64, n)
	effect := make([]float64, n)
	for t := 1; t < n; t++ {
		// cause is its own AR(1) process; effect depends on cause's last value.
		cause[t] = 0.4*cause[t-1] + rng.NormFloat64()
		effect[t] = 0.5*effect[t-1] + 0.7*cause[t-1] + 0.5*rng.NormFloat64()
	}

	// Does the past of cause help predict effect? (cause -> effect)
	fwd, err := causa.GrangerTest(cause, effect, 2)
	if err != nil {
		panic(err)
	}
	// And the reverse: does effect help predict cause? It should not.
	rev, err := causa.GrangerTest(effect, cause, 2)
	if err != nil {
		panic(err)
	}

	fmt.Printf("cause -> effect: significant=%v\n", fwd.PValue < 0.01)
	fmt.Printf("effect -> cause: significant=%v\n", rev.PValue < 0.01)

	// Output:
	// cause -> effect: significant=true
	// effect -> cause: significant=false
}

// ExamplePCStable recovers the structure of a collider A -> C <- B from
// observational data. A and B are independent causes of C; because they are
// marginally independent, the PC algorithm orients the v-structure, printing
// both compelled arrowheads into C. The random draw is seeded for a deterministic
// result.
func ExamplePCStable() {
	rng := rand.New(rand.NewSource(3))
	const n = 5000
	a := make([]float64, n)
	b := make([]float64, n)
	c := make([]float64, n)
	for t := 0; t < n; t++ {
		a[t] = rng.NormFloat64()
		b[t] = rng.NormFloat64()
		c[t] = 0.8*a[t] + 0.8*b[t] + rng.NormFloat64()
	}

	g, err := causa.PCStable([][]float64{a, b, c}, []string{"A", "B", "C"}, nil)
	if err != nil {
		panic(err)
	}

	for _, e := range g.DirectedEdges() {
		nodes := g.Nodes()
		fmt.Printf("%s -> %s\n", nodes[e.From], nodes[e.To])
	}
	fmt.Printf("undirected edges: %d\n", len(g.UndirectedEdges()))

	// Output:
	// A -> C
	// B -> C
	// undirected edges: 0
}

// ExampleDirectLiNGAM recovers the direction of a two-variable cause→effect model
// from observational data. Because the noise is non-Gaussian (uniform), LiNGAM
// identifies the direction that a constraint-based method could not: it prints the
// causal order (cause first) and the estimated connection strength. The random
// draw is seeded for a deterministic result.
func ExampleDirectLiNGAM() {
	rng := rand.New(rand.NewSource(1))
	const n = 3000
	cause := make([]float64, n)
	effect := make([]float64, n)
	for t := 0; t < n; t++ {
		cause[t] = rng.Float64()*2 - 1                   // uniform, non-Gaussian
		effect[t] = 1.5*cause[t] + (rng.Float64()*2 - 1) // effect = 1.5·cause + noise
	}

	res, err := causa.DirectLiNGAM([][]float64{cause, effect}, []string{"cause", "effect"}, nil)
	if err != nil {
		panic(err)
	}

	fmt.Printf("causal order: %v\n", res.OrderedNodes())
	fmt.Printf("cause -> effect: %.1f\n", res.Coefficient(0, 1))

	// Output:
	// causal order: [cause effect]
	// cause -> effect: 1.5
}

// ExamplePartialCorrelation shows how conditioning removes a spurious
// correlation. A latent common cause Z drives both X and Y, so X and Y are
// correlated on their own; but given Z the partial correlation collapses to ~0 —
// the numeric heart of constraint-based discovery. The draw is seeded so the
// output is deterministic.
func ExamplePartialCorrelation() {
	rng := rand.New(rand.NewSource(7))
	const n = 2000
	x := make([]float64, n)
	y := make([]float64, n)
	z := make([]float64, n)
	for t := 0; t < n; t++ {
		z[t] = rng.NormFloat64()
		x[t] = z[t] + rng.NormFloat64() // X = Z + noise
		y[t] = z[t] + rng.NormFloat64() // Y = Z + noise
	}
	data := [][]float64{x, y, z} // indices: X=0, Y=1, Z=2

	// Marginal correlation of X and Y (empty conditioning set).
	rMarg, err := causa.PartialCorrelation(data, 0, 1, nil)
	if err != nil {
		panic(err)
	}
	// Partial correlation of X and Y given Z.
	rCond, err := causa.PartialCorrelation(data, 0, 1, []int{2})
	if err != nil {
		panic(err)
	}

	fmt.Printf("X,Y correlated marginally: %v\n", math.Abs(rMarg) > 0.3)
	fmt.Printf("X,Y uncorrelated given Z:  %v\n", math.Abs(rCond) < 0.1)

	// Output:
	// X,Y correlated marginally: true
	// X,Y uncorrelated given Z:  true
}

// ExampleFisherZTest shows the default conditional-independence test PCStable
// uses. On the same latent-common-cause data (Z drives X and Y), the test reports
// X and Y as dependent on their own, but conditionally INDEPENDENT given Z — the
// edge-deletion decision the PC algorithm is built on. The draw is seeded so the
// output is deterministic.
func ExampleFisherZTest() {
	rng := rand.New(rand.NewSource(7))
	const n = 2000
	x := make([]float64, n)
	y := make([]float64, n)
	z := make([]float64, n)
	for t := 0; t < n; t++ {
		z[t] = rng.NormFloat64()
		x[t] = z[t] + rng.NormFloat64()
		y[t] = z[t] + rng.NormFloat64()
	}
	data := [][]float64{x, y, z} // indices: X=0, Y=1, Z=2

	pMarg, err := causa.FisherZTest(data, 0, 1, nil)
	if err != nil {
		panic(err)
	}
	pCond, err := causa.FisherZTest(data, 0, 1, []int{2})
	if err != nil {
		panic(err)
	}

	// Marginally X and Y are strongly dependent; conditioning on the common cause Z
	// collapses that dependence by many orders of magnitude (the edge-deletion the
	// PC algorithm makes). A fixed p>0.05 threshold would be brittle at this sample
	// size — a correctly-sized α=0.05 test still rejects ~5% of the time under true
	// independence — so the robust, deterministic claim is the contrast itself.
	fmt.Printf("marginal dependence significant (p < 1e-6):    %v\n", pMarg < 1e-6)
	fmt.Printf("given Z the signal collapses (pCond >> pMarg): %v\n", pCond > 1e6*pMarg)

	// Output:
	// marginal dependence significant (p < 1e-6):    true
	// given Z the signal collapses (pCond >> pMarg): true
}

// ExampleSEM shows the do-operator and a counterfactual on a fully specified
// linear structural model X -> Y -> Z (Y = 2X, Z = 3Y). Interventions and
// counterfactuals reduce to forward substitution through the structural
// equations, so the answers are exact and deterministic.
func ExampleSEM() {
	coef := [][]float64{
		{0, 0, 0}, // X is a root
		{2, 0, 0}, // Y = 2·X
		{0, 3, 0}, // Z = 3·Y
	}
	s, err := causa.NewSEM([]string{"X", "Y", "Z"}, coef, nil)
	if err != nil {
		panic(err)
	}

	// Intervention: force X = 1 and read the interventional expectation.
	got, _ := s.Intervene(map[string]float64{"X": 1})
	fmt.Printf("do(X=1): Y=%.0f Z=%.0f\n", got["Y"], got["Z"])

	// Total effect of X on Z along the chain (2 × 3).
	te, _ := s.TotalEffect("X", "Z")
	fmt.Printf("total effect X->Z: %.0f\n", te)

	// Counterfactual: we OBSERVED (X=1, Y=3, Z=8) — Y and Z carry disturbances.
	// Had X been 5 instead, holding the same disturbances, what would follow?
	cf, _ := s.Counterfactual(
		map[string]float64{"X": 1, "Y": 3, "Z": 8},
		map[string]float64{"X": 5},
	)
	fmt.Printf("counterfactual do(X=5): Y=%.0f Z=%.0f\n", cf["Y"], cf["Z"])

	// Output:
	// do(X=1): Y=2 Z=6
	// total effect X->Z: 6
	// counterfactual do(X=5): Y=11 Z=32
}

// ExampleFCI recovers a latent common cause. A hidden variable L drives both B
// and C, while A drives B and D drives C. FCI sees only {A, B, C, D} — never L —
// yet returns B ↔ C, the bidirected edge that says "neither B nor C causes the
// other; a common cause is hidden". A causal-sufficiency method (PCStable) cannot
// express this. The two outer edges come back A o→ B and C ←o D: the arrowheads
// into B and C are compelled, their far ends left undetermined. The draw is
// seeded for a deterministic result.
func ExampleFCI() {
	rng := rand.New(rand.NewSource(20260727))
	const n = 12000
	// True DAG over {A,B,C,D,L}: A→B, L→B, L→C, D→C. L is the latent confounder.
	a := make([]float64, n)
	b := make([]float64, n)
	c := make([]float64, n)
	d := make([]float64, n)
	for t := 0; t < n; t++ {
		av := rng.NormFloat64()
		dv := rng.NormFloat64()
		lv := rng.NormFloat64() // latent — never handed to FCI
		a[t] = av
		d[t] = dv
		b[t] = 0.9*av + 0.9*lv + rng.NormFloat64()
		c[t] = 0.9*lv + 0.9*dv + rng.NormFloat64()
	}

	// Only the observed variables are passed; L is withheld.
	g, err := causa.FCI([][]float64{a, b, c, d}, []string{"A", "B", "C", "D"}, nil)
	if err != nil {
		panic(err)
	}
	fmt.Println(g)

	// Output:
	// PAG with 4 nodes [A B C D]:
	//   A o-> B
	//   B <-> C
	//   C <-o D
}

// ExampleIdentify decides whether an interventional effect P(y | do(x)) can be
// computed from observational data given a causal diagram with latent
// confounders, and returns the estimand when it can. Two contrasting cases: a
// back-door graph (identifiable — the classic adjustment formula), and the bow
// arc, where a latent common cause makes the same effect NOT identifiable. The
// estimand is then evaluated on a discrete observational joint to get the actual
// interventional number.
func ExampleIdentify() {
	// Back-door: X → Y confounded by an OBSERVED Z (Z → X, Z → Y). X=0, Y=1, Z=2.
	bd, _ := causa.NewDiagram([]string{"X", "Y", "Z"},
		[][2]int{{0, 1}, {2, 0}, {2, 1}}, nil)
	r1, _ := causa.Identify(bd, []int{1}, []int{0})
	fmt.Println("back-door identifiable:", r1.Identifiable)
	fmt.Println("  estimand:", r1)

	// Evaluate it on a discrete joint P(X,Y,Z) (all binary) to get a number.
	joint, _ := causa.NewDistribution([]int{2, 2, 2},
		[]float64{0.05, 0.10, 0.08, 0.12, 0.15, 0.07, 0.18, 0.25})
	tab, _ := r1.Estimand.Evaluate(joint)
	p, _ := tab.ProbAt(map[int]int{0: 1, 1: 1}) // P(Y=1 | do(X=1))
	fmt.Printf("  P(Y=1 | do(X=1)) = %.4f\n", p)

	// Bow arc: X → Y with a LATENT common cause X ↔ Y. Same query, not identifiable.
	bow, _ := causa.NewDiagram([]string{"X", "Y"}, [][2]int{{0, 1}}, [][2]int{{0, 1}})
	r2, _ := causa.Identify(bow, []int{1}, []int{0})
	fmt.Println("bow-arc identifiable:", r2.Identifiable)
	fmt.Println(" ", r2)

	// Output:
	// back-door identifiable: true
	//   estimand: Σ_{Z}[P(Y|X,Z)·Σ_{X,Y}[P(X,Y,Z)]]
	//   P(Y=1 | do(X=1)) = 0.6728
	// bow-arc identifiable: false
	//   not identifiable (hedge over {X,Y} ⊇ {Y})
}

// ExampleIdentifyConditional identifies a CONDITIONAL effect P(y | do(x), z) — the
// effect of X on Y within the subpopulation where the context Z takes a value.
// Here Z → X → Y with Z → Y: conditioning on the pre-treatment context Z, the
// effect is simply P(y | x, z). The algorithm recognises (via do-calculus Rule 2)
// that Z behaves like an intervention and returns that estimand.
func ExampleIdentifyConditional() {
	// Z → X → Y, Z → Y. X=0, Y=1, Z=2.
	g, _ := causa.NewDiagram([]string{"X", "Y", "Z"},
		[][2]int{{2, 0}, {0, 1}, {2, 1}}, nil)
	r, _ := causa.IdentifyConditional(g, []int{1}, []int{0}, []int{2})
	fmt.Println("identifiable:", r.Identifiable)
	fmt.Println("  estimand:", r)

	// Evaluate P(Y=1 | do(X=1), Z=0) on a discrete observational joint.
	joint, _ := causa.NewDistribution([]int{2, 2, 2},
		[]float64{0.05, 0.10, 0.08, 0.12, 0.15, 0.07, 0.18, 0.25})
	tab, _ := r.Estimand.Evaluate(joint)
	p, _ := tab.ProbAt(map[int]int{0: 1, 1: 1, 2: 0})
	fmt.Printf("  P(Y=1 | do(X=1), Z=0) = %.4f\n", p)

	// Output:
	// identifiable: true
	//   estimand: P(Y|X,Z)
	//   P(Y=1 | do(X=1), Z=0) = 0.5455
}

// ExampleIdentifyPAG decides whether a causal effect is identifiable from a PAG — a
// whole Markov equivalence class of graphs, as returned by FCI — rather than from a
// single asserted diagram. The extra difficulty: an effect counts as identifiable
// only when the SAME estimand is valid for every graph the data leaves possible, so
// a single undetermined endpoint (a circle) can flip the answer.
func ExampleIdentifyPAG() {
	// Z ↔ X → Y: X is confounded with Z by a latent cause, but the edge X → Y is
	// VISIBLE, so P(Y | do(X)) is identifiable across the entire class.
	vis, _ := causa.NewPAG([]string{"Z", "X", "Y"}, []causa.PAGEdge{
		{A: 0, B: 1, MarkA: causa.Arrow, MarkB: causa.Arrow}, // Z ↔ X
		{A: 1, B: 2, MarkA: causa.Tail, MarkB: causa.Arrow},  // X → Y
	})
	r1, _ := vis.Identify([]int{2}, []int{1})
	fmt.Println("Z<->X->Y  identifiable:", r1.Identifiable)

	// X o→ Y: the circle at X leaves open both X → Y (where do(X) matters) and
	// X ↔ Y (where it does not); the two disagree, so the effect is NOT identifiable.
	amb, _ := causa.NewPAG([]string{"X", "Y"}, []causa.PAGEdge{
		{A: 0, B: 1, MarkA: causa.Circle, MarkB: causa.Arrow}, // X o→ Y
	})
	r2, _ := amb.Identify([]int{1}, []int{0})
	fmt.Println("X o-> Y    identifiable:", r2.Identifiable)

	// Output:
	// Z<->X->Y  identifiable: true
	// X o-> Y    identifiable: false
}

// ExampleIdentifyConditionalPAG identifies a CONDITIONAL effect P(y | do(x), z) — the
// effect of x on y within the context z — from a PAG rather than a single diagram.
// It shows the honest subtlety of equivalence-class reasoning: the same directed
// chain that would be identifiable as an asserted DAG is NOT identifiable as a PAG,
// because there its edges are invisible and the class admits hidden confounding.
func ExampleIdentifyConditionalPAG() {
	// Z o→ X → Y: Z's arrowhead into X makes X → Y visible, so the effect of X on Y
	// in the context Z is identifiable.
	g1, _ := causa.NewPAG([]string{"Z", "X", "Y"}, []causa.PAGEdge{
		{A: 0, B: 1, MarkA: causa.Circle, MarkB: causa.Arrow}, // Z o→ X
		{A: 1, B: 2, MarkA: causa.Tail, MarkB: causa.Arrow},   // X → Y
	})
	r1, _ := g1.IdentifyConditional([]int{2}, []int{1}, []int{0})
	fmt.Println("Z o-> X -> Y          :", r1.Identifiable)

	// Z → X → Y with Z → Y, as a PAG: the directed edges are INVISIBLE (no witness),
	// so the class admits confounding and P(Y | do(X), Z) is not identifiable —
	// unlike the same graph asserted as a DAG.
	g2, _ := causa.NewPAG([]string{"X", "Y", "Z"}, []causa.PAGEdge{
		{A: 2, B: 0, MarkA: causa.Tail, MarkB: causa.Arrow}, // Z → X
		{A: 0, B: 1, MarkA: causa.Tail, MarkB: causa.Arrow}, // X → Y
		{A: 2, B: 1, MarkA: causa.Tail, MarkB: causa.Arrow}, // Z → Y
	})
	r2, _ := g2.IdentifyConditional([]int{1}, []int{0}, []int{2})
	fmt.Println("Z -> X -> Y, Z -> Y   :", r2.Identifiable)

	// Output:
	// Z o-> X -> Y          : true
	// Z -> X -> Y, Z -> Y   : false
}

// ExamplePAGIDResult_Evaluate turns an identified PAG effect into actual numbers.
// The PAG A o→ X ← o B, X → Y (two colliders orient arrowheads into X, making X → Y
// visible) identifies P(Y | do(X)); Evaluate computes it from a discrete
// observational joint. Because Y's only parent is X, the interventional equals the
// conditional P(Y | X).
func ExamplePAGIDResult_Evaluate() {
	// A=0, B=1, X=2, Y=3.
	g, _ := causa.NewPAG([]string{"A", "B", "X", "Y"}, []causa.PAGEdge{
		{A: 0, B: 2, MarkA: causa.Circle, MarkB: causa.Arrow}, // A o→ X
		{A: 1, B: 2, MarkA: causa.Circle, MarkB: causa.Arrow}, // B o→ X
		{A: 2, B: 3, MarkA: causa.Tail, MarkB: causa.Arrow},   // X → Y
	})
	r, _ := g.Identify([]int{3}, []int{2})
	fmt.Println("identifiable:", r.Identifiable)

	// A discrete observational joint P(A,B,X,Y), all binary, in mixed-radix order.
	prob := make([]float64, 16)
	// P(A),P(B) uniform-ish; P(X=1|A,B) and P(Y=1|X) chosen; build the joint.
	pX := [4]float64{0.2, 0.5, 0.5, 0.8} // P(X=1 | A,B) keyed A*2+B
	pY := [2]float64{0.3, 0.9}           // P(Y=1 | X)
	i := 0
	for a := 0; a < 2; a++ {
		for b := 0; b < 2; b++ {
			for x := 0; x < 2; x++ {
				for y := 0; y < 2; y++ {
					px := pX[a*2+b]
					if x == 0 {
						px = 1 - px
					}
					py := pY[x]
					if y == 0 {
						py = 1 - py
					}
					prob[i] = 0.25 * px * py // P(A)=P(B)=0.5
					i++
				}
			}
		}
	}
	joint, _ := causa.NewDistribution([]int{2, 2, 2, 2}, prob)

	tab, _ := r.Evaluate(joint)
	p1, _ := tab.ProbAt(map[int]int{2: 1, 3: 1}) // P(Y=1 | do(X=1))
	p0, _ := tab.ProbAt(map[int]int{2: 0, 3: 1}) // P(Y=1 | do(X=0))
	fmt.Printf("P(Y=1 | do(X=1)) = %.4f\n", p1)
	fmt.Printf("P(Y=1 | do(X=0)) = %.4f\n", p0)

	// Output:
	// identifiable: true
	// P(Y=1 | do(X=1)) = 0.9000
	// P(Y=1 | do(X=0)) = 0.3000
}

// ExampleExpr_EvaluateGaussian evaluates an identified estimand on a continuous,
// linear-Gaussian observational distribution — the continuous companion to the
// discrete Evaluate. It deconfounds a back-door: the structural effect of X on Y is
// 2, while the naive regression slope Cov(X,Y)/Var(X) = 5/2 = 2.5 is biased by the
// observed common cause Z. The adjustment recovers the causal 2.
func ExampleExpr_EvaluateGaussian() {
	// Back-door: X → Y confounded by an OBSERVED Z (Z → X, Z → Y). X=0, Y=1, Z=2.
	g, _ := causa.NewDiagram([]string{"X", "Y", "Z"},
		[][2]int{{0, 1}, {2, 0}, {2, 1}}, nil)
	r, _ := causa.Identify(g, []int{1}, []int{0})

	// The observational joint N(0, Σ) implied by Z→X, Z→Y and X→Y (with unit noise);
	// its X→Y structural coefficient is 2.
	joint, _ := causa.NewGaussian([]float64{0, 0, 0}, [][]float64{
		{2, 5, 1},
		{5, 14, 3},
		{1, 3, 1},
	})
	f, _ := r.Estimand.EvaluateGaussian(joint)

	// P(Y | do(X)) is linear in X, so a unit step gives the causal slope.
	hi, _ := f.Condition(map[int]float64{0: 1})
	lo, _ := f.Condition(map[int]float64{0: 0})
	m1, _ := hi.MeanAt(1)
	m0, _ := lo.MeanAt(1)
	fmt.Printf("dE[Y | do(X)]/dX = %.4f\n", m1-m0)

	// Output:
	// dE[Y | do(X)]/dX = 2.0000
}

// ExampleExpr_BootstrapGaussianEffect turns the point estimate of a causal effect
// into an effect ± robustness: a bootstrap confidence interval. It simulates a
// back-door SCM whose true X→Y effect is 2 (the naive regression slope would be a
// confounded ~2.5), then resamples the data to bound the identified, deconfounded
// estimate.
func ExampleExpr_BootstrapGaussianEffect() {
	// Simulate rows from Z→X, Z→Y and the causal X→Y of coefficient 2. X=0, Y=1, Z=2.
	rng := rand.New(rand.NewSource(1))
	const n = 3000
	X := make([]float64, n)
	Y := make([]float64, n)
	Z := make([]float64, n)
	for i := 0; i < n; i++ {
		z := rng.NormFloat64()
		x := z + rng.NormFloat64()
		y := 2*x + z + rng.NormFloat64()
		Z[i], X[i], Y[i] = z, x, y
	}
	data := [][]float64{X, Y, Z}

	g, _ := causa.NewDiagram([]string{"X", "Y", "Z"}, [][2]int{{0, 1}, {2, 0}, {2, 1}}, nil)
	r, _ := causa.Identify(g, []int{1}, []int{0})
	ci, _ := r.Estimand.BootstrapGaussianEffect(data, 0, 1,
		causa.BootstrapOptions{Resamples: 500, Level: 0.95, Seed: 7})

	fmt.Printf("effect estimate ≈ %.2f\n", ci.Point)
	fmt.Println("95% CI contains the true effect 2:", ci.Lower <= 2 && 2 <= ci.Upper)
	fmt.Println("95% CI excludes 0 (effect is significant):", ci.Lower > 0)

	// Output:
	// effect estimate ≈ 1.99
	// 95% CI contains the true effect 2: true
	// 95% CI excludes 0 (effect is significant): true
}
