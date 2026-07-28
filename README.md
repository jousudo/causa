# causa

Causal inference for Go. Pure standard library. Zero dependencies.

> **Status: early development — `v0.11.0` released.** Granger causality shipped in `v0.1.0`,
> PC-stable constraint-based discovery in `v0.2.0`, DirectLiNGAM directional discovery in
> `v0.3.0`, linear-SEM interventions + counterfactuals (the do-operator) in `v0.4.0`,
> FCI latent-confounder discovery (returning a PAG) in `v0.5.0`, the Shpitser–Pearl ID
> algorithm for causal-effect identification in `v0.6.0`, the IDC algorithm for
> conditional-effect identification in `v0.7.0`, the IDP algorithm for
> identification over a PAG (an equivalence class) in `v0.8.0`, the CIDP
> algorithm for conditional identification over a PAG in `v0.9.0`, numeric
> evaluation of the PAG estimand in `v0.10.0`, and continuous (linear-Gaussian)
> estimand evaluation in `v0.11.0`.
> Pre-1.0, minor versions may break the API. Nothing
> below is claimed as shipped until it is implemented, tested against ground-truth datasets, and
> benchmarked. This README is kept honest by policy: capabilities are labeled exactly as they are.

## What

`causa` is a causal inference and causal discovery library for time series, written in pure
Go (stdlib only, CGO-free). It is being built to power deterministic root-cause analysis in
AIOpsFlow — an AIOps platform by the same author, currently in private development ahead of
its own open-source release — but `causa` is designed as a general-purpose, standalone
library with no ties to any host application.

## Why

Correlation is not causation, and nowhere does that bite harder than in production telemetry:
a network saturation event raises database CPU *and* API latency at the same time, and naive
correlation blames the database. Distinguishing real causal structure from coincident movement
requires actual causal mathematics — conditional independence, structural asymmetries,
intervention modeling — not just pattern matching.

Today that mathematics lives almost entirely in Python (DoWhy, causal-learn, gCastle,
Tigramite). A Go service that needs causal reasoning must ship a Python sidecar with a
scientific stack — hundreds of megabytes of runtime and a supply chain to audit. `causa`
exists to remove that hop: causal inference as a plain Go import, small enough to embed
anywhere Go runs.

## Roadmap

| Capability | Method | Status |
|---|---|---|
| Granger causality | Pairwise OLS autoregressions (QR-fitted) + F-test | **Released in `v0.1.0`** — ground-truth-validated and benchmarked; flags confounders by design (see below) |
| Constraint-based discovery | PC-stable algorithm (conditional-independence tests) → CPDAG | **Released in `v0.2.0`** — ground-truth-validated and benchmarked; recovers a Markov equivalence class, not a unique DAG (see below) |
| Directional discovery | DirectLiNGAM (deterministic, non-Gaussian noise) → causal order + weighted DAG | **Released in `v0.3.0`** — ground-truth-validated and benchmarked; identifies a fully directed model when the noise is non-Gaussian (see below) |
| Interventions / counterfactuals | Linear SEM + do-operator (forward substitution; Pearl abduction–action–prediction) | **Released in `v0.4.0`** — exact for a fully specified linear SEM; the general do-calculus *identification* problem (latent confounders) remains research (see below) |
| Latent-confounder discovery | FCI (Possible-D-SEP + Zhang's rules) → PAG | **Released in `v0.5.0`** — ground-truth-validated and benchmarked; drops causal sufficiency, reporting latent common causes as bidirected (↔) edges; assumes no selection bias (see below) |
| Causal-effect identification | Shpitser–Pearl ID → symbolic estimand + discrete evaluator | **Released in `v0.6.0`** — validated against brute-force truth on random latent SCMs; decides identifiability of `P(y \| do(x))` in a diagram with latent confounders, or proves non-identifiability with a hedge (see below) |
| Conditional-effect identification | Shpitser–Pearl IDC (do-calculus Rule 2 + m-separation + ID) | **Released in `v0.7.0`** — validated against brute-force truth on random latent SCMs; identifies `P(y \| do(x), z)`, the effect of `x` on `y` within a context `z` (see below) |
| Identification over an equivalence class | Jaber–Zhang–Bareinboim IDP (pc-components + regions, Prop. 6/7 over induced PAGs) | **Released in `v0.8.0`** — decides identifiability of `P(y \| do(x))` from a **PAG** (what FCI returns), not a single asserted diagram; decision cross-checked case-for-case against the reference `PAGId` implementation; symbolic (render-only) estimand; assumes no selection bias (see below) |
| Conditional identification over an equivalence class | Jaber–Zhang–Bareinboim CIDP (PAG do-calculus Rule 2 + definite-status m-separation + IDP) | **Released in `v0.9.0`** — decides identifiability of `P(y \| do(x), z)` from a **PAG**, the contextual effect; decision cross-checked against `PAGId::CIDP`; assumes no selection bias (see below) |
| Numeric evaluation of a PAG effect | Discrete evaluator over the identified IDP/CIDP estimand | **Released in `v0.10.0`** — turns an identified PAG effect into the interventional table `P(y \| do(x)[, z])` from a discrete joint; validated against brute-force truth on random latent SCMs (PAG per SCM via an oracle FCI) |
| Continuous (linear-Gaussian) evaluation | Canonical-form Gaussian factor algebra over the identified estimand (`Expr.EvaluateGaussian`) | **Released in `v0.11.0`** — evaluates an identified estimand on a *normal* observational joint, returning `P(y \| do(x))` as a Gaussian; exact for a linear-Gaussian model; validated against the closed-form structural effect (`SEM.TotalEffect`) on random latent SCMs |
| Uncertainty quantification | Bootstrap confidence intervals over the evaluated effect | Planned (`v0.12.0`) — turns a point estimate into an estimate ± robustness |
| Selection bias | Zhang's rules R5–R7 in FCI + selection in IDP/CIDP identification | Research (`v0.13.0`) — the largest, most independent step; least immediate value, highest risk |

Granger tells you that series *A* helps predict series *B* — necessary but not sufficient for
causation (confounders fool it). The PC algorithm and LiNGAM are what upgrade "predictive
precedence" into defensible causal structure: PC recovers a Markov equivalence class, and
DirectLiNGAM — where the non-Gaussian-noise assumption holds — pins down the full direction that
PC must leave reversible.

### Constraint-based discovery (PC-stable)

`PCStable(data, names, opts)` recovers causal structure from a panel of continuous variables
(one slice per variable). It is the order-independent PC-stable algorithm of Colombo & Maathuis
(2014): it thins a complete graph down to a skeleton with conditional-independence tests, orients
unshielded colliders, and closes under Meek's rules R1–R4. The default conditional-independence
test (`FisherZTest`) is the linear-Gaussian partial correlation — computed by QR-residualization
that reuses the same Householder solver as the Granger path — transformed by Fisher's *z*; the
`CITest` extension point lets you supply another test for non-Gaussian or discrete data.

The test of conditional independence $x_i \perp x_j \mid S$: residualize both variables on
$[1, S]$, correlate the residuals, and refer the variance-stabilized statistic to the standard
normal,

```math
r_{ij\cdot S} = \mathrm{corr}\big(x_i - \hat{x}_i^{(S)},\; x_j - \hat{x}_j^{(S)}\big),
\qquad
z = \tfrac{1}{2}\,\ln\frac{1+r_{ij\cdot S}}{1-r_{ij\cdot S}},
```

```math
T = \sqrt{n - |S| - 3}\;\lvert z\rvert \;\overset{H_0}{\sim}\; \mathcal{N}(0,1),
\qquad
p = 2\big(1 - \Phi(T)\big) = \mathrm{erfc}\big(T/\sqrt{2}\big);
```

the edge is deleted (independence accepted) when $p > \alpha$ (default $\alpha = 0.05$), and the
levels stop growing once $n - |S| - 3 < 1$ (the honest small-sample cap described below).

**Output is a CPDAG, not a DAG.** Constraint-based discovery identifies structure only up to
*Markov equivalence*. A **directed** edge `A → C` is compelled (every DAG consistent with the data
agrees on it); an **undirected** edge `A — B` is reversible (the equivalence class contains DAGs
orienting it either way). A chain `A → B → C` and a fork `A ← B → C` are indistinguishable from
observational data, so their edges come back undirected; a collider `A → C ← B` is oriented,
because it alone implies `A ⫫ B`. Read an undirected edge as "direction not identified", never as
"no causal link".

**Assumptions** (standard for constraint-based discovery, and stated because they are easy to
violate in practice): *causal sufficiency* (no unobserved common cause of two measured variables),
*faithfulness* (every conditional independence in the data is entailed by the graph), and a
correct CI test (the default assumes linear-Gaussian data). On small samples there is an honest
cap: conditioning sets stop growing once `n − |S| − 3 < 1`, so independencies that need larger
conditioning sets than the sample supports cannot be tested and some edges a larger sample would
remove may remain.

### Latent-confounder discovery (FCI)

`FCI(data, names, opts)` runs the Fast Causal Inference algorithm of Spirtes, Glymour & Scheines
and returns a **Partial Ancestral Graph (PAG)**. It is the constraint-based sibling of `PCStable`
with one crucial difference: **it drops causal sufficiency.** PC assumes every common cause is
measured; FCI does not, and where the data imply an unobserved common cause it says so — an edge
`A ↔ B` — instead of inventing a spurious direct link. This is the difference between "the database
causes the latency" and "something you are not measuring drives both".

**A PAG is read endpoint by endpoint, not as whole arrows.** Each edge end carries one of three
marks: an **arrowhead** (`>`, a *definite non-ancestor* — compelled across the whole equivalence
class), a **tail** (`-`, a *definite ancestor*), or a **circle** (`o`, *undetermined*). So the
edge types mean:

- `A → B` — a definite direction: `B` is not a cause of `A` in any graph of the class.
- `A ↔ B` — **a latent common cause**: neither causes the other; something unobserved drives both.
  This is the structure causal sufficiency cannot express, and the reason FCI exists.
- `A o→ B` — the arrowhead at `B` is compelled, but `A`'s end is undetermined.
- `A o—o B` — wholly undetermined direction (as an undirected CPDAG edge is).

Reading a circle as "no relationship" is the classic error — it means *direction not identified*.

**Algorithm.** FCI starts from the PC-stable skeleton (shared code, same order-independence), then
does what PC cannot: the **Possible-D-SEP** refinement. The PC skeleton can leave an edge that is
removable only by conditioning on variables *not adjacent* to the pair; FCI hunts for exactly those
separating sets. A vertex $v$ is in $\mathrm{Possible\text{-}D\text{-}SEP}(a)$ when some path
$a \dots v$ has, for **every** consecutive triple $\langle x, y, z\rangle$ on it, either $y$ a
collider on the path ($x \mathbin{*}\!\!\to y \leftarrow\!\!\mathbin{*} z$) or $\{x, y, z\}$ a
triangle:

```math
v \in \mathrm{Possible\text{-}D\text{-}SEP}(a) \iff
\exists\, \text{path } a \dots v \ \text{s.t. } \forall\, \langle x,y,z\rangle:\ 
y \text{ is a collider on the path} \ \lor\ x, z \text{ adjacent.}
```

For every still-adjacent pair FCI tests conditional independence against subsets of this set,
deleting the edge (and recording the separating set) when a test passes. The Possible-D-SEP sets
are snapshotted **before** any deletion, so — exactly like the PC-stable skeleton — the result does
not depend on the order pairs are visited. Unshielded colliders are then oriented (from the
refined separating sets), and the graph is completed by **Zhang's orientation rules R1–R4 and
R8–R10**, applied to closure: R1–R3 propagate arrowheads that would otherwise create an unlicensed
collider or cycle, R4 is the discriminating-path rule, and R8–R10 add the compelled *tails* that
distinguish a definite `→` from an `o→`.

**Scope — no selection bias.** This implementation assumes the sampling process induces no
selection (no conditioning by how the data were collected). It therefore never produces tail–tail
(`—`) edges, and **Zhang's selection-bias rules R5–R7 are deliberately omitted** — on a
no-selection PAG they can never fire, and running them could only introduce spurious `—` edges.
R1–R4 and R8–R10 are the sound **and complete** rule set for this class, so the PAG is maximally
informative under the stated assumption. Admitting selection bias is a strictly larger problem and
is out of scope here — stated plainly rather than left for a user to discover.

**Assumptions** (stated because, as everywhere here, violating them silently returns a plausible
but wrong graph): *faithfulness*, a correct CI test (the default `FisherZTest` assumes
linear-Gaussian data), and the *no-selection-bias* scope above. Causal sufficiency is **not**
required — that is the entire point. The honest small-sample cap carries over from PC: conditioning
sets stop growing once `n − |S| − 3 < 1`.

Reference: Spirtes, Glymour & Scheines, *Causation, Prediction, and Search* (2nd ed., 2000), ch. 6
(FCI, Possible-D-SEP); Zhang, "On the completeness of orientation rules for causal discovery in the
presence of latent confounders and selection bias", *Artificial Intelligence* 172 (2008)
(rules R1–R10); Colombo & Maathuis, *JMLR* 15 (2014) (order-independent skeleton).

### Directional discovery (DirectLiNGAM)

`DirectLiNGAM(data, names, opts)` estimates a Linear Non-Gaussian Acyclic Model (LiNGAM) from a
panel of continuous variables and returns a **full causal order plus the weighted coefficient
matrix `B`** of the structural model `x = B·x + e`. It is the deterministic DirectLiNGAM method of
Shimizu et al. (JMLR 2011) — **not** ICA-LiNGAM: there is no random ICA initialization, so, like
the rest of `causa`, the same input always yields the same output. It iteratively finds the most
exogenous variable by the paper's entropy-based independence measure (a maximum-entropy negentropy
approximation built from `E[log cosh]` and `E[x·exp(−x²/2)]` moments), regresses it out of the
rest, and recurses; the connection strengths are then least-squares estimates on the original data.
Reused throughout is the same Householder-QR OLS solver as the Granger and PC paths.

The model and the mathematics behind the direction choice: for standardized data the structural
model is

```math
x = B\,x + e,
```

with $B$ strictly lower-triangular in the causal order and $e$ mutually independent, non-Gaussian
disturbances. The differential entropy of a standardized variable $u$ is estimated by the
maximum-entropy approximation (Hyvärinen 1998),

```math
\hat{H}(u) \;\approx\; \tfrac{1}{2}\big(1+\ln 2\pi\big)
\;-\; k_1\big(\mathbb{E}[\ln\cosh u] - \gamma\big)^2
\;-\; k_2\big(\mathbb{E}[\,u\,e^{-u^2/2}\,]\big)^2,
```

with $k_1 = 79.047$, $k_2 = 7.4129$, $\gamma = 0.37457$. For a candidate cause $x_i$ against
$x_j$, the directional statistic is the log-likelihood ratio of the two directions,

```math
T \;=\; \big(H(x_j) + H(r_i^{(j)})\big) \;-\; \big(H(x_i) + H(r_j^{(i)})\big),
```

where $r_i^{(j)}$ is the standardized OLS residual of $x_i$ on $x_j$ (and vice versa): a positive
$T$ favors $x_i \to x_j$. Each candidate is scored by $\sum_j \min(0, T)^2$ — penalizing only the
pairs that testify *against* its exogeneity — and the minimizer (a true root scores ≈ 0) is peeled
off; ties break to the lowest index, keeping the whole procedure deterministic.

**Where PC leaves an edge undirected, LiNGAM directs it.** A chain `A → B → C` and a fork
`A ← B → C` are one Markov equivalence class — indistinguishable to a constraint-based method — but
DirectLiNGAM separates them, because non-Gaussian noise breaks the symmetry that made them
equivalent. The output is a single fully directed DAG, not an equivalence class.

**Assumptions** (stated bluntly because violating them silently returns a plausible but wrong
model): *linearity*, *acyclicity*, *causal sufficiency*, and — the load-bearing one — *non-Gaussian,
mutually independent noise* (at most one disturbance may be Gaussian). **On Gaussian noise the model
is fundamentally unidentifiable**: a linear-Gaussian SEM and its reverse fit the data equally well,
so the recovered order is arbitrary and meaningless. DirectLiNGAM cannot detect this and returns a
fully oriented model regardless; a LiNGAM result is only trustworthy when the non-Gaussianity
assumption genuinely holds. This failure mode is pinned by an honest-failure test. One conscious
divergence from the reference implementations: coefficient pruning is a simple absolute-magnitude
threshold (`LiNGAMOptions.PruneThreshold`), not adaptive-lasso, which would require an L1 optimizer
this stdlib-only library does not carry.

### Interventions and counterfactuals (SEM + do-operator)

`NewSEM(names, B, intercept)` and `FitSEM(data, names, order)` build a linear structural equation
model

```math
x = c + B\,x + e,
```

with $B$ acyclic — $B_{ij}$ is the direct causal effect of $x_j$ on $x_i$, non-zero only when $j$
precedes $i$ in a causal order — an intercept vector $c$, and mutually independent, mean-zero
disturbances $e$. `NewSEM` takes a hand-specified matrix (and topologically sorts it, rejecting a
cycle); `FitSEM` estimates $B$ and $c$ from data by regressing each variable on its predecessors in a
given causal order via the same Householder-QR solver as the rest of the library. The order a
DirectLiNGAM run recovers is exactly such an input, so **discovery and intervention compose**:
`FitSEM(data, r.Nodes(), r.CausalOrder())`.

**The do-operator.** `Intervene(do)` returns the interventional expectation $\mathbb{E}[x \mid
\mathrm{do}(x_S = \xi_S)]$. Fixing the intervened variables severs their incoming edges; every other
variable takes its structural mean given its parents. In causal order this is a single forward
substitution,

```math
\mathbb{E}[x_i \mid \mathrm{do}(\cdot)] =
\begin{cases}
\xi_i & i \in S,\\[2pt]
c_i + \sum_j B_{ij}\,\mathbb{E}[x_j \mid \mathrm{do}(\cdot)] & i \notin S,
\end{cases}
```

equivalently the reduced form $x = (I - B)^{-1}(c + e)$ with the intervened components clamped.
`TotalEffect(from, to)` returns the total causal effect — the sum over every directed path of the
product of its edge coefficients, the $(I-B)^{-1}$ entry — computed as $\mathbb{E}[\text{to} \mid
\mathrm{do}(\text{from}=1)] - \mathbb{E}[\text{to} \mid \mathrm{do}(\text{from}=0)]$.

**Counterfactuals.** `Counterfactual(observed, do)` answers "given that we *observed* $x$, what would
$x$ have been had we done $\mathrm{do}$?" by Pearl's abduction–action–prediction:

```math
\underbrace{e = (I - B)\,x_{\text{obs}} - c}_{\text{abduction}},
\qquad
\underbrace{\text{fix } \mathrm{do}\text{-vars, cut their in-edges}}_{\text{action}},
\qquad
\underbrace{x'_i = c_i + \sum_j B_{ij}\,x'_j + e_i}_{\text{prediction}},
```

re-propagating the structural equations while holding the abducted disturbances fixed. With an empty
intervention the counterfactual reproduces the observation exactly — the consistency the method
satisfies by construction.

**Scope** (stated bluntly, as everywhere here). This is the *identified* linear-SEM case: it assumes
you already hold the full structural model — every direct effect and the acyclic structure. It does
**not** solve the general do-calculus *identification* problem — recovering an interventional
distribution from an *observational* one plus a partially known graph with **latent confounders** —
which is exactly what `Identify` (below) does. What ships here is exact for a
fully specified linear SEM, where interventions and counterfactuals reduce to forward substitution
through the structural equations. Linearity, acyclicity and causal sufficiency carry over from the
model you supply; a mis-specified $B$ returns a plausible but wrong answer, silently.

### Causal-effect identification (ID algorithm)

`Identify(g, y, x)` answers the question the linear-SEM path cannot: given only a causal *diagram*
— not a fully specified model — **can the interventional distribution `P(y | do(x))` be computed
from observational data at all**, and if so, what is the formula? It is the Shpitser–Pearl ID
algorithm, sound *and complete* for a single causal diagram: if it returns an estimand the effect is
identifiable, and if it returns a hedge the effect is provably **not** identifiable — no algorithm
could do better.

The input is a `Diagram`, an acyclic directed mixed graph (ADMG) with two edge kinds: directed
`X → Y` (direct causation) and **bidirected `X ↔ Y`** (an unobserved common cause — a latent
confounder). Causal sufficiency is *not* assumed; that is the whole point. This is the graph a
`v0.5.0` FCI run is evidence about — though `Identify` takes a single asserted diagram, not a PAG
(identifying over an equivalence class is the separate **IDP** algorithm, `v0.8.0`, below).

**Identifiability, and why it can fail.** An effect is identifiable when every causal model
consistent with the diagram that agrees on the observational `P(V)` also agrees on `P(y | do(x))`.
When two models can match on all observables yet disagree on the interventional answer, no estimand
exists — the **bow arc** `X → Y` with `X ↔ Y` is the canonical failure. The algorithm detects these
via *hedges* (a C-forest structure); `IDResult.Identifiable` reports which case holds and
`IDResult.Hedge` names the obstructing vertices.

**The estimand and the discrete evaluator.** When identifiable, the result is a symbolic `Expr` over
the observational joint, built from marginalization, products and conditionals — the Tian–Pearl
C-component factorization. For the classic back-door graph it reads

```math
P(y \mid do(x)) \;=\; \sum_{z} P(y \mid x, z)\,P(z),
```

and for the front-door graph `X → M → Y` with `X ↔ Y` (identifiable *despite* the latent confounder)

```math
P(y \mid do(x)) \;=\; \sum_{m} P(m \mid x)\sum_{x'} P(y \mid m, x')\,P(x').
```

The estimand is exact but *not* algebraically simplified, so it may not read character-for-character
like the textbook form — its **value** is what is guaranteed. `Expr.Evaluate` turns it into numbers:
give it a discrete observational joint (`Distribution`) and it returns the interventional table
`P(y | do(x))`. This is how the implementation is validated — against brute-force truncated
factorization on *random* discrete latent SCMs, the parameterization-independent proof of
correctness — and it is a useful capability in its own right.

**Scope.** Sound and complete identification for one causal diagram **without selection bias**;
non-identifiability is returned as a result, not an error. Out of scope (and stated so plainly):
selection bias, identification over a PAG (IDP), and estimating the identified estimand from
*continuous* samples — the evaluator is for discrete joints. Reference: Shpitser & Pearl,
"Identification of Joint Interventional Distributions in Recursive Semi-Markovian Causal Models"
(AAAI 2006); Tian & Pearl, "A General Identification Condition for Causal Effects" (AAAI 2002).

**Conditional effects.** `IdentifyConditional(g, y, x, z)` identifies `P(y | do(x), z)` — the effect
of `x` on `y` *within the subpopulation where the context `z` takes a value* — the query that
actually drives contextual decisions ("what does this remediation do to the SLO **for this kind of
service**?"). It is the Shpitser–Pearl IDC algorithm: do-calculus **Rule 2** moves each conditioning
variable that behaves like an intervention out of `z` and into `x` — decided by an **m-separation**
test (d-separation generalized to mixed graphs, `X ⫫ Y | Z` accounting for bidirected edges) in the
graph with the edges into `x` and out of the candidate removed — until none remains movable; the
residual is then

```math
P(y \mid do(x), z) \;=\; \frac{P_x(y, z)}{P_x(z)} \;=\; \frac{P_x(y, z)}{\sum_{y} P_x(y, z)},
```

with the joint `P_x(y, z)` handed to the ID algorithm above. With `z` empty it is exactly `Identify`;
if the joint is not identifiable, neither is the conditional effect. Reference: Shpitser & Pearl,
"Identification of Conditional Interventional Distributions" (UAI 2006).

### Identification over a PAG (IDP)

`IdentifyPAG(g, y, x)` (or the `(*PAG).Identify` method) closes the loop with discovery: it decides
whether `P(y | do(x))` is identifiable directly from a **PAG** — the Markov equivalence class that
`FCI` returns — rather than from a single diagram you had to assert. This is the guarantee an
autonomous system actually needs: when the graph itself was *learned from data*, the honest question
is not "is the effect identifiable in one graph I picked?" but "is it identifiable in **every** graph
the data leaves possible?" It is the Jaber–Zhang–Bareinboim IDP algorithm, sound and complete for
this problem under the no-selection-bias scope.

The difficulty over a single diagram is the circle mark. An edge `X o→ Y` leaves open both `X → Y`
(where `do(x)` matters) and `X ↔ Y` (where it does not); the two disagree on `P(y | do(x))`, so the
effect is **not** identifiable — a single undetermined endpoint flips the answer. IDP works by the
same C-component logic as ID, lifted to PAGs: it reduces the query over the *possible ancestors* of
`Y`, splitting the working set by **buckets** (o–o components), **pc-components** and **regions**, and
peeling off marginalizable pieces via Propositions 6 and 7 — with edge **visibility** (whether a
directed edge can be hiding confounding) judged on the full PAG throughout.

```go
// Z ↔ X → Y: X is confounded with Z, but the edge X → Y is VISIBLE, so the
// effect is identifiable across the whole equivalence class.
g, _ := causa.NewPAG([]string{"Z", "X", "Y"}, []causa.PAGEdge{
    {A: 0, B: 1, MarkA: causa.Arrow, MarkB: causa.Arrow}, // Z ↔ X
    {A: 1, B: 2, MarkA: causa.Tail, MarkB: causa.Arrow},  // X → Y
})
r, _ := g.Identify([]int{2}, []int{1})
fmt.Println(r.Identifiable) // true
```

**Scope and validation.** IDP returns the identifiability **decision** — the sound, tested guarantee,
cross-checked case-for-case against the authors' reference `PAGId` implementation on byte-identical
adjacency matrices — together with the symbolic Prop. 6/7 estimand. To turn it into numbers, call
`(*PAGIDResult).Evaluate` (see below); the raw `Estimand` is the exact formula and `String()`s
faithfully. Out of scope for now: — as everywhere in this library — selection bias. Reference: Jaber,
Ribeiro, Zhang & Bareinboim, "Causal Identification under Markov Equivalence: Calculus, Algorithm, and
Completeness" (NeurIPS 2022).

**Conditional effects over a PAG.** `IdentifyConditionalPAG(g, y, x, z)` (or `(*PAG).IdentifyConditional`)
is to `IdentifyPAG` what IDC is to ID: it decides whether the **contextual** effect `P(y | do(x), z)`
is identifiable from the equivalence class — "what does this remediation do to the SLO **for this
class of service**?", now honest about latent confounding. It is the Jaber–Zhang–Bareinboim CIDP
algorithm: the PAG **do-calculus Rule 2** (a definite-status **m-separation** test in the manipulated
PAG `P_{\overline{W},\underline{X}}` — edges into `W` removed, then *visible* edges out of `X`
removed) moves each context variable that behaves like an intervention from `z` into `x`, after which
the residual `P_x(y, z) / \sum_y P_x(y, z)` is handed to IDP. Same scope as IDP: the identifiability
**decision** is the validated guarantee (cross-checked against `PAGId::CIDP`), and the estimand is
evaluated with `(*PAGIDResult).Evaluate`. With `z` empty it is exactly `IdentifyPAG`.

### Numeric evaluation of a PAG effect

`(*PAGIDResult).Evaluate(joint)` turns an identified IDP or CIDP effect into actual numbers: given a
discrete observational joint `P(V)`, it returns the interventional table `P(y | do(x))` — or
`P(y | do(x), z)` — read with `ProbAt` exactly as an `Identify` estimand's `Evaluate` output. The raw
Prop. 6/7 estimand is a c-factor expression that, evaluated directly, carries "spectator" variables
(non-ancestors of the outcome the c-factor form leaves in, which come out constant); `Evaluate`
reduces them away and returns a clean table over `Y ∪ X (∪ Z)`. So call `Evaluate`, not
`Estimand.Evaluate`.

```go
r, _ := g.Identify([]int{y}, []int{x})   // identifiable
tab, _ := r.Evaluate(joint)              // joint is a discrete *Distribution over P(V)
p, _ := tab.ProbAt(map[int]int{x: 1, y: 1}) // P(Y=1 | do(X=1))
```

**Validation.** Correctness is checked against brute-force truth on **random discrete latent SCMs**,
the same parameterization-independent standard used for `Identify`: for each SCM the PAG is derived by
an **oracle FCI** (a d-separation oracle fed to this package's own `FCI`, so the PAG is exactly the one
the SCM's equivalence class implies), the observational joint and the true `P(y | do(x)[, z])` are
computed by marginalizing latents and intervening, and the evaluated estimand is required to match —
across random parameterizations, including latent-confounded, Prop-6-heavy estimands. As with
`Expr.Evaluate`, the evaluator is for **discrete** joints.

### Continuous (linear-Gaussian) evaluation

`Expr.EvaluateGaussian(joint)` is the continuous companion to `Expr.Evaluate`: where the discrete
evaluator reads a probability table, this reads a **normal** observational joint `N(μ, Σ)`
(`NewGaussian`) and returns `P(y | do(x))` as a Gaussian, **exact for a linear-Gaussian model**. It
is the estimand backend most telemetry needs — continuous metrics rather than binned states.

The identified estimand is the *same* symbolic `Expr` the ID/IDC algorithms produce; only the numeric
backend changes. A Gaussian factor in **canonical (information) form** — `φ(x) = exp(−½·xᵀ K x + hᵀx + g)`
— closes exactly the three operations an estimand is built from: products **add** `(K, h, g)`, a
conditional's ratio **subtracts** them, and marginalizing a variable block out is a **Schur
complement**. So `EvaluateGaussian` walks the same AST as `Evaluate`, swapping dense tables for
Gaussian factors. The result is a `GaussianFactor` over the query's free variables `Y ∪ X`; fix the
intervention with `Condition` to read the outcome's mean and covariance:

```go
r, _ := causa.Identify(g, []int{y}, []int{x})    // identifiable
f, _ := r.Estimand.EvaluateGaussian(joint)       // joint is a *GaussianDistribution N(μ, Σ)
hi, _ := f.Condition(map[int]float64{x: 1})       // P(Y | do(X=1)) as a Gaussian
lo, _ := f.Condition(map[int]float64{x: 0})       // P(Y | do(X=0))
m1, _ := hi.MeanAt(y)
m0, _ := lo.MeanAt(y)                              // m1 − m0 is the causal slope dE[Y|do(X)]/dX
```

The interventional distribution `P(y | do(x))` is a distribution over `y` **for each fixed** `x`, so
the estimand factor is proper in the outcome but flat in the intervention (its `x`-block is degenerate
by design) — never marginalize `x` out; `Condition` on it. A degenerate input joint, or an
intermediate factor whose eliminated block is not positive definite, is reported as
`ErrNotPositiveDefinite` rather than silently returning garbage — the Cholesky factorization that
drives the algebra doubles as the positive-definiteness test.

**Validation.** Correctness is checked against **closed-form structural truth** on random
**linear-Gaussian latent SCMs**, the continuous analogue of the discrete SCM harness: for each diagram
a random linear-Gaussian model is drawn (one latent root per bidirected edge), the exact observed
covariance `Σ = (I−B)⁻¹ Ω (I−B)⁻ᵀ` is formed, and the interventional slope read off the evaluated
estimand is required to match the structural total effect the independent `SEM.TotalEffect` do-operator
path computes — across random parameterizations, over back-door and (latent-confounded) front-door
estimands. **Scope.** Exact for a linear-Gaussian model given the *distribution*; estimating `Σ` from
finite samples and quantifying the resulting uncertainty is `v0.12.0` (bootstrap).

### Granger causality

`GrangerTest(cause, effect, lags)` is available since `v0.1.0` (import path
`github.com/jousudo/causa`). It fits a restricted autoregression of `effect` on its own lags and an unrestricted one
that adds `cause`'s lags, both via a Householder-QR least-squares solver, and reports the
F-statistic and its p-value:

```math
\text{restricted:}\quad y_t = c + \sum_{i=1}^{p} a_i\,y_{t-i} + \varepsilon_t
\qquad\quad
\text{unrestricted:}\quad y_t = c + \sum_{i=1}^{p} a_i\,y_{t-i} + \sum_{i=1}^{p} b_i\,x_{t-i} + \varepsilon_t
```

```math
F = \frac{(\mathrm{RSS}_r - \mathrm{RSS}_u)/p}{\mathrm{RSS}_u/(n - 2p - 1)}
\;\overset{H_0}{\sim}\; F(p,\; n-2p-1),
\qquad H_0:\ b_1 = \dots = b_p = 0,
```

with the p-value evaluated through the regularized incomplete beta function (continued-fraction
form). "x Granger-causes y" is the rejection of $H_0$ — x's past improves the prediction of y
beyond y's own past. **Known limitation — confounding:** if a hidden common cause drives
both series, Granger reports causality even when no direct edge exists. This is inherent to the
method, not a defect, and it is exactly why the PC algorithm and LiNGAM are on the roadmap; the
behavior is pinned by a dedicated test (`TestGrangerFlagsConfounder`) and documented on the
function so it is never mistaken for a true causal claim.

## Design principles

- **Standard library only.** No third-party imports, ever. CGO-free. The whole supply chain
  is the Go toolchain.
- **Deterministic and reproducible.** Same input, same output — causal claims must be
  auditable.
- **Documented like it matters.** Every exported symbol carries godoc; every feature ships
  with testable `Example` functions and benchmarks.
- **Validated, not asserted.** Algorithms are tested against synthetic ground-truth suites
  and cross-checked against reference implementations before any release claims them.
- **Semantic versioning** from `v0.1.0`. Pre-1.0, minor versions may break the API; this is
  stated here instead of discovered by users.

## Installation

```bash
go get github.com/jousudo/causa
```

## Contributing

Issues, discussion and code contributions are welcome; note that pre-1.0, minor versions may
still adjust the API. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
stdlib-only constraint, godoc/Example/benchmark expectations, and PR etiquette.

## License

[Apache License 2.0](LICENSE)
