// Package causa provides causal inference and causal discovery for time
// series in pure Go (standard library only, CGO-free).
//
// Status: early development — v0.12.0 released; pre-v1.0, minor versions may
// still change the API.
//
// Implemented:
//   - Granger causality — pairwise VAR fitting (QR-fitted OLS) + F-test
//     (GrangerTest). Released in v0.1.0.
//   - Constraint-based causal discovery — the order-independent PC-stable
//     algorithm returning a CPDAG (PCStable), with a pluggable
//     conditional-independence test (CITest) whose linear-Gaussian
//     partial-correlation / Fisher-z default is shipped (FisherZTest,
//     PartialCorrelation). Released in v0.2.0.
//   - Directional causal discovery — the deterministic DirectLiNGAM algorithm
//     (DirectLiNGAM) for linear acyclic models with non-Gaussian independent
//     noise, recovering a full causal order and weighted coefficient matrix.
//     Released in v0.3.0.
//   - Interventions and counterfactuals — a linear structural equation model
//     (SEM, FitSEM) with the do-operator: interventional expectations
//     (Intervene), total causal effects (TotalEffect) and counterfactuals
//     (Counterfactual) by Pearl's abduction–action–prediction. Released in
//     v0.4.0. Exact for a fully specified linear SEM (e.g. one DirectLiNGAM
//     recovered); the general do-calculus IDENTIFICATION problem with latent
//     confounders remains research.
//   - Latent-confounder causal discovery — the FCI (Fast Causal Inference)
//     algorithm (FCI) returning a Partial Ancestral Graph (PAG). Unlike
//     PCStable it does NOT assume causal sufficiency: unobserved common causes
//     are admitted and reported as bidirected (↔) edges, recovered via
//     Possible-D-SEP refinement and Zhang's complete orientation rules
//     (assuming no selection bias). Released in v0.5.0.
//   - Causal-effect identification — the Shpitser–Pearl ID algorithm
//     (Identify) decides whether an interventional distribution P(y | do(x)) is
//     identifiable from the observational distribution in a causal diagram with
//     latent confounders, returning a symbolic estimand over P(V) (or a hedge
//     proving non-identifiability). A discrete evaluator (Expr.Evaluate) turns
//     the estimand into numbers from an observational joint. Released in v0.6.0.
//   - Conditional causal-effect identification — the Shpitser–Pearl IDC
//     algorithm (IdentifyConditional) identifies a CONDITIONAL interventional
//     distribution P(y | do(x), z), the effect of x on y within the context
//     z, via do-calculus Rule 2 (an m-separation test) plus the ID algorithm.
//     Released in v0.7.0.
//   - Causal-effect identification over an equivalence class — the
//     Jaber–Zhang–Bareinboim IDP algorithm (IdentifyPAG, or the (*PAG).Identify
//     method) decides whether P(y | do(x)) is identifiable from a PAG — the
//     Markov equivalence class FCI returns, not a single asserted diagram —
//     which is identifiable only when the SAME estimand holds for every graph in
//     the class. Sound and complete under the no-selection-bias scope. Returns a
//     symbolic (render-only) estimand; the identifiability DECISION is the
//     validated guarantee, cross-checked against the reference PAGId
//     implementation. Released in v0.8.0.
//   - Conditional causal-effect identification over an equivalence class — the
//     Jaber–Zhang–Bareinboim CIDP algorithm (IdentifyConditionalPAG, or the
//     (*PAG).IdentifyConditional method) decides whether the CONDITIONAL effect
//     P(y | do(x), z) — the effect of x on y within a context z — is identifiable
//     from a PAG, via the PAG do-calculus Rule 2 (definite-status m-separation on
//     a manipulated PAG) plus IDP. Decision cross-checked against PAGId::CIDP.
//     Released in v0.9.0.
//   - Numeric evaluation of a PAG effect — (*PAGIDResult).Evaluate turns an
//     identified IDP/CIDP estimand into the interventional table P(y | do(x)[, z])
//     from a discrete observational joint, the discrete-data companion to the
//     symbolic decision. Validated against brute-force truth on random latent SCMs
//     (with the PAG derived from each SCM by an oracle FCI). Released in v0.10.0.
//   - Continuous (linear-Gaussian) evaluation — Expr.EvaluateGaussian evaluates an
//     identified estimand against a normal observational joint (NewGaussian),
//     returning P(y | do(x)) as a Gaussian factor (GaussianFactor) over the query's
//     free variables; fix the intervention with Condition to read the outcome's
//     mean and covariance. A canonical-form Gaussian factor algebra (product,
//     division, Schur-complement marginalization) walks the same estimand AST as the
//     discrete Evaluate, exact for a linear-Gaussian model. Validated against the
//     closed-form structural effect (SEM.TotalEffect) on random latent SCMs, over
//     back-door and front-door estimands. Released in v0.11.0.
//   - Uncertainty quantification — a nonparametric bootstrap (Bootstrap) turns a
//     point-estimated effect into a confidence interval by resampling the data,
//     with SampleGaussian / SampleDistribution to fit an observational distribution
//     from raw samples and Expr.BootstrapGaussianEffect composing the whole
//     continuous path (fit → evaluate → contrast) into an interval for the causal
//     slope. Validated by its coverage on known linear-Gaussian SCMs — a nominal
//     95% interval covers the true effect about 95% of the time. Released in
//     v0.12.0.
//
// Research: identification under selection bias. See the README for the honest
// roadmap and the assumptions each method rests on: no capability is claimed
// before it is implemented, validated against ground-truth datasets, and
// benchmarked.
package causa
