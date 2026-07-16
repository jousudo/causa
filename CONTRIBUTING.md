# Contributing to causa

`causa` is pre-`v0.1.0` and still finding its core API. Issues and discussion are welcome now;
code contributions are easiest to land once Granger causality (the first release target) has
set the shape of the public API. If you want to work on something substantial, open an issue
first so the approach can be agreed before you invest time in code.

## Building and testing

The only dependency is the Go toolchain (`go.mod` pins `go 1.25`). No package manager, no
vendoring, nothing to install beyond Go itself:

```bash
go build ./...
go vet ./...
go test -race ./...
gofmt -l .          # must print nothing
```

These four commands are exactly what CI runs (`.github/workflows/ci.yml`), plus `staticcheck`
and `govulncheck`. A PR that passes all of them locally will pass CI.

## Hard constraint: standard library only

`causa` imports **nothing outside the Go standard library**, and it is **CGO-free**. This is
not a style preference — it is the reason the library exists (see the README's "Why"): a Go
program that needs causal inference today has to embed a Python/C scientific stack. A PR that
adds a third-party import or a `cgo` file will not be merged, no matter how good the algorithm
is. If a needed primitive (e.g. a specific linear-algebra routine) isn't in `math`/`sort` etc.,
implement the minimal version needed here rather than importing it.

## Expectations for new exported API

Every exported type, function, and method must ship with:

- **Godoc** — a doc comment starting with the symbol's name, explaining what it does and any
  non-obvious behavior (edge cases, complexity, numerical stability caveats).
- **A runnable `Example`** (`Example_xxx` in a `_test.go` file) demonstrating typical use —
  these are checked by `go test` and rendered on pkg.go.dev.
- **A benchmark** (`Benchmark_xxx`) for anything with non-trivial computational cost (matrix
  operations, iterative fitting, statistical tests). Causal methods are often O(n²) or worse in
  the number of variables; benchmarks make that cost visible instead of surprising.
- **Tests against ground truth**, not just "it runs": synthetic data with a known causal
  structure, and where possible a cross-check against a reference implementation's published
  results (see the README's "Validated, not asserted" principle). A p-value or coefficient
  computed by an algorithm implemented here should be traceable to a source (paper, textbook,
  reference implementation) in a code comment.

## API stability

`causa` is pre-1.0. Per semver, **any** release before `v1.0.0` may break the public API,
including patch-looking `v0.x` bumps if the change is deliberate and documented in the release
notes. Once `v1.0.0` ships, normal semver compatibility guarantees apply. Don't design around
"this will never change" until then.

## Package layout

The public API lives in the root `causa` package — flat, one import, idiomatic for a focused
library (see any well-known single-purpose Go library: `errors`, `context`). An `internal/`
package may appear later for implementation details that must not be part of the public API
(numerical helpers, matrix routines, etc.), but it is not created ahead of need — add it when
the first thing that belongs there exists, not before.

## PR etiquette

- Keep PRs focused: one algorithm or one fix per PR.
- Write commit messages and comments in English.
- Include the "why," not just the "what," in the PR description — for a causal-inference
  library, the reasoning behind a numerical choice matters as much as the code.
- By opening a PR you certify you have the right to submit the contribution under this
  project's license (Apache-2.0) — no separate sign-off process beyond that.
