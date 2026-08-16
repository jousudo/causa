# Changelog

All notable changes to this project are documented in this file. The project
follows semantic versioning after v1.0; before v1.0, release notes call out any
source or behavioral incompatibility explicitly.

## [Unreleased]

## [v0.14.0] - 2026-08-16

### Added

- Context-aware PC-Stable and FCI entry points with conditional-independence test
  budgets and auditable discovery diagnostics.
- Context-aware bootstrap entry points, replicate budgets, and attempted/failed
  replicate diagnostics.
- Configurable dense-table cell budgets for distribution construction, sampling,
  and discrete estimand evaluation.
- Regression and fuzz tests for state-space overflow, resource budgets,
  cancellation, and nil evaluator inputs.
- Cross-platform tests, a 90% statement-coverage gate, and a fuzz smoke test in CI.

### Changed

- Dense discrete APIs now reject integer-overflowing cardinality products and
  tables larger than `DefaultMaxDenseCells` before allocation. Callers with a
  reviewed memory budget can override this through `DenseOptions`.
- Staticcheck and govulncheck installations in CI are pinned to reproducible
  versions.
- Documentation now distinguishes time-series methods from algorithms that
  require independent observational rows.
- The selection-bias identification roadmap now acknowledges the March 2026
  Chen-Mooij preprint while keeping the capability unsupported until independently
  reproduced.

### Compatibility

- Existing PC-Stable, FCI, bootstrap, and estimand entry points remain available
  and delegate to the new bounded implementations with zero-value defaults.
- No selection-bias identification behavior changed: unsupported PAGs still fail
  closed with `ErrSelectionBiasUnsupported`.
