# Changelog

All notable changes to this project are documented in this file. The project
follows semantic versioning after v1.0; before v1.0, release notes call out any
source or behavioral incompatibility explicitly.

## [Unreleased]

## [v0.15.0] - 2026-08-16

### Added

- Reduced-form multivariate VAR fitting with intercepts, coefficient matrices,
  innovation residuals/covariance, and defensive result accessors.
- Common-sample AIC, BIC, and Hannan-Quinn lag-order selection plus residual
  cross-autocorrelation diagnostics.
- Conditional multivariate Granger F-tests that remove the proposed cause lags
  while retaining every other supplied variable history.
- Moving-block and Politis-Romano stationary bootstrap engines, including
  context-aware and Gaussian-effect helpers for serially dependent observations.
- An independent stdlib-only base-R numeric oracle covering VAR coefficients,
  innovation covariance, information criteria, F statistics, and p-values.
- Synthetic confounder, known-order, dependent-uncertainty, error, cancellation,
  example, and benchmark coverage for the temporal APIs.

### Changed

- Package and README data-regime guidance now distinguishes i.i.d. row resampling
  from dependence-preserving block resampling.
- Temporal documentation explicitly separates predictive reduced-form VAR from
  contemporaneous structural identification and intervention claims.

### Reliability

- VAR regression designs reject integer overflow and exceedance of
  `DefaultMaxVARDesignCells`; reviewed callers can override the cap with
  `VAROptions`.
- Dependent bootstraps inherit `BootstrapOptions.MaxResamples`, deterministic
  seeds, failed-replicate diagnostics, and cooperative cancellation.

### Compatibility

- This release is additive. Existing `GrangerTest`, `Bootstrap`, and Gaussian
  effect APIs retain their behavior and i.i.d./pairwise statistical scope.

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
