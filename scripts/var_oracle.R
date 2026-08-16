#!/usr/bin/env Rscript
# Independent numeric oracle for causa.FitVAR and causa.VARGrangerTest.
#
# This script uses only base R and shares no implementation with the Go
# package. Each equation is fitted by R's LINPACK/LAPACK-backed lm.fit, while
# causa uses a stdlib-only Householder QR solver. It emits the quantities locked
# by var_oracle_test.go: intercepts, coefficient matrices, innovation
# covariance, information criteria, and a conditional nested-model F-test.

make_data <- function() {
  variables <- 3L
  burn <- 40L
  keep <- 120L
  total <- burn + keep
  values <- matrix(0, nrow = total, ncol = variables)
  state <- c(13, 29, 47)

  innovation <- function(channel) {
    state[channel] <<- (97 * state[channel] + 37) %% 9973
    (state[channel] / 9973) - 0.5
  }

  for (t in 3:total) {
    values[t, 1] <- 0.55 * values[t - 1, 1] - 0.20 * values[t - 2, 1] + innovation(1)
    values[t, 2] <- 0.15 * values[t - 1, 1] + 0.40 * values[t - 1, 2] +
      0.05 * values[t - 1, 3] - 0.10 * values[t - 2, 2] + innovation(2)
    values[t, 3] <- -0.25 * values[t - 1, 1] + 0.35 * values[t - 1, 2] +
      0.30 * values[t - 1, 3] + 0.10 * values[t - 2, 1] + innovation(3)
  }
  values[(burn + 1):total, , drop = FALSE]
}

var_design <- function(values, lags, holdback = lags) {
  n <- nrow(values)
  variables <- ncol(values)
  rows <- n - holdback
  design <- matrix(1, nrow = rows, ncol = 1 + variables * lags)
  for (row in seq_len(rows)) {
    t <- row + holdback
    for (lag in seq_len(lags)) {
      cols <- 2 + (lag - 1) * variables + seq_len(variables) - 1
      design[row, cols] <- values[t - lag, ]
    }
  }
  design
}

fit_var <- function(values, lags) {
  design <- var_design(values, lags)
  response <- values[(lags + 1):nrow(values), , drop = FALSE]
  coefficients <- matrix(0, nrow = ncol(design), ncol = ncol(values))
  residuals <- matrix(0, nrow = nrow(design), ncol = ncol(values))
  for (equation in seq_len(ncol(values))) {
    fit <- lm.fit(design, response[, equation])
    coefficients[, equation] <- fit$coefficients
    residuals[, equation] <- fit$residuals
  }
  sigma <- crossprod(residuals) / nrow(residuals)
  logdet <- as.numeric(determinant(sigma, logarithm = TRUE)$modulus)
  parameters <- ncol(values) * ncol(design)
  observations <- nrow(design)
  criteria <- c(
    AIC = logdet + 2 * parameters / observations,
    BIC = logdet + log(observations) * parameters / observations,
    HQIC = logdet + 2 * log(log(observations)) * parameters / observations
  )
  list(coefficients = coefficients, residuals = residuals, sigma = sigma,
       logdet = logdet, criteria = criteria)
}

conditional_granger <- function(values, cause, effect, lags) {
  # cause/effect are one-based in this R script.
  design <- var_design(values, lags)
  response <- values[(lags + 1):nrow(values), effect]
  unrestricted <- lm.fit(design, response)
  remove <- unlist(lapply(seq_len(lags), function(lag) 1 + (lag - 1) * ncol(values) + cause))
  restricted <- lm.fit(design[, -remove, drop = FALSE], response)
  rss_u <- sum(unrestricted$residuals^2)
  rss_r <- sum(restricted$residuals^2)
  df1 <- lags
  df2 <- length(response) - ncol(design)
  statistic <- ((rss_r - rss_u) / df1) / (rss_u / df2)
  c(F = statistic, PValue = pf(statistic, df1, df2, lower.tail = FALSE),
    RSSRestricted = rss_r, RSSUnrestricted = rss_u,
    DFNumerator = df1, DFDenominator = df2)
}

print_vector <- function(label, values) {
  cat(label, " = []float64{", paste(sprintf("%.17g", values), collapse = ", "), "}\n", sep = "")
}

values <- make_data()
model <- fit_var(values, 2L)
cat("// Golden values for var_oracle_test.go.\n")
print_vector("intercept", model$coefficients[1, ])
for (lag in 1:2) {
  for (equation in 1:3) {
    rows <- 2 + (lag - 1) * 3 + 0:2
    print_vector(sprintf("A%d[%d]", lag, equation - 1), model$coefficients[rows, equation])
  }
}
for (row in 1:3) {
  print_vector(sprintf("sigma[%d]", row - 1), model$sigma[row, ])
}
print_vector("criteria", model$criteria)
print_vector("granger_0_to_1", conditional_granger(values, 1L, 2L, 2L))
