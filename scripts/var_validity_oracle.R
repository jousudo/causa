#!/usr/bin/env Rscript
# Independent numeric oracle for causa v0.16 VAR validity diagnostics.
#
# This script uses only base R and shares no implementation with the Go
# package. LAPACK computes the companion eigenvalues, pchisq calibrates the
# Portmanteau statistic, and p.adjust supplies the Holm and BH references.

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

var_design <- function(values, lags) {
  rows <- nrow(values) - lags
  variables <- ncol(values)
  design <- matrix(1, nrow = rows, ncol = 1 + variables * lags)
  for (row in seq_len(rows)) {
    t <- row + lags
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
  list(coefficients = coefficients, residuals = residuals)
}

companion_eigenvalues <- function(model, variables, lags) {
  matrices <- lapply(seq_len(lags), function(lag) {
    rows <- 2 + (lag - 1) * variables + seq_len(variables) - 1
    t(model$coefficients[rows, , drop = FALSE])
  })
  top <- do.call(cbind, matrices)
  if (lags == 1L) {
    companion <- top
  } else {
    lower <- cbind(diag(variables * (lags - 1L)),
                   matrix(0, nrow = variables * (lags - 1L), ncol = variables))
    companion <- rbind(top, lower)
  }
  eigen(companion, only.values = TRUE)$values
}

whiteness <- function(residuals, lags, max_lag, adjusted) {
  residuals <- scale(residuals, center = TRUE, scale = FALSE)
  observations <- nrow(residuals)
  variables <- ncol(residuals)
  covariance0 <- crossprod(residuals) / observations
  inverse0 <- solve(covariance0)
  statistic <- 0
  for (lag in seq_len(max_lag)) {
    covariance <- t(residuals[(lag + 1):observations, , drop = FALSE]) %*%
      residuals[1:(observations - lag), , drop = FALSE] / observations
    term <- sum(diag(t(covariance) %*% inverse0 %*% covariance %*% inverse0))
    statistic <- statistic + if (adjusted) term / (observations - lag) else term
  }
  statistic <- statistic * if (adjusted) observations^2 else observations
  degrees <- variables^2 * (max_lag - lags)
  c(statistic = statistic, p_value = pchisq(statistic, degrees, lower.tail = FALSE),
    degrees = degrees)
}

conditional_granger <- function(values, cause, effect, lags) {
  design <- var_design(values, lags)
  response <- values[(lags + 1):nrow(values), effect]
  unrestricted <- lm.fit(design, response)
  remove <- vapply(seq_len(lags), function(lag) {
    1 + (lag - 1) * ncol(values) + cause
  }, numeric(1))
  restricted <- lm.fit(design[, -remove, drop = FALSE], response)
  rss_u <- sum(unrestricted$residuals^2)
  rss_r <- sum(restricted$residuals^2)
  df1 <- lags
  df2 <- length(response) - ncol(design)
  statistic <- ((rss_r - rss_u) / df1) / (rss_u / df2)
  pf(statistic, df1, df2, lower.tail = FALSE)
}

print_vector <- function(label, values) {
  cat(label, " = []float64{", paste(sprintf("%.17g", values), collapse = ", "), "}\n", sep = "")
}

values <- make_data()
model <- fit_var(values, 2L)
roots <- companion_eigenvalues(model, 3L, 2L)
roots <- roots[order(-Mod(roots), -Re(roots), -Im(roots))]
p_values <- unlist(lapply(seq_len(3L), function(cause) {
  vapply(seq_len(3L), function(effect) {
    if (cause == effect) NA_real_ else conditional_granger(values, cause, effect, 2L)
  }, numeric(1))
}))
p_values <- p_values[!is.na(p_values)]

cat("// Golden values for var_validity_oracle_test.go.\n")
print_vector("eigen_real", Re(roots))
print_vector("eigen_imag", Im(roots))
print_vector("whiteness_adjusted", whiteness(model$residuals, 2L, 12L, TRUE))
print_vector("whiteness_unadjusted", whiteness(model$residuals, 2L, 12L, FALSE))
print_vector("scan_raw", p_values)
print_vector("scan_holm", p.adjust(p_values, method = "holm"))
print_vector("scan_bh", p.adjust(p_values, method = "BH"))
