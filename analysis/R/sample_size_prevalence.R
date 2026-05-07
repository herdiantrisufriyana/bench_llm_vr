sample_size_prevalence <- function(p, e, N, z = 1.96) {
  n0 <- (z^2 * p * (1 - p)) / e^2
  n_adj <- n0 / (1 + (n0 - 1) / N)
  list(n0 = ceiling(n0), n_adj = ceiling(n_adj), p = p, e = e, N = N)
}
