cohens_kappa <- function(human, llm, levels) {
  tbl <- table(factor(human, levels), factor(llm, levels))
  n <- sum(tbl)
  p_o <- sum(diag(tbl)) / n
  p_e <- sum(rowSums(tbl) * colSums(tbl)) / n^2
  kappa <- (p_o - p_e) / (1 - p_e)
  se <- sqrt(p_e / (n * (1 - p_e)))
  list(
    kappa = kappa, p_o = p_o, p_e = p_e, se = se,
    ci_lo = kappa - 1.96 * se, ci_hi = kappa + 1.96 * se, n = n
  )
}
