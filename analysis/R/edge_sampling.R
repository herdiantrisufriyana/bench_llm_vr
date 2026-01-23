edge_sampling <- function(edge_df_list, strata1, strata2, size_n){
  set.seed((seed))
  edge_df_list[[strata2]] |>
    filter(article_type_filter == strata1) |>
    filter(seq(n()) %in% sample(seq(n()), size_n, FALSE)) |>
    write_csv(
      paste0(
        "inst/extdata/human_calibration_sample_"
        , strata1, "_", strata2, "_unlabeled.csv"
      )
    )
}