read_human_calibration_sample_unlabeled <- function(strata1, strata2, size_n){
  filename <-
    paste0(
      "human_calibration_sample_"
      , strata1, "_", strata2, "_unlabeled.csv"
    )
  
  filepath <-
    list.files("inst/extdata", full.names = TRUE, pattern = filename)
  
  n <- size_n
  
  if(length(filepath) == 0){
    data.frame(
      row = integer(n)
      , article_type_filter = character(n)
      , doi = character(n)
      , rank = numeric(n)
      , filename = character(n)
      , doc_sha256 = character(n)
      , edge_id = character(n)
      , var1 = character(n)
      , var2 = character(n)
      , evidence = character(n)
      , label = character(n)
      , stringsAsFactors = FALSE
    )
  }else{
    read_csv(filepath, show_col_types = FALSE)
  }
}