build_index <- function(input = "analysis/index.Rmd") {
  stopifnot(file.exists(input))

  name <- tools::file_path_sans_ext(basename(input))

  ## Render HTML to repo root (for GitHub Pages)
  rmarkdown::render(
    input = input,
    output_file = file.path(getwd(), paste0(name, ".html")),
    output_dir = getwd()
  )

  ## Extract R script into analysis/ only
  knitr::purl(
    input = input,
    output = file.path("analysis", paste0(name, ".R")),
    documentation = 0
  )

  invisible(TRUE)
}