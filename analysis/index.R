seed <- 2026-01-22

library(tidyverse)
library(knitr)
library(kableExtra)
library(ggpubr)
library(pbapply)
library(readxl)
library(ggforce)

lapply(list.files("R/", full.names = TRUE), source)

my_theme_set()

sessionInfo()

raw_data_files <-
  list.files("../data_registry/", full.names = TRUE, recursive = TRUE)

article_type_filter <-
  c(ma = "meta-analysis"
    , rct = "randomized controlled trial"
    , os = "observational study"
  )

search_log <-
  raw_data_files[str_detect(raw_data_files, "search_log\\.xlsx$")] |>
  read_xlsx()

llm <-
  c(gpt5 = "gpt-5"
    , claude45 = "claude-4.5-sonnet"
    , gemini3 = "gemini-3-pro"
    , mistral2 = "mistral-large-2"
    , llama4 = "llama-4"
    , deepseek3 = "deepseek-v3"
  )

reconciled_llm_names <-
  names(llm) |>
  sapply(
    \(x) ifelse(c(x, x) %in% "gemini3", c("gemini3_part1", "gemini3"), c(x, x))
  ) |>
  reduce(c) |>
  unique()

llm_as_judge <-
  c(gpt41mini = "gpt-4.1-mini"
    , claude45 = "claude-4.5-sonnet"
  )

llm_results3 <-
  intersect(
    paste0("../data_registry//phase3_", reconciled_llm_names, "/export.csv")
    , raw_data_files[str_detect(raw_data_files, "export\\.csv$")]
  )

llm_results3 <-
  llm_results3 |>
  `names<-`(
    unlist(str_extract_all(llm_results3, paste0(names(llm), collapse = "|")))
  ) |>
  lapply(read_csv, show_col_types = FALSE)

llm_results3 <-
  unique(names(llm_results3)) |>
  `names<-`(unique(names(llm_results3))) |>
  lapply(\(x) reduce(llm_results3[x], rbind))

llm_results2 <-
  intersect(
    paste0("../data_registry//phase2_doc_log__", llm, ".csv")
    , raw_data_files[str_detect(raw_data_files, "phase2_doc_log__")]
  )

llm_results2 <-
  llm_results2 |>
  `names<-`(
    data.frame(path = llm_results2) |>
      mutate(
        code =
          path |>
          sapply(\(x) str_extract_all(x, paste0(llm, collapse = "|"))[[1]])
      ) |>
      left_join(data.frame(code = llm, name = names(llm)), by = "code") |>
      pull(name)
  ) |>
  lapply(read_csv, show_col_types = FALSE)

judge1_llm_results <-
  intersect(
    paste0(
      "../data_registry//phase3_", reconciled_llm_names, "/judge_gpt41mini.csv"
    )
    , raw_data_files[str_detect(raw_data_files, "judge_gpt41mini\\.csv$")]
  )

judge1_llm_results <-
  judge1_llm_results |>
  `names<-`(
    unlist(
      str_extract_all(judge1_llm_results, paste0(names(llm), collapse = "|"))
    )
  ) |>
  lapply(read_csv, show_col_types = FALSE)

judge1_llm_results <-
  unique(names(judge1_llm_results)) |>
  `names<-`(unique(names(judge1_llm_results))) |>
  lapply(\(x) reduce(judge1_llm_results[x], rbind))

judge2_llm_results <-
  intersect(
      paste0(
        "../data_registry//phase3_", reconciled_llm_names, "/judge_claude45.csv"
      )
      , raw_data_files[str_detect(raw_data_files, "judge_claude45\\.csv$")]
    )

judge2_llm_results <-
  judge2_llm_results |>
  `names<-`(
    judge2_llm_results |>
      str_extract_all(paste0(paste0("phase3_", names(llm)), collapse = "|")) |>
      unlist() |>
      str_remove_all("phase3_")
  ) |>
  lapply(read_csv, show_col_types = FALSE)

judge2_llm_results <-
  unique(names(judge2_llm_results)) |>
  `names<-`(unique(names(judge2_llm_results))) |>
  lapply(\(x) reduce(judge2_llm_results[x], rbind))

eligible_main_pdf <-
  search_log |>
  filter(eligible == "yes") |>
  group_by(article_type_filter) |>
  mutate(row = seq(n())) |>
  ungroup() |>
  filter(row <= 10) |>
  select(row, article_type_filter, doi, rank, filename)

docs_per_llm <-
  llm_results3 |>
  lapply(
    select
    , filename = pdf_path
    , doc_sha256, , est_token = est_total_tokens_phase2
  ) |>
  lapply(unique)

docs_per_llm <-
  docs_per_llm |>
  lapply(\(x) inner_join(eligible_main_pdf, x, by = "filename")) |>
  imap(
    ~ .x |>
      left_join(
        llm_results2[[.y]] |>
          group_by(doc_sha256) |>
          arrange(doc_sha256, desc(timestamp_utc)) |>
          filter(seq(n()) == 1) |>
          select(doc_sha256, sec_per_doc)
        , by = "doc_sha256"
      )
  )

edges_per_llm <-
  llm_results3 |>
  lapply(
    select
    , filename = pdf_path
    , doc_sha256, edge_id, var1, var2, evidence = evidence_text, label
  )

edges_per_llm <-
  edges_per_llm |>
  lapply(
    \(x)
    eligible_main_pdf |>
      inner_join(x, by = "filename", relationship = "many-to-many")
  )

judge1_results_per_llm <-
  judge1_llm_results |>
  lapply(
    select
    , filename = pdf_path
    , doc_sha256, edge_id, var1, var2, evidence = evidence_text, label
  )

judge1_results_per_llm <-
  judge1_results_per_llm |>
  lapply(
    \(x)
    eligible_main_pdf |>
      inner_join(x, by = "filename", relationship = "many-to-many")
  )

judge2_results_per_llm <-
  judge2_llm_results |>
  lapply(
    select
    , filename = pdf_path
    , doc_sha256, edge_id, var1, var2, evidence = evidence_text, label
  )

judge2_results_per_llm <-
  judge2_results_per_llm |>
  lapply(
    \(x)
    eligible_main_pdf |>
      inner_join(x, by = "filename", relationship = "many-to-many")
  )

step_label <- read_csv("inst/extdata/step_label.csv", show_col_types = FALSE)

max_rank_to_eligible_10 <-
  search_log |>
  mutate_at("article_type_filter", \(x) factor(x, unique(x))) |>
  group_by(article_type_filter) |>
  filter(!is.na(filename)) |>
  filter(seq(n()) <= 10) |>
  summarize(max_rank = max(rank))

search_log_sum <-
  search_log |>
  left_join(max_rank_to_eligible_10, by = join_by(article_type_filter)) |>
  mutate_at("article_type_filter", str_to_sentence) |>
  mutate_at("article_type_filter", \(x) factor(x, unique(x))) |>
  group_by(article_type_filter) |>
  filter(rank <= max_rank) |>
  select(
    article_type_filter
    , vvr_exp_abstract
    , full_text_pub
    , vvr_exp_full
    , pdf_access
    , pdf_selectable
    , eligible
  ) |>
  gather(step, status, -article_type_filter) |>
  left_join(step_label, by = join_by(step)) |>
  mutate(step = label) |>
  select(-label) |>
  mutate_at("step", \(x) factor(x, unique(x))) |>
  mutate_at("status", factor, c("no", "yes")) |>
  group_by(article_type_filter, step, status) |>
  summarize(n = n(), .groups = "drop") |>
  spread(status, n, fill = 0)

search_log_sum <-
  search_log_sum |>
  mutate_at("article_type_filter", as.character) |>
  mutate(
    article_type_filter =
      ifelse(duplicated(article_type_filter), "", article_type_filter)
  ) |>
  `names<-`(
    colnames(search_log_sum) |>
      str_replace_all("_", " ") |>
      str_to_sentence()
  )

show_table(search_log_sum, "search_log_sum", "Search log summary")

strata_n <- length(article_type_filter) * length(llm)
ssize_per_strata <- 20
ssize <- strata_n * ssize_per_strata

strata <-
  expand.grid(
    article_type_filter
    , names(llm)
    , stringsAsFactors = FALSE
  )

## strata |>
##   filter(Var2 %in% names(edges_per_llm)) |>
##   pmap(
## 	  \(Var1, Var2)
## 	  edge_sampling(edges_per_llm, Var1, Var2, ssize_per_strata)
## 	)

## human_calibration_label <-
##   strata |>
##   pmap(
## 	  \(Var1, Var2)
## 	  read_human_calibration_sample_unlabeled(Var1, Var2, ssize_per_strata)
## 	) |>
##   reduce(rbind)
## 
## if(file.exists("inst/extdata/human_calibration_sample_300_edges_labeled.csv")){
##   human_calibration_label <-
##     human_calibration_label |>
##     select(-label) |>
##     left_join(
##       read_csv(
##         "inst/extdata/human_calibration_sample_300_edges_labeled.csv"
##         , show_col_types = FALSE
##       )
##       , by =
##         join_by(
##           row, article_type_filter, doi, rank, filename, doc_sha256,
##           edge_id, var1, var2, evidence
##         )
##     )
## }
## 
## human_calibration_label|>
##   write_csv("inst/extdata/human_calibration_sample_300_edges_unlabeled.csv")

human_calibration_label <-
  read_csv(
    "inst/extdata/human_calibration_sample_300_edges_labeled.csv"
    , show_col_types = FALSE
  )

judge1_human_per_llm <-
  judge1_results_per_llm |>
  lapply(rename, llm_as_judge = label) |>
  lapply(
    left_join
    , rename(human_calibration_label, human_as_judge = label)
    , by =
      join_by(
        row, article_type_filter, doi, rank, filename, doc_sha256
        , edge_id, var1, var2, evidence
      )
  )

judge2_human_per_llm <-
  judge2_results_per_llm |>
  lapply(rename, llm_as_judge = label) |>
  lapply(
    left_join
    , rename(human_calibration_label, human_as_judge = label)
    , by =
      join_by(
        row, article_type_filter, doi, rank, filename, doc_sha256
        , edge_id, var1, var2, evidence
      )
  )

calibration_data <-
  list(judge1 = judge1_human_per_llm, judge2 = judge2_human_per_llm) |>
  lapply(
    lapply
    , rename
    , label_llm = llm_as_judge
    , label_human = human_as_judge
  ) |>
  lapply(lapply, filter, !is.na(label_human)) |>
  lapply(\(x) imap(x, ~ mutate(.x, llm = .y))) |>
  lapply(reduce, rbind) |>
  imap(
    ~ .x |>
      mutate(judge = ifelse(.y == "judge1", llm_as_judge[1], llm_as_judge[2]))
  ) |>
  reduce(rbind) |>
  left_join(
    rownames_to_column(data.frame(code = llm), var = "llm"), by = "llm"
  ) |>
  mutate(llm = code) |>
  select(-code) |>
  mutate_at("llm", factor, llm) |> 
  mutate_at("judge", factor, llm_as_judge)

calibration_sum <-
  calibration_data |> 
  group_by(llm, judge, label_human, label_llm) |>
  summarize(n = n(), .groups = "drop")

calibration_sum <-
  calibration_sum |>
  `names<-`(
    colnames(calibration_sum) |>
      str_replace_all("_", " ") |>
      str_to_sentence() |>
      str_replace_all("Llm|llm", "LLM") |>
      str_replace_all("N", "n")
  )

show_table(calibration_sum, "calibration_sum", "Calibration summary.")

usd_per_token_per_llm <-
  read_csv("inst/extdata/usd_per_token_per_llm.csv", show_col_types = FALSE)

doc_edge_metrics <-
  calibration_data |>
  left_join(
    docs_per_llm |>
      imap(~ mutate(.x, llm = llm[names(llm) == .y])) |>
      reduce(rbind) |>
      left_join(usd_per_token_per_llm, by = join_by(llm))
    , by =
      join_by(row, article_type_filter, doi, rank, filename, doc_sha256, llm)
  ) |>
  mutate_at(c("article_type_filter", "doi", "llm"), \(x) factor(x, unique(x))) |>
  mutate_at("label_human", factor, c("C&G", "INC", "UNG")) |>
  group_by(
    article_type_filter, doi, llm
    , label_human, est_token, usd_per_token, sec_per_doc
  ) |>
  summarize(n = n(), .groups = "drop") |>
  spread(label_human, n, fill = 0) |>
  mutate(
    egrc = `C&G` / (`C&G` + INC) * 100
    , uer = UNG / (`C&G` + INC + UNG) * 100
    , ed = `C&G` + INC + UNG
    , est_token_edge = est_token / ed
    , sec_per_doc_edge = sec_per_doc / ed
  ) |>
  group_by(doi) |>
  mutate(med = mean(ed)) |>
  ungroup() |>
  mutate(red = ed / med)  |>
  mutate(
    cost_doc = est_token * usd_per_token
    , cost_edge = est_token_edge * usd_per_token
  ) |>
  select(
    llm, egrc, uer, red
    , runtime_doc = sec_per_doc, runtime_edge = sec_per_doc_edge
    , cost_doc, cost_edge
  ) 

doc_edge_metrics_sum <-
  doc_edge_metrics|>
  gather(metric, value, -llm) |>
  filter(!(is.nan(value) | is.na(value))) |>
  mutate_at("metric", \(x) factor(x, unique(x))) |>
  group_by(llm, metric) |>
  summarize(
    avg = mean(value)
    , lb = mean(value) - qnorm(0.975) * sd(value) / sqrt(n())
    , ub = mean(value) + qnorm(0.975) * sd(value) / sqrt(n())
    , .groups = "drop"
  ) |>
  mutate(
  	across(
  		c("avg", "lb", "ub")
  		, \(x)
  			case_when(
  				metric %in% c("egrc", "uer")
  				~ ifelse(x < 0, 0, ifelse(x > 100, 100, x))
  				, metric == "red" ~ x
  				, metric %in% c("cost_doc", "cost_edge")
  				  ~ ifelse(x < 0, 0, x)
  				, TRUE ~ x
  			)
  	)
  )

metric_label <-
  read_csv("inst/extdata/metric_label.csv", show_col_types = FALSE)

doc_edge_metrics_report <-
  doc_edge_metrics_sum |>
  mutate(
  	across(
  		c("avg", "lb", "ub")
  		, \(x)
  			case_when(
  				metric %in% c("egrc", "uer", "runtime_doc", "runtime_edge")
  				~ sprintf("%.0f", x)
  				, metric == "red" ~ sprintf("%.2f", x)
  				, metric %in% c("cost_doc", "cost_edge") ~ sprintf("%.3f", x)
  				, TRUE ~ sprintf("%.3f", x)
  			)
  	)
  ) |>
  mutate(estimate = paste0(avg, " (", lb, ", ", ub, ")")) |>
  select(-avg, -lb, -ub) |>
  spread(metric, estimate) |>
  column_to_rownames(var = "llm") |>
  t() |>
  as.data.frame() |>
  rownames_to_column(var = "metric") |>
  left_join(metric_label, by = join_by(metric)) |>
  mutate(metric = label) |>
  select(-label)

doc_edge_metrics_report <-
  doc_edge_metrics_report |>
  `names<-`(
    colnames(doc_edge_metrics_report) |>
      str_replace_all("metric", "Metric")
  )

doc_edge_metrics_report |>
  show_table("doc_edge_metrics_report", "Model benchmark.")

cost_performance_tradeoff <-
  doc_edge_metrics_sum |>
  filter(metric %in% c("egrc", "uer", "red")) |>
  mutate_at("metric", as.character) |>
  left_join(metric_label, by = join_by(metric)) |>
  mutate(metric = label) |>
  select(-label)  |>
  mutate_at("metric", str_wrap, 20) |>
  mutate_at("metric", \(x) factor(x, unique(x))) |>
  left_join(
    doc_edge_metrics_sum |>
      filter(metric == "cost_doc") |>
      select(-metric) |>
      rename_at(c("avg", "lb", "ub"), paste0, "_cost")
    , by = join_by(llm)
  ) |>
	mutate(
		a = (ub_cost - lb_cost) / 2
		, b = (ub - lb) / 2
		, c = lb_cost + a
		, d = lb + b
	) |>
  ggplot(aes(avg_cost, avg, color = llm)) +
  geom_ellipse(
		aes(x0 = c, y0 = d, a = a, b = b, angle = 0)
		, linewidth = 0.4
	) +
  geom_point(
    data =
      data.frame(
        avg_cost = c(0.01, 0.01, 0.01, 0.06, 0.06, 0.06)
        , avg = c(50, 0, 0.75, 100, 50, 1/0.75)
        , metric = rep(c("egrc", "uer", "red"), 2)
      ) |>
      left_join(metric_label, by = join_by(metric)) |>
      mutate(metric = label) |>
      select(-label)  |>
      mutate_at("metric", str_wrap, 20) |>
      mutate_at("metric", \(x) factor(x, unique(x)))
    , color = NA, na.rm = TRUE
  ) +
  geom_hline(
  	data =
  		data.frame(
  			avg = c(NA, NA, 1)
        , metric = c("egrc", "uer", "red")
  		) |>
  		left_join(metric_label, by = join_by(metric)) |>
  		mutate(metric = label) |>
  		select(-label)  |>
      mutate_at("metric", str_wrap, 20) |>
      mutate_at("metric", \(x) factor(x, unique(x)))
  	, aes(yintercept = avg)
  	, linetype = "dashed"
  	, linewidth = 0.4
  	, color = "grey40"
  	, na.rm = TRUE
  ) +
  geom_point() +
  facet_wrap(~ metric, scale = "free_y", ncol = 1) +
  xlab("Estimated cost") +
  ylab("Estimate (95% CI)") +
  guides(color = guide_legend(nrow = 3, byrow = TRUE)) +
  theme(
    legend.position = "bottom"
    , legend.title = element_blank()
  )

cost_performance_tradeoff
