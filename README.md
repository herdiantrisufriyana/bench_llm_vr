# Benchmarking LLMs for Evidence-Grounded Variable Relationship Extraction

This repository contains a reproducible benchmark for evaluating large language models (LLMs) on evidence‑grounded variable–variable relationship extraction from biomedical research articles.

The project is organized into two main parts:

- **Pipeline & benchmarking code** (root level):  
  Implements Phase 1–3 ingestion, extraction, and adjudication workflows used to generate evaluation data.

- **Data analysis & reporting** (`/analysis`):  
  A Dockerized RStudio-based analysis environment for calibration, statistical analysis, tables, and figures used in the EMBC 2026 paper.

## Getting started

All analysis, figures, and tables should be run from the **analysis environment**.

➡️ **See [`analysis/README.md`](analysis/README.md) for full setup and usage instructions.**

## Scope

- Phase 1–3 outputs under `data_registry/` are treated as immutable inputs.
- No LLM inference is performed during analysis.
- Results are reported at the article level and aggregated across models.

This README intentionally stays minimal.  
The analysis README is the authoritative entry point.
