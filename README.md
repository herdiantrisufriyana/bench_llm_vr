# Benchmarking LLMs for Evidence-Grounded Variable Relationship Extraction

**Authors**  
Herdiantri Sufriyana<sup>1</sup>, Emily Chia-Yu Su<sup>1,2,3</sup>  

**Affiliations**  
<sup>1</sup> Institute of Biomedical Informatics, College of Medicine, National Yang Ming Chiao Tung University, Taipei 112, Taiwan  
<sup>2</sup> Graduate Institute of Biomedical Informatics, College of Medical Science and Technology, Taipei Medical University, Taipei 110, Taiwan  
<sup>3</sup> Clinical Big Data Research Center, Taipei Medical University Hospital, Taipei 110, Taiwan  

**Emails**  
herdi@nycu.edu.tw (Herdiantri Sufriyana)  
emilysu@nycu.edu.tw (Emily Chia-Yu Su)

## Overview

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
