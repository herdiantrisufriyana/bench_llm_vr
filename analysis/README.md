# Benchmarking LLMs for Evidence-Grounded Variable Relationship Extraction

## Vignette

[Progress report](https://herdiantrisufriyana.github.io/bench_llm_vr/index.html)

## System requirements

Install Docker desktop once in your machine. Start the service every time you build this project image or run the container.

## Installation guide

Change `bench_llm_vr_analysis` to the project image name.

Build the project image once for a new machine (currently support AMD64 and ARM64).

```{bash}
docker build -t bench_llm_vr_analysis analysis/
```

Run the container every time you start working on the project. Change left-side port numbers for either Rstudio or Jupyter lab if any of them is already used by other applications.

In terminal:

```{bash}
docker run -d \
  --name bench_llm_vr_analysis_container \
  -p 8787:8787 \
  -p 8888:8888 \
  -v "$(pwd)":/home/rstudio/project \
  bench_llm_vr_analysis
```

In command prompt:

```{bash}
docker run -d ^
  --name bench_llm_vr_analysis_container ^
  -p 8787:8787 ^
  -p 8888:8888 ^
  -v "%cd%":/home/rstudio/project ^
  bench_llm_vr_analysis
```

## Instructions for use

### Rstudio

Change port number in the link, accordingly, if it is already used by other applications.

Visit http://localhost:8787.
Username: rstudio
Password: 1234

Your working directory is ~/project.





