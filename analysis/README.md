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





