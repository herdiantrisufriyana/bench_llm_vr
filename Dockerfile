FROM python:3.11-slim

# -------------------------
# System deps
# -------------------------
RUN apt-get update && apt-get install -y \
	build-essential \
	git \
	curl \
	&& rm -rf /var/lib/apt/lists/*

# -------------------------
# R base and system deps
# -------------------------
RUN apt-get update && apt-get install -y \
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Workdir
# -------------------------
WORKDIR /app

# -------------------------
# Copy benchmark repo
# -------------------------
COPY . /app

# -------------------------
# Install Python deps
# -------------------------
# 1) Hypathesis requirements (submodule)
RUN pip install --no-cache-dir -r apps/hypathesis/requirements.txt

# 2) Benchmark-only deps (if any later)
# (safe even if empty)
RUN pip install --no-cache-dir pandas numpy scipy scikit-learn openpyxl

# -------------------------
# Environment defaults
# -------------------------
ENV PYTHONPATH=/app/apps/hypathesis:/app
ENV CHROMA_DIR=/chroma
ENV FILES_DIR=/files
ENV RUN_ID=default

# -------------------------
# Create mount points
# -------------------------
RUN mkdir -p /chroma /files

# -------------------------
# R packages for analysis
# -------------------------
RUN R -e "install.packages(c( \
    'tidyverse', \
    'data.table', \
    'lme4', \
    'emmeans', \
    'broom', \
    'knitr', \
    'rmarkdown' \
), repos='https://cloud.r-project.org')"

# -------------------------
# Default command (override per run)
# -------------------------
CMD ["bash"]