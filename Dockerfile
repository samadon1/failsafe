# Container image for Failsafe experiments on Nebius Serverless Jobs.
# Each job runs one `failsafe run …` (see failsafe/executors/nebius.py) and uploads its artifact.
#
# Build + push to your Nebius container registry, then set NEBIUS_FAILSAFE_IMAGE to the pushed tag:
#   docker build -t cr.nebius.cloud/<registry>/failsafe:latest .
#   docker push  cr.nebius.cloud/<registry>/failsafe:latest
#
# On a GPU preset (gpu-l40s-a) the detector runs on CUDA, producing the GPU metrics that are
# UNAVAILABLE on the local M3. For Nemotron proposals inside a job, point FAILSAFE_LLM_* at Nebius
# Token Factory at build or run time (kept out of the image by default — pass at submit).
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3-pip git awscli libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY failsafe ./failsafe
COPY missions ./missions
COPY scenarios ./scenarios
RUN uv sync --frozen

# Nemotron → Nebius Token Factory (OpenAI-compatible). Supply the key at submit time, e.g.:
#   --env FAILSAFE_LLM_API_KEY=... --env FAILSAFE_LLM_MODEL=...
ENV FAILSAFE_LLM_BASE_URL=https://api.tokenfactory.nebius.com/v1

ENTRYPOINT ["uv", "run", "failsafe"]
