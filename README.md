# From Atoms to Chains: Divergence-Guided Reasoning Curriculum for Unlabeled LLM Domain Adaptation

This repository contains the official implementation for the paper "From Atoms to Chains: Divergence-Guided Reasoning Curriculum for Unlabeled LLM Domain Adaptation".

## Introduction

We introduce **DGRC (Divergence-Guided Reasoning Curriculum)**, a pipeline designed to generate high-quality curricula for the unlabeled domain adaptation of Large Language Models (LLMs). The core idea of DGRC is to leverage the reasoning differences (i.e., "divergences") between a powerful teacher model and a smaller student model to automatically identify, verify, and curate critical reasoning steps.

The final outputs of this pipeline are two high-quality instruction-tuning curricula:
- **Atomic Knowledge Curriculum**: A collection of simple, verifiable "atomic" question-answer pairs that break down complex reasoning processes.
- **Verified CoT Curriculum**: A set of complex reasoning chains (CoTs) that have been cross-verified against the atomic knowledge to ensure their logical consistency and correctness.

## Requirements

This project is developed with Python 3.10. We highly recommend using a virtual environment manager like `conda` or `venv`.

The core dependencies are listed below. You can install them manually via `pip`:

```python
python==3.10.18
torch==2.7.1
transformers==4.55.0
vllm==0.10.0
flash-attn==2.8.1
flashinfer-python==0.2.10
httpx==0.23.3
```

## Configuration: Set API Keys

Before running the pipeline, you must configure the API credentials for the teacher model (e.g., OpenAI GPT-4.1).

### Model API Parameters (`src/`)
In the `src/` directory, you will find model configuration files. Please open `src/models_api.py` and `src/models.py` and enter your API key, endpoint URL, and any other required parameters.

```python
# Example in src/models_api.py
class Client_GPT4d1:
    def __init__(self):
        api_key = "YOUR_API_KEY_HERE"  # <--- Modify this
        azure_endpoint = "YOUR_API_ENDPOINT_HERE"  # <--- Modify this
        api_version = "YOUR_API_VERSION_HERE"  # <--- Modify this
        # ...
```

### API Key in Shell Scripts (`scripts/`)
The shell script for running the teacher model also requires the API key to be set as an environment variable. Please open `scripts/run_teacher_gpt41.sh` and add your API key.

```bash
# Example in scripts/run_teacher_gpt41.sh
export OPENAI_API_KEY="YOUR_API_KEY_HERE"  # <--- Modify this
```

## How to Run

The entire process is divided into two main stages: Sample Generation and Running the DGRC Pipeline.

### Step 1: Generate Samples from Teacher and Student Models
First, you need to generate multiple reasoning paths for your dataset using both the teacher and student models.

```bash
# Run the teacher model (e.g., GPT-4.1) to generate samples
bash scripts/run_teacher_gpt41.sh

# Run the student model (e.g., Qwen-7B) to generate samples
bash scripts/run_qwen.sh
```

### Step 2: Run the DGRC Curriculum Generation Pipeline
Once samples from both models have been generated, run the main pipeline script. This script will compare the reasoning paths, identify divergences, and produce the final curated curriculum datasets.

```bash
# This command will execute the full DGRC pipeline to produce the final curricula
bash scripts/run_DGRC_pipeline.sh
```

The final output files will be saved in the `data/` directory.

### Parameter Tuning
For more advanced customization, such as modifying file paths, model names, or specific parameters in the pipeline, you can edit the Python source code files directly within the `src/` directory.

## Project Structure

```
.
├── data/                 # For output data
├── DGRC_benchmark/       # For DGRC benchmark datasets
├── model/                # For model checkpoints
├── scripts/
│   ├── run_teacher_gpt41.sh  <-- Run this first
│   ├── run_qwen.sh           <-- Run this second
│   └── run_DGRC_pipeline.sh  <-- Run this last to generate the curriculum
├── src/
│   ├── models_api.py       <-- Add API endpoint/key here
│   ├── models.py           <-- Add model parameters/key here
│   ├── cot_verify.py       # Core pipeline logic
│   └── ...                 # Other pipeline scripts
└── README.md
```# DGRC
