# Mixture of Distribution (MoD)

This repository contains the codebase for the paper: **MoD: A Distribution-Based Approach for Merging Large Language Models**.

## Project Overview 

- **configs**: Configuration files for running the program and evaluation phases.
- **mod**: Source code for the `mod` package.
- **scripts**: Scripts to execute the main MoD program and evaluation phases.
- **tests**: Test cases to validate functionality before release.

## Setup

To set up the environment and run the program, follow the steps below:

### 1. Create a Virtual Environment

Create a Python virtual environment and install the required dependencies.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Hugging Face Token (Optional)

If certain resources from the Hugging Face Hub are required, configure your API token.

```bash
export HF_AUTH_TOKEN="YOUR_HF_TOKEN"
```

### 3. Run the Program

After setting up the environment and configurations, run the program with the provided script.

```bash
python ./scripts/mod.py --config ./configs/1.5B/mod_config.yml --out_path qwen2.5-1.5B-mod
```

## Evaluation

For evaluation, please refer to the instructions in the [mod-evaluate repository](https://github.com/knovel-eng/mod-evaluate).

## Testing

To ensure code integrity, create a pull request and run tests before pushing changes.

```bash
pip install pytest
pytest ./tests/
```
