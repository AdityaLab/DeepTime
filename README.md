# The TimerBed Evaluation Suite and The VL-Time Method
<p align="center">
  <a href="https://arxiv.org/abs/2503.11835"><img src="https://img.shields.io/badge/arXiv-2503.11835-b31b1b.svg" alt="arXiv"></a>
</p>
This repository contains the implementation of TimerBed and the proposed VL-Time method for time series reasoning tasks.

## Folder Structure

- `/Dataset`: Contains datasets for 6 time reasoning tasks, unified into a classification task format. Due to size limitations, RCW and ECG datasets are not included in this repository but will be made available in the final version through alternative means.
- `/SupervisedModels`: Contains supervised time series models used for comparison.
- `/LLMs`: Contains LLM-based methods, including traditional numerical modeling and the proposed VL-Time method.

## Installation and Setup

### Step 1: Install Dependencies

Install the required libraries by running:

```bash
pip3 install -r ENV.txt
```

## Running Experiments

### Supervised Methods

1. Navigate to the SupervisedModels directory:
   ```bash
   cd /SupervisedModels
   ```

2. Run experiments using scripts in the `scripts` folder. For example:
   ```bash
   bash Transformer.sh
   ```
   - Use `--root_path` to specify the dataset
   - Use `--model` to specify the model

### LLM Methods

1. Navigate to the LLMs directory:
   ```bash
   cd /LLMs
   ```

2. Set up API key:
   - Add your API key in `/LLMs/Method/LMM.py`
   - Specifically, set the `openai_api_key`

3. Prepare visualization data:
   - Place the visualization data in the `LLMs/Dataset` folder
   - Currently, data for visual modeling is provided

4. Run experiments:
   ```bash
   bash Scripts.sh
   ```
   - Set `modal="L"` for traditional numerical modeling
   - Set `modal="V"` for VL-Time visual modeling
   - Use `--model` to specify different LLMs
   - Set `--num_shot_per_class=0` for zero-shot reasoning; use values >0 for few-shot setting
   - Use `--hint="Please solve this problem step by step"` for chain-of-thought reasoning


## Acknowledgement

This library is constructed based on the following repos:

https://github.com/thuml/Time-Series-Library/

https://github.com/stanfordmlgroup/ManyICL
