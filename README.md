# Note
This is a preview version for implementing DeMix. The detailed code, models for reproducing the results, and the DeMix Corpora are available only after passing compliance verification.

# DeMix Pipeline

## 1. Prepare Candidate Dataset
- Prepare and preprocess the candidate datasets for mixing.

## 2. Train Component Models
- Train a separate component model for each candidate dataset (or data source).

## 3. Sample Mixtures
- Generate candidate data-mixture samples:
  - Run `iterative_sample/sample.py`

## 4. Merge Component Models
- Generate the merge configuration YAML:
  - Run `model_merge/generate_merge_yaml.py`
- Merge component models:
  - Run `model_merge/merge_model.sh`

## 5. Benchmark Merged Models
- Evaluate merged models using **OpenCompass** or other benchmarking utilities.

## 6. Train Predictor & Iterate
- Train the predictor model:
  - Run `iterative_sample/train_predictor.py`
- Repeat from **Step 3** (Sample Mixtures) onward until convergence,
  and obtain the final optimal data mixture.


# DeMix Corpora
Coming soon after passing compliance verification.