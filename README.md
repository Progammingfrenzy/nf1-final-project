# NF1 Familial vs Sporadic Case Classification

## Project Summary
This project studies whether machine learning can distinguish familial and sporadic neurofibromatosis type 1 (NF1) cases using clinical symptom data.

The project compares three classical baseline models:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

It also includes one recent tabular machine learning method:
- TabPFN

In addition to model comparison, the project includes biological interpretation by:
- comparing symptom frequencies between familial and sporadic cases
- ranking features with ANOVA
- analyzing permutation importance for the best model

## Files Included
- `nf1_baselines.py` – main Python script
- `requirements.txt` – Python package dependencies
- `dataset_link.txt` – dataset and source references
- `outputs/` – generated CSV result files
- `slides/` – final presentation files
- `report/` – final written report

## Dataset
Source dataset:
- UCI Neurofibromatosis Type 1 clinical symptoms dataset

The target column is:
- `Case Type`
  - `1 = Familial`
  - `0 = Sporadic`

## How to Run
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt