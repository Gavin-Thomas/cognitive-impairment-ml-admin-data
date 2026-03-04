# Machine Learning Classification of MCI and Dementia Using Administrative Health Data

Classification of mild cognitive impairment (MCI) and dementia using 10 machine learning algorithms trained on linked Canadian administrative health data (physician claims, NACRS, DAD, prescriptions) with specialist-confirmed diagnoses from the PROMPT registry as the reference standard.

## Data Availability

The source dataset is not included due to privacy restrictions on linked health records. The pipeline expects administrative health code counts, prescription records, and a `cognitive_status` target variable.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Primary analysis (main ML pipeline — runtime: 2-6 hours)
python scripts/run_primary_analysis.py

# Secondary analysis (adds age/sex as predictors)
python scripts/run_secondary_analysis.py

# Comparison of primary vs secondary
python scripts/run_comparison.py

# Post-hoc misclassification demographics
python scripts/run_posthoc_misclassification.py
```

## Project Structure

```
scripts/          Runnable entry-point scripts
utils/            Shared modules (visualizations, statistical tests, epidemiology)
trained_models/           Generated results, figures, and trained models
```

## Key Methods

- 70/30 stratified train/test split; SMOTE within CV folds
- GridSearchCV (5-fold) for hyperparameter tuning
- BCa bootstrap confidence intervals (500 iterations)
- Permutation importance and SHAP for interpretability
- TRIPOD+AI reporting compliance
