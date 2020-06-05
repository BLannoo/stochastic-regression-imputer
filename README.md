# Description

This is a small project to demonstrate a data cleaning technique called "stochastic regression imputation"

# To execute

Run this in the root of the project
```bash
python3 -m venv .venv
. ./.venv/bin/activate
pip install -r requirements.txt
./check.sh
PYTHONPATH=$PYTHONPATH:. python src/scripts/impute_missing_data.py
```

# Data source

Data downloaded from: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
