# Models

Model `.pkl` files are excluded from this repository (the Random Forest alone is 214MB).

To reproduce all three models, run the pipeline from `main/code/`:

```bash
python src/preprocess.py   # generates outputs/X_train.csv etc. and models/scaler.pkl
python src/train.py        # trains all three models and saves .pkl files here
python src/evaluate.py     # computes full metrics and saves plots to outputs/figures/
python src/explain.py      # computes SHAP values and saves plots to outputs/figures/
```

You will need `Base.csv` in `main/` (available from the [Bank Account Fraud dataset on Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)).

## Files produced here

| File | Description |
|------|-------------|
| `scaler.pkl` | MinMaxScaler fitted on training data only (days_since_request, session_length_in_minutes, intended_balcon_amount) |
| `rf_baseline.pkl` | Random Forest, 100 trees, no imbalance handling. Lower bound benchmark. |
| `xgb_model.pkl` | XGBoost with scale_pos_weight=89.7. Primary model. |
| `xgb_smote.pkl` | XGBoost trained on SMOTE-oversampled data (10:1 ratio). |
