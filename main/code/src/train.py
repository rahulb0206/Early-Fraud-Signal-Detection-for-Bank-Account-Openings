# Trains Random Forest (baseline) and XGBoost (primary) on the preprocessed splits.
# RF runs on raw imbalanced data — no tuning, just a benchmark.
# XGBoost runs twice: once with scale_pos_weight=89.7, once on SMOTE-oversampled data.
# Both XGB variants get a light grid search over key hyperparameters.
# All models saved to models/ as .pkl files.
#
# run from main/code/: python src/train.py
# expects outputs/X_train.csv etc. to already exist (run preprocess.py first)

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

SCALE_POS_WEIGHT = 89.7   # 988971 / 11029
N_JOBS = 2               # cap parallelism — -1 copies full dataset per core


class ModelTrainer:

    def __init__(self, data_dir="outputs", model_dir="models"):
        self.data_dir  = data_dir
        self.model_dir = model_dir
        self.X_train   = None
        self.X_test    = None
        self.y_train   = None
        self.y_test    = None

    def load_splits(self):
        self.X_train = pd.read_csv(f"{self.data_dir}/X_train.csv")
        self.X_test  = pd.read_csv(f"{self.data_dir}/X_test.csv")
        self.y_train = pd.read_csv(f"{self.data_dir}/y_train.csv").squeeze()
        self.y_test  = pd.read_csv(f"{self.data_dir}/y_test.csv").squeeze()
        print(f"loaded — train: {self.X_train.shape}  test: {self.X_test.shape}")
        print(f"train fraud rate: {self.y_train.mean()*100:.2f}%   test: {self.y_test.mean()*100:.2f}%")

    def _quick_eval(self, model, label):
        # threshold at 0.5 for a quick sanity check — proper eval is in evaluate.py
        probs = model.predict_proba(self.X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        recall    = recall_score(self.y_test, preds, zero_division=0)
        precision = precision_score(self.y_test, preds, zero_division=0)
        f1        = f1_score(self.y_test, preds, zero_division=0)
        roc_auc   = roc_auc_score(self.y_test, probs)
        print(f"\n[{label}]")
        print(f"  recall={recall:.3f}  precision={precision:.3f}  f1={f1:.3f}  roc-auc={roc_auc:.3f}")
        return {"recall": recall, "precision": precision, "f1": f1, "roc_auc": roc_auc}

    def train_baseline_rf(self):
        # no tuning, no imbalance handling — purely a naive benchmark
        print("\ntraining Random Forest baseline...")
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=N_JOBS
        )
        rf.fit(self.X_train, self.y_train)
        self._quick_eval(rf, "RF baseline")

        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(rf, f"{self.model_dir}/rf_baseline.pkl")
        print(f"saved -> {self.model_dir}/rf_baseline.pkl")
        return rf

    def train_xgb(self):
        # scale_pos_weight handles imbalance without touching the data
        print("\ntraining XGBoost (scale_pos_weight)...")

        param_grid = {
            "n_estimators":    [200, 400],
            "max_depth":       [4, 6],
            "learning_rate":   [0.05, 0.1],
            "subsample":       [0.8],
            "colsample_bytree":[0.8],
        }
        base_xgb = XGBClassifier(
            scale_pos_weight=SCALE_POS_WEIGHT,
            use_label_encoder=False,
            eval_metric="aucpr",
            tree_method="hist",
            random_state=42,
            n_jobs=N_JOBS,
            verbosity=0,
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid = GridSearchCV(
            base_xgb, param_grid,
            scoring="recall",
            cv=cv,
            n_jobs=1,
            verbose=1,
        )
        grid.fit(self.X_train, self.y_train)
        best_xgb = grid.best_estimator_
        print(f"best params: {grid.best_params_}")
        self._quick_eval(best_xgb, "XGBoost (scale_pos_weight)")

        joblib.dump(best_xgb, f"{self.model_dir}/xgb_model.pkl")
        print(f"saved -> {self.model_dir}/xgb_model.pkl")
        return best_xgb

    def train_xgb_smote(self):
        # SMOTE only on training data — test set stays untouched
        print("\napplying SMOTE to training set...")
        sm = SMOTE(random_state=42, sampling_strategy=0.1)
        X_res, y_res = sm.fit_resample(self.X_train, self.y_train)
        print(f"after SMOTE — fraud: {y_res.sum():,}  legit: {(y_res==0).sum():,}")

        print("training XGBoost (SMOTE)...")
        param_grid = {
            "n_estimators":  [200, 400],
            "max_depth":     [4, 6],
            "learning_rate": [0.05, 0.1],
        }
        base_xgb = XGBClassifier(
            use_label_encoder=False,
            eval_metric="aucpr",
            tree_method="hist",
            random_state=42,
            n_jobs=N_JOBS,
            verbosity=0,
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid = GridSearchCV(
            base_xgb, param_grid,
            scoring="recall",
            cv=cv,
            n_jobs=1,
            verbose=1,
        )
        grid.fit(X_res, y_res)
        del X_res, y_res          # free SMOTE data immediately
        best_xgb_smote = grid.best_estimator_
        print(f"best params: {grid.best_params_}")
        self._quick_eval(best_xgb_smote, "XGBoost (SMOTE)")

        joblib.dump(best_xgb_smote, f"{self.model_dir}/xgb_smote.pkl")
        print(f"saved -> {self.model_dir}/xgb_smote.pkl")
        return best_xgb_smote

    def run(self):
        self.load_splits()
        rf          = self.train_baseline_rf()
        xgb         = self.train_xgb()
        xgb_smote   = self.train_xgb_smote()

        print("\n--- training complete ---")
        print("models saved:")
        print(f"  {self.model_dir}/rf_baseline.pkl")
        print(f"  {self.model_dir}/xgb_model.pkl      <- primary model")
        print(f"  {self.model_dir}/xgb_smote.pkl")
        print("run evaluate.py to get full metrics at 5% FPR threshold")
        return rf, xgb, xgb_smote


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()
