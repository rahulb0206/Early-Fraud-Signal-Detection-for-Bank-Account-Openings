# Preprocessing pipeline
# Loads Base.csv, engineers a few features, one-hot encodes categoricals,
# does a temporal train/test split (months 0-5 train, 6-7 test), and
# MinMax-scales three skewed columns on train only to avoid leakage.
# Saves the fitted scaler and the four split CSVs so other notebooks
# can just load them directly without re-running this.
#
# run from main/code/: python src/preprocess.py

import os
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

SENTINEL_COLS    = ["prev_address_months_count", "current_address_months_count", "bank_months_count"]
CATEGORICAL_COLS = ["payment_type", "employment_status", "housing_status", "device_os", "source"]
SCALE_COLS       = ["days_since_request", "session_length_in_minutes", "intended_balcon_amount"]
TARGET           = "fraud_bool"


class FraudPreprocessor:

    def __init__(self):
        self.scaler = None

    def load(self, filepath):
        df = pd.read_csv(filepath)
        counts = df[TARGET].value_counts()
        print(f"loaded {df.shape[0]:,} rows x {df.shape[1]} cols  ({df.memory_usage(deep=True).sum()/1e6:.1f} MB)")
        print(f"fraud: {counts[1]:,} ({df[TARGET].mean()*100:.2f}%)   legit: {counts[0]:,}")
        return df

    def audit(self, df):
        nan_cols = df.isnull().sum()
        nan_cols = nan_cols[nan_cols > 0]
        if nan_cols.empty:
            print("no NaNs found")
        else:
            print("NaN counts:\n", nan_cols.to_string())

        sentinel_counts = {}
        for col in SENTINEL_COLS:
            n = (df[col] == -1).sum()
            sentinel_counts[col] = n
            print(f"  {col}: {n:,} sentinel -1s ({n/len(df)*100:.1f}%)")

        return {"nan_counts": nan_cols.to_dict(), "sentinel_counts": sentinel_counts}

    def engineer_features(self, df):
        df = df.copy()

        # -1 means the field wasn't provided, treat as a flag
        df["has_prev_address"]      = (df["prev_address_months_count"] != -1).astype(int)
        df["is_dirty_device"]       = (df["device_fraud_count"] > 0).astype(int)

        # short-window vs long-window velocity
        df["velocity_ratio"]        = df["velocity_6h"] / (df["velocity_4w"] + 1)
        df["credit_to_income_ratio"]= df["proposed_credit_limit"] / (df["income"] + 1)

        return df

    def encode_categoricals(self, df):
        df    = df.copy()
        cols  = [c for c in CATEGORICAL_COLS if c in df.columns]
        before = df.shape[1]
        df    = pd.get_dummies(df, columns=cols, drop_first=False, dtype=int)
        print(f"one-hot: {before} -> {df.shape[1]} cols")
        return df

    def temporal_split(self, df):
        # no random split — preserve time order so test is truly unseen future data
        train = df[df["month"] <= 5].drop(columns=["month"]).copy()
        test  = df[df["month"] >= 6].drop(columns=["month"]).copy()

        X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
        X_test,  y_test  = test.drop(columns=[TARGET]),  test[TARGET]

        print(f"train: {len(X_train):,} rows  ({y_train.mean()*100:.2f}% fraud)")
        print(f"test:  {len(X_test):,} rows   ({y_test.mean()*100:.2f}% fraud)")
        return X_train, X_test, y_train, y_test

    def scale(self, df, fit=True):
        df   = df.copy()
        cols = [c for c in SCALE_COLS if c in df.columns]

        if fit:
            self.scaler = MinMaxScaler()
            df[cols] = self.scaler.fit_transform(df[cols])
        else:
            if self.scaler is None:
                raise ValueError("call scale(fit=True) on train before transforming test")
            df[cols] = self.scaler.transform(df[cols])

        return df

    def run(self, filepath, save=True):
        df = self.load(filepath)
        self.audit(df)
        df = self.engineer_features(df)
        df = self.encode_categoricals(df)

        # split before scaling — scaler must only see training data
        X_train, X_test, y_train, y_test = self.temporal_split(df)
        X_train = self.scale(X_train, fit=True)
        X_test  = self.scale(X_test,  fit=False)

        if save:
            os.makedirs("models",  exist_ok=True)
            os.makedirs("outputs", exist_ok=True)
            joblib.dump(self.scaler, "models/scaler.pkl")
            X_train.to_csv("outputs/X_train.csv", index=False)
            X_test.to_csv("outputs/X_test.csv",   index=False)
            y_train.to_csv("outputs/y_train.csv", index=False)
            y_test.to_csv("outputs/y_test.csv",   index=False)
            print("saved scaler and splits to models/ and outputs/")

        print(f"\nX_train {X_train.shape}   X_test {X_test.shape}")
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    prep = FraudPreprocessor()
    X_train, X_test, y_train, y_test = prep.run("../Base.csv")
