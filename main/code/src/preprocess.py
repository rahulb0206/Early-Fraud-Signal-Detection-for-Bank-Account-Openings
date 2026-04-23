"""
Preprocessing pipeline for the bank account fraud detection project.

Loads Base.csv, engineers features, one-hot encodes categoricals, performs a
temporal train/test split (months 0-5 train, months 6-7 test), and MinMax-scales
three skewed columns fitted on training data only to prevent leakage.

Saves the fitted scaler and the four split CSVs to models/ and outputs/ so
downstream steps can load them without re-running this script.

Columns dropped before training (synthetic or uninterpretable values):
    velocity_6h, velocity_24h, velocity_4w  : synthetic perturbation caused
        inverted order (6h mean > 4w mean) with negative values. Not usable.
    velocity_ratio                          : derived from velocity columns, removed with them.
    zip_count_4w                            : up to 6,700 applications per ZIP in 4 weeks.
                                              Clearly synthetic, not real geography.
    bank_branch_count_8w                    : up to 2,385 branch visits in 8 weeks. Impossible.
    device_fraud_count                      : entire column is 0 across all rows. No signal.
    is_dirty_device                         : derived from device_fraud_count, also all 0.

Run from main/code/:
    python src/preprocess.py
"""

import os
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

SENTINEL_COLS    = ["prev_address_months_count", "current_address_months_count", "bank_months_count"]
CATEGORICAL_COLS = ["payment_type", "employment_status", "housing_status", "device_os", "source"]
SCALE_COLS       = ["days_since_request", "session_length_in_minutes", "intended_balcon_amount"]
TARGET           = "fraud_bool"

# columns dropped: synthetic perturbed values with no interpretable meaning
DROP_COLS = [
    "velocity_6h", "velocity_24h", "velocity_4w",
    "zip_count_4w", "bank_branch_count_8w", "device_fraud_count",
]


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

    def drop_uninterpretable(self, df):
        cols = [c for c in DROP_COLS if c in df.columns]
        df = df.drop(columns=cols)
        print(f"dropped {len(cols)} uninterpretable columns: {cols}")
        return df

    def engineer_features(self, df):
        df = df.copy()

        # -1 is the sentinel for "no prior address provided"; make it an explicit binary flag
        df["has_prev_address"]       = (df["prev_address_months_count"] != -1).astype(int)

        # credit limit requested relative to income (+1 avoids division by zero)
        df["credit_to_income_ratio"] = df["proposed_credit_limit"] / (df["income"] + 1)

        return df

    def encode_categoricals(self, df):
        df    = df.copy()
        cols  = [c for c in CATEGORICAL_COLS if c in df.columns]
        before = df.shape[1]
        df    = pd.get_dummies(df, columns=cols, drop_first=False, dtype=int)
        print(f"one-hot: {before} -> {df.shape[1]} cols")
        return df

    def temporal_split(self, df):
        # temporal split: preserves time order so test set is truly unseen future data
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
        df = self.drop_uninterpretable(df)
        df = self.engineer_features(df)
        df = self.encode_categoricals(df)

        # split before scaling: scaler is fitted on train only to prevent test leakage
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
