# SHAP explainability for the fraud detection model.
#
# Uses TreeExplainer (exact, not approximate) on the primary XGBoost model.
# Generates:
#   - shap_summary_bar.png       : global mean |SHAP| per feature (top 20)
#   - shap_summary_beeswarm.png  : direction + magnitude of each feature's impact
#   - shap_waterfall_tp.png      : why the model correctly caught a fraud case
#   - shap_waterfall_fn.png      : why the model missed a fraud case
#   - shap_waterfall_fp.png      : why the model wrongly flagged a legit case
#   - shap_values.csv            : full SHAP matrix for all test rows (used by Streamlit app)
#
# Key purpose: gain-based feature importance (from train.py) is biased toward
# binary/high-cardinality features and features used in early splits. SHAP
# corrects for this by attributing each feature's contribution accounting for
# all possible feature orderings — giving a fairer global ranking and honest
# per-prediction explanations.
#
# run from main/code/: python src/explain.py

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

MODEL_PATH  = "models/xgb_model.pkl"
X_TEST_PATH = "outputs/X_test.csv"
Y_TEST_PATH = "outputs/y_test.csv"
FIG_DIR     = "outputs/figures"
OUT_DIR     = "outputs"

# how many rows to compute SHAP on — full 205k is slow, 20k is representative
SHAP_SAMPLE = 20_000
RANDOM_SEED = 42


class SHAPExplainer:

    def __init__(self):
        self.model      = None
        self.X_test     = None
        self.y_test     = None
        self.shap_vals  = None   # shape (n_samples, n_features)
        self.base_value = None   # expected model output (log-odds)
        self.X_sample   = None   # the subset used for global plots

    def load(self):
        self.model  = joblib.load(MODEL_PATH)
        self.X_test = pd.read_csv(X_TEST_PATH)
        self.y_test = pd.read_csv(Y_TEST_PATH).squeeze()
        print(f"loaded model: {MODEL_PATH}")
        print(f"test set    : {self.X_test.shape[0]:,} rows x {self.X_test.shape[1]} features")
        print(f"fraud cases : {self.y_test.sum():,} ({self.y_test.mean()*100:.2f}%)")

    def _xgb_shap(self, X):
        # use XGBoost's native SHAP computation — avoids shap/xgboost version conflicts
        # pred_contribs=True returns shape (n, n_features+1) where last col is bias
        dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
        contribs = self.model.get_booster().predict(dmatrix, pred_contribs=True)
        shap_values = contribs[:, :-1]   # drop last col (bias term)
        bias        = contribs[0, -1]    # same for all rows
        return shap_values, bias

    def compute_shap(self):
        # sample for global plots — stratified to preserve fraud rate
        rng = np.random.default_rng(RANDOM_SEED)
        fraud_idx = self.y_test[self.y_test == 1].index.tolist()
        legit_idx = self.y_test[self.y_test == 0].index.tolist()

        # keep all fraud in sample, fill rest with legit
        n_legit = min(SHAP_SAMPLE - len(fraud_idx), len(legit_idx))
        sampled_legit = rng.choice(legit_idx, size=n_legit, replace=False).tolist()
        sample_idx = fraud_idx + sampled_legit

        self.X_sample = self.X_test.loc[sample_idx].reset_index(drop=True)
        y_sample      = self.y_test.loc[sample_idx].reset_index(drop=True)

        print(f"\ncomputing SHAP on {len(self.X_sample):,} rows "
              f"({y_sample.sum():,} fraud, {(y_sample==0).sum():,} legit)...")
        self.shap_vals, self.base_value = self._xgb_shap(self.X_sample)
        print(f"done. base value (log-odds): {self.base_value:.4f}")

        # also compute full test set SHAP values for the CSV export
        print(f"computing full SHAP values for all {len(self.X_test):,} test rows (for app)...")
        full_shap, _ = self._xgb_shap(self.X_test)
        shap_df = pd.DataFrame(full_shap, columns=self.X_test.columns)
        shap_df.to_csv(f"{OUT_DIR}/shap_values.csv", index=False)
        print(f"saved -> {OUT_DIR}/shap_values.csv")

    def plot_summary_bar(self):
        # mean absolute SHAP value per feature — honest global importance
        # compare this with the gain-based chart from evaluate.py to see the difference
        os.makedirs(FIG_DIR, exist_ok=True)
        mean_abs_shap = np.abs(self.shap_vals).mean(axis=0)
        feature_names = self.X_sample.columns.tolist()
        order = np.argsort(mean_abs_shap)[::-1][:20]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(
            [feature_names[i] for i in order[::-1]],
            mean_abs_shap[order[::-1]],
            color="steelblue"
        )
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title("Global Feature Importance — Mean |SHAP Value| (Top 20)\n"
                     "SHAP corrects for gain-bias: continuous features no longer undervalued",
                     fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/shap_summary_bar.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"saved -> {FIG_DIR}/shap_summary_bar.png")

    def plot_summary_beeswarm(self):
        # beeswarm via shap — each dot = one sample, colour = feature value
        # shows both magnitude AND direction of each feature's impact on fraud probability
        os.makedirs(FIG_DIR, exist_ok=True)
        shap.summary_plot(
            self.shap_vals,
            self.X_sample,
            plot_type="dot",
            max_display=20,
            show=False,
        )
        plt.title("Feature Impact Direction — SHAP Beeswarm (Top 20)\n"
                  "Red = high feature value, Blue = low. Right = pushes toward fraud.",
                  fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"saved -> {FIG_DIR}/shap_summary_beeswarm.png")

    def _waterfall(self, row_idx, label, filename):
        # generates a waterfall plot explaining one specific prediction
        row_shap, base = self._xgb_shap(self.X_test.iloc[[row_idx]])
        prob = self.model.predict_proba(self.X_test.iloc[[row_idx]])[0, 1]

        explanation = shap.Explanation(
            values        = row_shap[0],
            base_values   = base,
            data          = self.X_test.iloc[row_idx].values,
            feature_names = self.X_test.columns.tolist(),
        )

        os.makedirs(FIG_DIR, exist_ok=True)
        shap.waterfall_plot(explanation, max_display=15, show=False)
        plt.title(f"{label}\nFraud probability: {prob*100:.1f}%", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/{filename}", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"saved -> {FIG_DIR}/{filename}  (fraud_prob={prob*100:.1f}%)")

    def plot_waterfalls(self):
        # need predicted probabilities + labels to find TP, FN, FP cases
        probs = self.model.predict_proba(self.X_test)[:, 1]

        # use same 5% FPR threshold logic as evaluate.py
        from sklearn.metrics import roc_curve
        fpr_arr, tpr_arr, thresholds = roc_curve(self.y_test, probs)
        idx = np.searchsorted(fpr_arr, 0.05, side="right") - 1
        idx = max(0, min(idx, len(thresholds) - 1))
        threshold = thresholds[idx]
        preds = (probs >= threshold).astype(int)
        print(f"\nusing threshold={threshold:.4f} (matches 5% FPR from evaluate.py)")

        tp_idx = np.where((self.y_test.values == 1) & (preds == 1))[0]
        fn_idx = np.where((self.y_test.values == 1) & (preds == 0))[0]
        fp_idx = np.where((self.y_test.values == 0) & (preds == 1))[0]

        # pick the most confident case in each group for a clean story
        tp_row = tp_idx[np.argmax(probs[tp_idx])]    # highest fraud score among caught fraud
        fn_row = fn_idx[np.argmax(probs[fn_idx])]    # highest fraud score among missed fraud
        fp_row = fp_idx[np.argmax(probs[fp_idx])]    # highest fraud score among wrong flags

        print(f"\ngenerating waterfall plots...")
        self._waterfall(tp_row, "True Positive — Correctly Caught Fraud", "shap_waterfall_tp.png")
        self._waterfall(fn_row, "False Negative — Fraud the Model Missed", "shap_waterfall_fn.png")
        self._waterfall(fp_row, "False Positive — Legitimate Application Wrongly Flagged", "shap_waterfall_fp.png")

    def run(self):
        self.load()
        self.compute_shap()
        self.plot_summary_bar()
        self.plot_summary_beeswarm()
        self.plot_waterfalls()
        print("\nall done — SHAP outputs saved to outputs/figures/ and outputs/")
        print("\nkey check: compare shap_summary_bar.png with the gain-based feature_importance.png")
        print("if credit_risk_score and name_email_similarity rank higher in SHAP,")
        print("that confirms gain-bias was suppressing continuous features in the original chart.")


if __name__ == "__main__":
    exp = SHAPExplainer()
    exp.run()
