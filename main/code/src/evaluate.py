# Evaluation module for the fraud detection models.
#
# Loads trained models and test splits, computes all metrics at a fixed 5% FPR
# threshold (not 0.5 — default threshold is meaningless for imbalanced fraud data).
# Metrics: Recall, Precision, F1, PR-AUC, ROC-AUC, Gini, KS statistic.
# Also runs a failure mode analysis — breaks down what kinds of applications
# the model misses (false negatives) and wrongly flags (false positives).
# Temporal drift check compares performance on month 6 vs month 7 separately.
#
# run from main/code/: python src/evaluate.py

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)

TARGET    = "fraud_bool"
FPR_CAP   = 0.05   # evaluate everything at this fixed false positive rate


class EvaluationEngine:

    def __init__(self, model_dir="models", data_dir="outputs", fig_dir="outputs/figures"):
        self.model_dir = model_dir
        self.data_dir  = data_dir
        self.fig_dir   = fig_dir
        self.X_test    = None
        self.y_test    = None

    def load(self):
        self.X_test = pd.read_csv(f"{self.data_dir}/X_test.csv")
        self.y_test = pd.read_csv(f"{self.data_dir}/y_test.csv").squeeze()
        print(f"test set: {self.X_test.shape[0]:,} rows  fraud: {self.y_test.sum():,} ({self.y_test.mean()*100:.2f}%)")

    def _threshold_at_fpr(self, y_true, probs, target_fpr=FPR_CAP):
        # find the score cutoff where FPR <= target_fpr
        fpr, tpr, thresholds = roc_curve(y_true, probs)
        idx = np.searchsorted(fpr, target_fpr, side="right") - 1
        idx = max(0, min(idx, len(thresholds) - 1))
        return thresholds[idx], fpr[idx], tpr[idx]

    def evaluate_model(self, model, label):
        probs = model.predict_proba(self.X_test)[:, 1]
        threshold, actual_fpr, actual_tpr = self._threshold_at_fpr(self.y_test, probs)
        preds = (probs >= threshold).astype(int)

        recall    = recall_score(self.y_test, preds, zero_division=0)
        precision = precision_score(self.y_test, preds, zero_division=0)
        f1        = f1_score(self.y_test, preds, zero_division=0)
        roc_auc   = roc_auc_score(self.y_test, probs)
        pr_auc    = average_precision_score(self.y_test, probs)
        gini      = 2 * roc_auc - 1
        cm        = confusion_matrix(self.y_test, preds)

        # KS statistic — separation between fraud and legit score distributions
        fraud_scores  = probs[self.y_test == 1]
        legit_scores  = probs[self.y_test == 0]
        ks_stat, _    = ks_2samp(fraud_scores, legit_scores)

        print(f"\n{'='*45}")
        print(f"  {label}")
        print(f"{'='*45}")
        print(f"  threshold @ {FPR_CAP*100:.0f}% FPR : {threshold:.4f}")
        print(f"  actual FPR              : {actual_fpr*100:.2f}%")
        print(f"  recall                  : {recall:.3f}")
        print(f"  precision               : {precision:.3f}")
        print(f"  f1                      : {f1:.3f}")
        print(f"  roc-auc                 : {roc_auc:.3f}")
        print(f"  pr-auc                  : {pr_auc:.3f}")
        print(f"  gini                    : {gini:.3f}")
        print(f"  ks statistic            : {ks_stat:.3f}")
        print(f"  confusion matrix:")
        print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
        print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")

        return {
            "label": label, "threshold": threshold, "recall": recall,
            "precision": precision, "f1": f1, "roc_auc": roc_auc,
            "pr_auc": pr_auc, "gini": gini, "ks": ks_stat,
            "probs": probs, "preds": preds,
        }

    def failure_mode_analysis(self, results, model_label="XGBoost"):
        # looks at what the primary model gets wrong and tries to find patterns
        res   = next(r for r in results if model_label in r["label"])
        preds = res["preds"]
        probs = res["probs"]

        fn_mask = (self.y_test == 1) & (preds == 0)   # fraud missed
        fp_mask = (self.y_test == 0) & (preds == 1)   # legit wrongly flagged
        tp_mask = (self.y_test == 1) & (preds == 1)   # fraud caught

        fn_df = self.X_test[fn_mask].copy()
        fp_df = self.X_test[fp_mask].copy()
        tp_df = self.X_test[tp_mask].copy()

        print(f"\n--- failure mode analysis ({model_label}) ---")
        print(f"false negatives (fraud missed) : {fn_mask.sum():,}")
        print(f"false positives (legit flagged): {fp_mask.sum():,}")
        print(f"true positives  (fraud caught) : {tp_mask.sum():,}")

        # compare key features between caught vs missed fraud
        check_cols = [c for c in ["velocity_6h", "velocity_ratio", "credit_risk_score",
                                   "is_dirty_device", "has_prev_address",
                                   "credit_to_income_ratio"] if c in self.X_test.columns]
        if check_cols:
            print("\n  avg feature values — caught fraud vs missed fraud:")
            print(f"  {'feature':<30} {'caught':>10} {'missed':>10}")
            for col in check_cols:
                caught_mean = tp_df[col].mean() if len(tp_df) else float("nan")
                missed_mean = fn_df[col].mean() if len(fn_df) else float("nan")
                print(f"  {col:<30} {caught_mean:>10.3f} {missed_mean:>10.3f}")

        # save false negative and false positive samples for manual inspection
        os.makedirs(self.data_dir, exist_ok=True)
        fn_df["fraud_prob"] = probs[fn_mask]
        fp_df["fraud_prob"] = probs[fp_mask]
        fn_df.to_csv(f"{self.data_dir}/false_negatives.csv", index=False)
        fp_df.to_csv(f"{self.data_dir}/false_positives.csv", index=False)
        print(f"\n  saved false_negatives.csv and false_positives.csv to {self.data_dir}/")

    def temporal_drift(self, base_csv_path="../Base.csv", model_label="XGBoost"):
        # re-loads raw data to recover month labels, then evaluates month 6 vs 7 separately
        # month is dropped during preprocessing so we need to go back to the source
        print(f"\n--- temporal drift check ({model_label}) ---")
        try:
            raw = pd.read_csv(base_csv_path, usecols=["month", TARGET])
            test_raw = raw[raw["month"] >= 6].reset_index(drop=True)
        except Exception as e:
            print(f"  could not load {base_csv_path}: {e}")
            return

        if len(test_raw) != len(self.X_test):
            print(f"  row count mismatch — raw test {len(test_raw)} vs X_test {len(self.X_test)}")
            print("  skipping drift check — make sure Base.csv matches the preprocessed splits")
            return

        model_file = f"{self.model_dir}/{'xgb_model.pkl' if 'SMOTE' not in model_label else 'xgb_smote.pkl'}"
        model  = joblib.load(model_file)
        probs  = model.predict_proba(self.X_test)[:, 1]
        months = test_raw["month"].values

        for m in [6, 7]:
            mask = months == m
            if mask.sum() == 0:
                continue
            y_m  = self.y_test.values[mask]
            p_m  = probs[mask]
            thresh, _, _ = self._threshold_at_fpr(y_m, p_m)
            preds_m = (p_m >= thresh).astype(int)
            recall_m  = recall_score(y_m, preds_m, zero_division=0)
            roc_auc_m = roc_auc_score(y_m, p_m)
            print(f"  month {m}: {mask.sum():,} rows | recall={recall_m:.3f} | roc-auc={roc_auc_m:.3f}")

    def plot_pr_curves(self, results):
        os.makedirs(self.fig_dir, exist_ok=True)
        plt.figure(figsize=(8, 6))
        for r in results:
            prec, rec, _ = precision_recall_curve(self.y_test, r["probs"])
            plt.plot(rec, prec, label=f"{r['label']} (PR-AUC={r['pr_auc']:.3f})")
        baseline = self.y_test.mean()
        plt.axhline(baseline, linestyle="--", color="gray", label=f"random baseline ({baseline:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.fig_dir}/pr_curve.png", dpi=150)
        plt.close()
        print(f"saved -> {self.fig_dir}/pr_curve.png")

    def plot_roc_curves(self, results):
        os.makedirs(self.fig_dir, exist_ok=True)
        plt.figure(figsize=(8, 6))
        for r in results:
            fpr, tpr, _ = roc_curve(self.y_test, r["probs"])
            plt.plot(fpr, tpr, label=f"{r['label']} (AUC={r['roc_auc']:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="random")
        plt.axvline(FPR_CAP, linestyle="--", color="red", alpha=0.5, label=f"{FPR_CAP*100:.0f}% FPR cutoff")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.fig_dir}/roc_curve.png", dpi=150)
        plt.close()
        print(f"saved -> {self.fig_dir}/roc_curve.png")

    def plot_threshold_analysis(self, results, model_label="XGBoost"):
        # shows how recall and precision change as you move the threshold
        res = next(r for r in results if model_label in r["label"])
        probs = res["probs"]
        thresholds = np.linspace(0, 1, 200)
        recalls, precisions, fprs = [], [], []

        negatives = (self.y_test == 0).sum()
        for t in thresholds:
            preds = (probs >= t).astype(int)
            recalls.append(recall_score(self.y_test, preds, zero_division=0))
            precisions.append(precision_score(self.y_test, preds, zero_division=0))
            fp = ((preds == 1) & (self.y_test == 0)).sum()
            fprs.append(fp / negatives)

        os.makedirs(self.fig_dir, exist_ok=True)
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(thresholds, recalls,    label="Recall",    color="steelblue")
        ax1.plot(thresholds, precisions, label="Precision", color="darkorange")
        ax1.set_xlabel("Threshold")
        ax1.set_ylabel("Score")
        ax2 = ax1.twinx()
        ax2.plot(thresholds, fprs, label="FPR", color="red", linestyle="--", alpha=0.6)
        ax2.axhline(FPR_CAP, color="red", linestyle=":", alpha=0.4, label=f"{FPR_CAP*100:.0f}% FPR target")
        ax2.set_ylabel("False Positive Rate")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
        plt.title(f"Threshold Analysis — {model_label}")
        plt.tight_layout()
        plt.savefig(f"{self.fig_dir}/threshold_analysis.png", dpi=150)
        plt.close()
        print(f"saved -> {self.fig_dir}/threshold_analysis.png")

    def run(self, base_csv_path="../Base.csv"):
        self.load()

        models = {
            "RF Baseline":              "rf_baseline.pkl",
            "XGBoost (scale_pos_weight)": "xgb_model.pkl",
            "XGBoost (SMOTE)":          "xgb_smote.pkl",
        }

        results = []
        for label, fname in models.items():
            path = f"{self.model_dir}/{fname}"
            if not os.path.exists(path):
                print(f"skipping {label} — {path} not found")
                continue
            model = joblib.load(path)
            r = self.evaluate_model(model, label)
            results.append(r)

        if results:
            self.plot_pr_curves(results)
            self.plot_roc_curves(results)
            self.plot_threshold_analysis(results, model_label="XGBoost (scale_pos_weight)")
            self.failure_mode_analysis(results, model_label="XGBoost")
            self.temporal_drift(base_csv_path=base_csv_path)

        print("\nall done — figures saved to outputs/figures/")
        return results


if __name__ == "__main__":
    eng = EvaluationEngine()
    eng.run(base_csv_path="../Base.csv")
