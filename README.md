# Early-Detection of Abnormal Public Spending Using Multi-Resolution Data

## Project Overview
This project implements a proactive machine learning framework for the **Early-Detection** of fiscal anomalies in public spending.

## Team Members
* **Anisha** (Department of Mathematics, UIUC)
* **Ananyaa Tanwar** (School of Information Sciences, UIUC)
* **Rahul Balasubramani** (School of Information Sciences, UIUC)

## Data Sources
We utilize datasets from the **Urbana County Regional Planning Commission (CCRPC)**:
1. **Dataset A (Macro):** City Budget and Spending (2024–2026).
2. **Dataset B (Micro):** Urbana Open Expenditures (Payment Details).

## Methodology
We compare three unsupervised models to identify diverse anomaly typologies:
* **Isolation Forest:** For global outlier detection.
* **Local Outlier Factor (LOF):** For department-specific local anomalies.
* **One-Class SVM:** For boundary-based spending behavior analysis.


## Evaluation & Explainability
- **Synthetic Injection:** Because data is unlabeled, we evaluate efficacy by "poisoning" datasets with 20+ known anomalous patterns to measure **Recall**.
- **SHAP Explainability:** Every flagged transaction includes a reasoning string (e.g., "Flagged due to High Amount + New Vendor").
- **Deployment:** Interactive audit dashboard built with **Streamlit**.
