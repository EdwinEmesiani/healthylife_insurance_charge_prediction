## HealthyLife Insurance Charge Prediction

**Models compared:**
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

**Example metrics (test set):**
- Random Forest: lowest MSE, highest R² (best generalisation).
- Decision Tree: competitive, but more variance.
- Linear Regression: interpretable baseline.

**System attributes:**
- Reproducible training via `train.py`.
- Deployed Gradio app with `model.joblib`.
- Conceptual monitoring for model and data drift.
- Prepared for CI/CD integration.

**Potential Enhancements:**
- Add hyperparameter search for Random Forest.
- Implement full EvidentlyAI dashboards for drift.
- Wire retraining triggers into GitHub Actions.
