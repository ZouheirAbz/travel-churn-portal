import joblib

meta = {
    "suggested_model_from_cv": "Gradient Boosting",
    "roc_auc": 0.973,
    "best_threshold_by_f1": 0.30,
    "best_threshold_precision": 0.763636,
    "best_threshold_recall": 0.933333,
    "best_threshold_f1": 0.84
}

joblib.dump(meta, "dashboard_meta.pkl")
print("dashboard_meta.pkl recreated successfully.")
