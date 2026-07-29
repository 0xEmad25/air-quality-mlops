from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from air_quality.features import FEATURES

CANDIDATES: dict[str, object] = {
    "baseline_majority": DummyClassifier(strategy="most_frequent"),
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        n_estimators=250, max_depth=10, class_weight="balanced", random_state=42
    ),
}


def split_by_time(df: pd.DataFrame):
    """Chronological 70/15/15 train/validation/test split — no leakage."""
    n = len(df)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)
    return df.iloc[:train_end], df.iloc[train_end:valid_end], df.iloc[valid_end:]


def make_pipeline(model):
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), FEATURES),
    ])
    return Pipeline([("preprocess", preprocess), ("model", model)])


def main() -> None:
    df = pd.read_parquet("data/processed/model_table.parquet")
    train, valid, test = split_by_time(df)

    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("riyadh-air-quality")

    best_name, best_pipeline, best_f1 = None, None, -1.0

    for name, model in CANDIDATES.items():
        pipeline = make_pipeline(model)
        pipeline.fit(train[FEATURES], train["high_pollution_next_hour"])

        y_pred = pipeline.predict(valid[FEATURES])
        try:
            y_prob = pipeline.predict_proba(valid[FEATURES])[:, 1]
        except AttributeError:
            y_prob = y_pred.astype(float)

        f1 = f1_score(valid["high_pollution_next_hour"], y_pred)
        recall = recall_score(valid["high_pollution_next_hour"], y_pred)
        try:
            roc_auc = roc_auc_score(valid["high_pollution_next_hour"], y_prob)
        except ValueError:
            roc_auc = float("nan")

        metrics = {
            "validation_f1": f1,
            "validation_recall": recall,
            "validation_roc_auc": roc_auc,
        }

        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "threshold": 0.50})
            # Filter out NaN/None metrics to avoid MLflow SQLite UNIQUE constraint bug
            clean_metrics = {k: v for k, v in metrics.items() if not pd.isna(v)}
            mlflow.log_metrics(clean_metrics)
            mlflow.sklearn.log_model(pipeline, name="model", skops_trusted_types=["numpy.dtype"])
            print(f"  {name:25s}  F1={f1:.4f}  Recall={recall:.4f}  ROC-AUC={roc_auc if not pd.isna(roc_auc) else 'N/A':>8}")

        if not pd.isna(f1) and name != "baseline_majority" and f1 > best_f1:
            best_name, best_pipeline, best_f1 = name, pipeline, f1

    assert best_pipeline is not None, "No model was trained successfully"

    # Evaluate the selected model on the held-out test set
    test_y_pred = best_pipeline.predict(test[FEATURES])
    try:
        test_y_prob = best_pipeline.predict_proba(test[FEATURES])[:, 1]
    except AttributeError:
        test_y_prob = test_y_pred.astype(float)

    test_metrics = {
        "test_f1": f1_score(test["high_pollution_next_hour"], test_y_pred),
        "test_recall": recall_score(test["high_pollution_next_hour"], test_y_pred),
        "test_roc_auc": roc_auc_score(test["high_pollution_next_hour"], test_y_prob),
    }

    print(f"\nSelected model: {best_name}")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    Path("models").mkdir(exist_ok=True)
    joblib.dump({"pipeline": best_pipeline, "threshold": 0.50}, "models/model.joblib")
    print("\nSaved model artifact to models/model.joblib")

    # Save reference batch for Evidently monitoring
    test[FEATURES + ["high_pollution_next_hour"]].to_csv(
        "data/monitoring/reference.csv", index=False
    )
    print("Saved reference batch to data/monitoring/reference.csv")


if __name__ == "__main__":
    main()
