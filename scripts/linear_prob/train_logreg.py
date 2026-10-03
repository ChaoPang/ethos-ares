"""Trains a logistic regression linear probe on top of frozen ETHOS features.

Usage:
    python scripts/linear_prob/train_logreg.py \
        --train_features /path/to/train_features /path/to/tuning_features \
        --test_features /path/to/held_out_features \
        --output_dir /path/to/results

Each features argument is a parquet file or a directory of parquet files, as written by
`ethos_linear_prob_features`; several training sources are concatenated.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def load_features(fps: list[Path]) -> dict:
    files = []
    for fp in fps:
        # a features directory also holds meta.json / .done, so only take its parquet files
        found = sorted(fp.glob("*.parquet")) if fp.is_dir() else [fp]
        if not found:
            raise FileNotFoundError(f"No parquet files in {fp}")
        files.extend(found)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return {
        "subject_id": df["subject_id"].to_numpy(),
        "prediction_time": df["prediction_time"].tolist(),
        "features": np.stack(df["features"].apply(np.asarray)),
        "boolean_value": df["boolean_value"].to_numpy().astype(bool),
    }


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_fp = output_dir / "metrics.json"
    model_fp = output_dir / "model.pickle"

    train = load_features([Path(fp) for fp in args.train_features])
    test = load_features([Path(args.test_features)])

    for name, data in (("train", train), ("test", test)):
        n_pos = int(data["boolean_value"].sum())
        print(f"{name}: {len(data['boolean_value']):,} examples, {n_pos:,} positive")
    if len(set(train["boolean_value"].tolist())) < 2:
        raise SystemExit("The training set has a single class, cannot fit a classifier.")
    if len(set(test["boolean_value"].tolist())) < 2:
        raise SystemExit("The test set has a single class, AUC is undefined.")

    if model_fp.exists():
        print(f"Loading existing model from {model_fp}")
        with open(model_fp, "rb") as f:
            model = pickle.load(f)
    else:
        model = LogisticRegressionCV(scoring="roc_auc", random_state=42, max_iter=1000)
        model.fit(train["features"], train["boolean_value"])
        with open(model_fp, "wb") as f:
            pickle.dump(model, f)

    y_pred = model.predict_proba(test["features"])[:, 1]

    predictions = pl.DataFrame(
        {
            "subject_id": test["subject_id"].tolist(),
            "prediction_time": test["prediction_time"],
            "predicted_boolean_probability": y_pred.tolist(),
            "boolean_value": test["boolean_value"].tolist(),
        }
    )
    predictions.write_parquet(output_dir / "test_predictions.parquet")

    roc_auc = roc_auc_score(test["boolean_value"], y_pred)
    precision, recall, _ = precision_recall_curve(test["boolean_value"], y_pred)
    pr_auc = auc(recall, precision)

    metrics = {"roc_auc": roc_auc, "pr_auc": pr_auc, "n_train": len(train["boolean_value"]),
               "n_test": len(test["boolean_value"])}
    print("Linear probe results:", metrics)
    with open(result_fp, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a logistic regression linear probe")
    parser.add_argument(
        "--train_features", nargs="+", required=True,
        help="Features (file or directory) from ethos_linear_prob_features, one or more",
    )
    parser.add_argument(
        "--test_features", required=True,
        help="Features (file or directory) from ethos_linear_prob_features",
    )
    parser.add_argument("--output_dir", required=True, help="Directory to save model/metrics")
    main(parser.parse_args())
