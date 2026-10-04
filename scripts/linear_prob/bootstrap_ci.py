"""Bootstrap confidence intervals of ROC-AUC and PR-AUC for every cohort of a linear probing run.

A cohort can have several labels per patient (one per visit), which are correlated, so the
bootstrap resamples patients with all their labels, not single labels. Each replicate draws as many
patients as the test set has, with replacement; a patient drawn twice counts twice (as a sample
weight). The interval is the percentile interval of the replicates.

PR-AUC is computed as in train_logreg.py (trapezoid area under the precision-recall curve), so the
point estimate matches metrics.json. `--pr-auc average_precision` uses the average precision of
zero-shot runs instead (scripts/zero_shot/run_zero_shot.py).

Usage:
    python scripts/linear_prob/bootstrap_ci.py --results-dir ~/Documents/linear_probing

Reads <results-dir>/<cohort>/logreg/test_predictions.parquet (subject_id, boolean_value and
predicted_boolean_probability; the zero-shot predictions.parquet with y and risk works too, see
--glob) and writes <results-dir>/bootstrap_ci.csv and a bootstrap_ci.json next to each predictions
file.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

LABEL_COLUMNS = ("boolean_value", "y")
SCORE_COLUMNS = ("predicted_boolean_probability", "risk")


def read_predictions(fp: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """subject index per label, label, score"""
    df = pl.read_parquet(fp)
    label_col = next(c for c in LABEL_COLUMNS if c in df.columns)
    score_col = next(c for c in SCORE_COLUMNS if c in df.columns)
    subject_idx = df["subject_id"].rank("dense").to_numpy().astype(np.int64) - 1
    return subject_idx, df[label_col].cast(pl.Int8).to_numpy(), df[score_col].to_numpy()


def metrics(y, score, weight, pr_auc_kind: str) -> tuple[float, float]:
    roc = roc_auc_score(y, score, sample_weight=weight)
    if pr_auc_kind == "average_precision":
        return roc, average_precision_score(y, score, sample_weight=weight)
    precision, recall, _ = precision_recall_curve(y, score, sample_weight=weight)
    return roc, auc(recall, precision)


def replicates(subject_idx, y, score, n_subjects, seeds, pr_auc_kind) -> list[tuple[float, float]]:
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, n_subjects, n_subjects)
        weight = np.bincount(draws, minlength=n_subjects)[subject_idx]
        keep = weight > 0
        w = weight[keep]
        if len(np.unique(y[keep])) < 2:  # a replicate without both classes has no AUC
            out.append((np.nan, np.nan))
            continue
        out.append(metrics(y[keep], score[keep], w, pr_auc_kind))
    return out


def bootstrap(fp: Path, args) -> dict:
    subject_idx, y, score = read_predictions(fp)
    n_subjects = int(subject_idx.max()) + 1
    roc, pr = metrics(y, score, None, args.pr_auc)

    seeds = np.random.SeedSequence(args.seed).generate_state(args.n_boot)
    chunks = np.array_split(seeds, args.n_jobs * 4)
    reps = Parallel(n_jobs=args.n_jobs)(
        delayed(replicates)(subject_idx, y, score, n_subjects, c, args.pr_auc) for c in chunks if len(c)
    )
    reps = np.array([r for chunk in reps for r in chunk], dtype=float)
    reps = reps[~np.isnan(reps).any(axis=1)]
    lo, hi = 100 * args.alpha / 2, 100 * (1 - args.alpha / 2)

    result = {"n_labels": len(y), "n_patients": n_subjects, "prevalence": float(y.mean()),
              "n_boot": len(reps), "alpha": args.alpha, "pr_auc_kind": args.pr_auc}
    for name, i, point in (("roc_auc", 0, roc), ("pr_auc", 1, pr)):
        result[name] = float(point)
        result[f"{name}_low"], result[f"{name}_high"] = (
            float(v) for v in np.percentile(reps[:, i], [lo, hi])
        )
        result[f"{name}_boot_mean"] = float(reps[:, i].mean())
    return result


def main(args):
    results_dir = Path(args.results_dir).expanduser()
    files = sorted(results_dir.glob(args.glob))
    if args.cohorts:
        files = [f for f in files if f.relative_to(results_dir).parts[0] in args.cohorts]
    if not files:
        raise SystemExit(f"No predictions match {results_dir / args.glob}")

    rows = []
    for fp in files:
        cohort = fp.relative_to(results_dir).parts[0]
        res = bootstrap(fp, args)
        (fp.parent / "bootstrap_ci.json").write_text(json.dumps(res, indent=2))
        rows.append({"cohort": cohort, **res})
        print(
            f"{cohort:32s} n={res['n_labels']:>7,} patients={res['n_patients']:>7,} "
            f"ROC-AUC {res['roc_auc']:.3f} [{res['roc_auc_low']:.3f}, {res['roc_auc_high']:.3f}]  "
            f"PR-AUC {res['pr_auc']:.3f} [{res['pr_auc_low']:.3f}, {res['pr_auc_high']:.3f}]",
            flush=True,
        )
    pl.DataFrame(rows).write_csv(results_dir / "bootstrap_ci.csv")
    print(f"\nWrote {results_dir / 'bootstrap_ci.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-dir", required=True, help="folder with one folder per cohort")
    parser.add_argument("--glob", default="*/logreg/test_predictions.parquet",
                        help="predictions files below --results-dir; the first folder is the cohort")
    parser.add_argument("--cohorts", nargs="+", help="default: every cohort found")
    parser.add_argument("--n-boot", type=int, default=1000, help="bootstrap replicates")
    parser.add_argument("--alpha", type=float, default=0.05, help="0.05: 95%% interval")
    parser.add_argument("--pr-auc", choices=["trapezoid", "average_precision"], default="trapezoid")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
