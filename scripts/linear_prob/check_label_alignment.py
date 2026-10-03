"""Checks that a cohort's label prediction_time is on the same clock as the MEDS event times.

For every label with a `visit_occurrence_id`, the MEDS rows of that visit are looked at per source
table: how far their latest timestamp is from the label's prediction_time. A table that shares the
labels' clock has a latest timestamp at (or before) prediction_time; a table shifted by the UTC
offset shows a constant 4 h (daylight time) or 5 h (standard time) difference. `at_prediction_time`
close to 1 for the `visit` table means the labels were made at the visit end and the visit-end row
is included in the model input (the cutoff is inclusive).

Usage:
    python scripts/linear_prob/check_label_alignment.py \
        --labels ~/ohdsi.../cehrgpt_tasks_local_time/hf_readmission_meds/data \
        --meds /home/cp3016/katara_resources/post_transform/data
"""

import argparse
import sys
from pathlib import Path

import polars as pl


def parquet_files(path: str) -> list[Path]:
    p = Path(path).expanduser()
    return sorted(p.rglob("*.parquet")) if p.is_dir() else [p]


def main(args):
    pl.Config.set_tbl_rows(30)
    label_files = parquet_files(args.labels)
    meds_files = parquet_files(args.meds)
    labels = pl.read_parquet(label_files)
    meds_schema = pl.scan_parquet(meds_files[0]).collect_schema()

    print("label prediction_time :", labels.schema["prediction_time"])
    print("MEDS time             :", meds_schema["time"])
    needed = ["subject_id", "time", "table", "visit_id"]
    missing = [c for c in needed if c not in meds_schema.names()]
    if "visit_occurrence_id" not in labels.columns or missing:
        raise SystemExit(
            "The visit based check needs `visit_occurrence_id` in the labels and "
            f"`visit_id` + `table` in the MEDS data (missing in MEDS: {missing or 'none'})."
        )

    labels = labels.select(
        "subject_id",
        "visit_occurrence_id",
        pl.col("prediction_time").cast(pl.Datetime("us")),
    ).unique()
    # sample patients (not labels) and keep all of their labels, so memory stays small
    subjects = labels.get_column("subject_id").unique()
    if len(subjects) > args.sample_patients:
        subjects = subjects.sample(args.sample_patients, seed=0)
    labels = labels.filter(pl.col("subject_id").is_in(subjects))

    # latest timestamp per (patient, visit, source table), one MEDS file at a time
    parts = []
    for i, fp in enumerate(meds_files, 1):
        scan = pl.scan_parquet(fp)
        if any(c not in scan.collect_schema().names() for c in needed):
            print(f"skipping {fp.name}: not an event file", file=sys.stderr)
            continue
        parts.append(
            scan.select(needed)
            .filter(pl.col("subject_id").is_in(subjects))
            .group_by("subject_id", "visit_id", "table")
            .agg(last_time=pl.col("time").max())
            .collect()
        )
        if i % 25 == 0:
            print(f"scanned {i}/{len(meds_files)} MEDS files", file=sys.stderr)
    visit_tables = (
        pl.concat(parts)
        .group_by("subject_id", "visit_id", "table")
        .agg(last_time=pl.col("last_time").max())
    )
    joined = labels.join(
        visit_tables,
        left_on=["subject_id", "visit_occurrence_id"],
        right_on=["subject_id", "visit_id"],
    ).with_columns(
        delta_h=((pl.col("last_time") - pl.col("prediction_time")).dt.total_seconds() / 3600)
    )
    print(f"\nlabels checked: {labels.height:,} (with matching MEDS visit rows: "
          f"{joined.select(pl.struct('subject_id', 'visit_occurrence_id').n_unique()).item():,})")

    print("\nper source table: latest event of the label's visit minus prediction_time, in hours")
    print(
        joined.group_by("table")
        .agg(
            n=pl.len(),
            at_prediction_time=(pl.col("delta_h").abs() < 1 / 60).mean().round(3),
            after_by_3_to_6h=pl.col("delta_h").is_between(3, 6).mean().round(3),
            median_h=pl.col("delta_h").median().round(2),
        )
        .sort("n", descending=True)
    )

    visit = joined.filter(pl.col("table") == "visit")
    if visit.height:
        print("\n`visit` rows: offset to prediction_time (hours), by month of the label")
        print(
            visit.with_columns(
                month=pl.col("prediction_time").dt.month(),
                offset_h=pl.col("delta_h").round(0).cast(pl.Int64),
            )
            .filter(pl.col("offset_h").is_between(-1, 6))
            .group_by("month", "offset_h")
            .len()
            .sort("month", "offset_h")
            .pivot(on="offset_h", index="month", values="len")
            .fill_null(0)
        )
    print(
        "\nReading: a table with `at_prediction_time` high shares the labels' clock (a `visit` "
        "offset of 0 h in every month means no mismatch). A constant 4 h (Apr-Oct) / 5 h "
        "(Nov-Mar) offset on some tables means those tables are shifted by the New York UTC "
        "offset; keep the labels as they are unless every table is shifted, since converting "
        "them would move the cutoff past the true prediction time for the tables that share "
        "its clock. If `visit` is at 1.0, the visit-end row is part of the input: fine when "
        "it is legitimately known at the prediction time, a leak when it defines the outcome."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--labels", required=True, help="label parquet file or directory")
    parser.add_argument("--meds", required=True, help="MEDS parquet file or directory (all splits)")
    parser.add_argument(
        "--sample-patients", type=int, default=5_000, help="patients to check, with all their labels"
    )
    main(parser.parse_args())
