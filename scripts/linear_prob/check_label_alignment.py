"""Checks that a cohort's label prediction_time is on the same clock as the MEDS event times.

For every label with a `visit_occurrence_id`, the MEDS rows of that visit are looked at per source
table: how far their latest timestamp is from the label's prediction_time. A table that shares the
labels' clock has a latest timestamp at (or before) prediction_time; a table shifted by the UTC
offset shows a constant 4 h (daylight time) or 5 h (standard time) difference.

Usage:
    python scripts/linear_prob/check_label_alignment.py \
        --labels ~/ohdsi.../cehrgpt_tasks_local_time/hf_readmission_meds/data \
        --meds /home/cp3016/katara_resources/post_transform/data
"""

import argparse
from pathlib import Path

import polars as pl


def parquet_glob(path: str) -> str:
    p = Path(path).expanduser()
    return str(p / "**" / "*.parquet") if p.is_dir() else str(p)


def main(args):
    pl.Config.set_tbl_rows(30)
    labels = pl.read_parquet(parquet_glob(args.labels))
    meds = pl.scan_parquet(parquet_glob(args.meds))
    meds_schema = meds.collect_schema()

    print("label prediction_time :", labels.schema["prediction_time"])
    print("MEDS time             :", meds_schema["time"])
    missing = [c for c in ("visit_id", "table") if c not in meds_schema.names()]
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
    if args.sample and len(labels) > args.sample:
        labels = labels.sample(args.sample, seed=0)

    # latest timestamp per (visit, source table)
    visit_tables = (
        meds.join(
            labels.lazy().select("subject_id", "visit_occurrence_id").unique(),
            left_on=["subject_id", "visit_id"],
            right_on=["subject_id", "visit_occurrence_id"],
        )
        .group_by("subject_id", "visit_id", "table")
        .agg(last_time=pl.col("time").max())
        .collect()
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
        "\nReading: a table with `at_prediction_time` high shares the labels' clock. A `visit` "
        "table whose offset is 4 h in Apr-Oct and 5 h in Nov-Mar is shifted by the New York UTC "
        "offset. Keep the labels as they are in that case: converting them would push the "
        "cutoff past the true prediction time for the tables that share its clock."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--labels", required=True, help="label parquet file or directory")
    parser.add_argument("--meds", required=True, help="MEDS parquet file or directory (all splits)")
    parser.add_argument("--sample", type=int, default=100_000, help="max labels to check")
    main(parser.parse_args())
