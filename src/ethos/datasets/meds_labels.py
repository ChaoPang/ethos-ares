from datetime import timedelta
from pathlib import Path

import polars as pl
import torch as th

from .base import InferenceDataset


# column names of the older cehr-bert / cehr-gpt label tables and their MEDS label equivalents
LEGACY_LABEL_COLUMNS = {
    "person_id": "subject_id",
    "index_date": "prediction_time",
    "label": "boolean_value",
}


def read_label_table(labels_fp: str | Path) -> pl.DataFrame:
    """Reads label parquet file(s), a file or a folder searched recursively, as MEDS labels.

    Besides the MEDS columns (subject_id, prediction_time, boolean_value), the older
    person_id / index_date / label names are accepted and renamed. A 0/1 label becomes a boolean.
    Other columns (e.g. time_to_event, outcome_date) are kept as they are.
    """
    labels_fp = Path(labels_fp)
    source = str(labels_fp / "**" / "*.parquet") if labels_fp.is_dir() else labels_fp
    df = pl.read_parquet(source)
    original_columns = df.columns
    df = df.rename(
        {old: new for old, new in LEGACY_LABEL_COLUMNS.items()
         if old in df.columns and new not in df.columns}
    )
    missing = [c for c in LEGACY_LABEL_COLUMNS.values() if c not in df.columns]
    if missing:
        raise ValueError(
            f"'{labels_fp}' lacks {missing}; expected subject_id, prediction_time, boolean_value "
            f"(or {list(LEGACY_LABEL_COLUMNS)}), found {original_columns}"
        )
    if df.schema["boolean_value"] != pl.Boolean:
        values = set(df["boolean_value"].drop_nulls().unique().to_list())
        if not values <= {0, 1}:
            raise ValueError(f"The label column of '{labels_fp}' must be 0/1 or boolean, got {values}")
        df = df.with_columns(pl.col("boolean_value").cast(pl.Boolean))
    return df


def _resolve_label_indices(
    dataset: InferenceDataset, labels_fp: str | Path, strict_cutoff: bool = False
) -> tuple[th.Tensor, list[dict]]:
    """Matches ACES/MEDS labels (subject_id, prediction_time, boolean_value) to the index of
    the last token at or before each label's prediction_time.

    Returns (start_indices, rows) for only the labels that could be matched; labels whose
    subject_id isn't in the dataset, or whose prediction_time falls before that patient's
    first recorded token, are skipped.

    prediction_time is compared as-is with the tokenized `times`, so both must be on the same
    wall-clock; verify it with scripts/linear_prob/check_label_alignment.py instead of assuming
    it. Events at exactly prediction_time are included, unless `strict_cutoff` is set, in which case
    only events strictly before it are. Use that when the label is made at the very event that
    defines the outcome (e.g. a visit-end row carrying the discharge disposition).

    Do not convert the labels to UTC to "fix" a mismatch without checking which tables are off:
    in one OMOP MEDS extract only the `visit` rows were shifted by the UTC offset while the
    clinical tables shared the labels' clock, and converting would then push the cutoff up to
    5 h past the true prediction time and let future clinical events into the input. In the
    CUMC post_transform data the visit rows coincide with the labels' prediction_time.
    """
    labels_fp = Path(labels_fp)
    labels_df = read_label_table(labels_fp).select("subject_id", "prediction_time", "boolean_value")

    pred_time_dtype = labels_df.schema["prediction_time"]
    if not isinstance(pred_time_dtype, pl.Datetime):
        raise TypeError(
            f"Expected 'prediction_time' in '{labels_fp}' to be a Datetime column, got "
            f"{pred_time_dtype}. It must be an actual timestamp so its unit can be rescaled to "
            "microseconds -- a raw integer column is ambiguous (unknown unit/epoch) and cannot "
            "be matched against the tokenized dataset's `times` safely."
        )

    if pred_time_dtype.time_zone is not None:
        raise TypeError(
            f"'prediction_time' in '{labels_fp}' is timezone-aware ({pred_time_dtype}), but the "
            "tokenized times are naive. Convert it to the events' wall-clock time and store it "
            "naive."
        )

    labels_df = (
        labels_df
        # `times` is stored in microseconds since epoch (see TimelineDataset.tensorize), but a
        # MEDS label's prediction_time may be written with a different precision (e.g. pandas/
        # Spark commonly use nanoseconds) -- rescale explicitly rather than reinterpreting bits.
        .with_columns(
            prediction_time=pl.col("prediction_time").cast(pl.Datetime("us")).cast(pl.Int64)
        )
        .sort("subject_id", "prediction_time", maintain_order=True)
    )

    patient_starts = dataset.patient_offsets
    patient_ends = th.cat((patient_starts[1:], th.tensor([len(dataset.times)])))
    patient_range = {
        pid.item(): (start.item(), end.item())
        for pid, start, end in zip(dataset.patient_ids, patient_starts, patient_ends)
    }

    start_indices, kept_rows, skipped = [], [], 0
    # labels are sorted by subject_id, so consecutive rows usually share a patient
    cached_pid, times_slice = None, None
    for row in labels_df.iter_rows(named=True):
        pt_range = patient_range.get(row["subject_id"])
        if pt_range is None:
            skipped += 1
            continue
        start, end = pt_range
        if row["subject_id"] != cached_pid:
            times_slice = dataset.times[start:end]
            cached_pid = row["subject_id"]
        cutoff = th.tensor(row["prediction_time"], dtype=times_slice.dtype)
        offset = th.searchsorted(times_slice, cutoff, right=not strict_cutoff).item() - 1
        if offset < 0:
            # prediction_time falls before this patient's first recorded token
            skipped += 1
            continue
        # never let the input reach past the prediction time
        in_input = (lambda t: t < cutoff) if strict_cutoff else (lambda t: t <= cutoff)
        if not in_input(times_slice[offset]) or (
            offset + 1 < len(times_slice) and in_input(times_slice[offset + 1])
        ):
            raise RuntimeError(
                f"Cutoff for subject {row['subject_id']} is not the last event "
                f"{'before' if strict_cutoff else 'at or before'} "
                f"{row['prediction_time']}."
            )
        start_indices.append(start + offset)
        kept_rows.append(row)

    if not start_indices:
        raise ValueError(
            f"None of the {len(labels_df)} labels could be matched to patients/times in the "
            "tokenized dataset."
        )
    if skipped:
        print(
            f"Skipped {skipped}/{len(labels_df)} labels with no matching patient or no data "
            "before prediction_time."
        )
    return th.tensor(start_indices), kept_rows


class LabelledCohortDataset(InferenceDataset):
    """Builds timelines cut at externally supplied prediction times, pairing each with its
    ACES-style MEDS label (subject_id, prediction_time, boolean_value), instead of deriving the
    cutoff from special tokens the way the other `InferenceDataset` subclasses do.

    Intended for linear probing: extracting a frozen model's hidden state at each label's
    prediction_time to train a downstream classifier on top of it. For running the model's own
    generation-based inference against MEDS labels instead, see `MedsLabelledDataset`.
    """

    def __init__(
        self,
        input_dir: str | Path,
        labels_fp: str | Path,
        n_positions: int = 2048,
        strict_cutoff: bool = False,
        **kwargs,
    ):
        super().__init__(input_dir, n_positions, **kwargs)
        self.start_indices, self.labels = _resolve_label_indices(self, labels_fp, strict_cutoff)

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx) -> tuple[th.Tensor, dict]:
        start_idx = self.start_indices[idx]
        row = self.labels[idx]
        y = {
            "subject_id": row["subject_id"],
            "prediction_time": row["prediction_time"],
            "boolean_value": row["boolean_value"],
            "data_idx": start_idx.item(),
        }
        return super().__getitem__(start_idx), y


class MedsLabelledDataset(InferenceDataset):
    """Runs the standard `ethos_infer` generation loop against externally supplied ACES/MEDS
    labels (subject_id, prediction_time, boolean_value), instead of requiring the ground truth
    be reconstructable by scanning the raw timeline for what actually happened next.

    `outcome_stoken` names the special token(s) that represent a positive outcome (e.g.
    ST.DEATH for a mortality label, or several admission-type codes for a readmission label
    when no single canonical "admission" token exists in the vocabulary) -- the generation loop
    watches for any of them exactly like it does for the built-in tasks' outcome tokens. A
    label's `boolean_value` supplies the ground truth ("expected") that scoring compares the
    model's generated/sampled outcome against.

    `include_base_stop_stokens` controls whether `InferenceDataset`'s default stop tokens
    (ST.DEATH, ST.TIMELINE_END) are also watched for alongside `outcome_stoken`. Set to False
    if the tokenized dataset doesn't contain one of those tokens at all (e.g. no deaths were
    recorded) -- vocab.encode would otherwise raise a KeyError for every worker process.
    """

    def __init__(
        self,
        input_dir: str | Path,
        labels_fp: str | Path,
        outcome_stoken: str | list[str],
        n_positions: int = 2048,
        time_limit_days: float | None = None,
        include_base_stop_stokens: bool = True,
        strict_cutoff: bool = False,
        **kwargs,
    ):
        super().__init__(input_dir, n_positions, **kwargs)
        outcome_stokens = (
            [outcome_stoken] if isinstance(outcome_stoken, str) else list(outcome_stoken)
        )
        base_stop_stokens = self.stop_stokens if include_base_stop_stokens else []
        self.stop_stokens = outcome_stokens + [
            s for s in base_stop_stokens if s not in outcome_stokens
        ]
        self._outcome_stokens = set(outcome_stokens)
        if time_limit_days is not None:
            self.time_limit = timedelta(days=time_limit_days)
        self.start_indices, self.labels = _resolve_label_indices(self, labels_fp, strict_cutoff)

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx) -> tuple[th.Tensor, dict]:
        start_idx = self.start_indices[idx]
        row = self.labels[idx]
        y = {
            "expected": "POSITIVE" if row["boolean_value"] else "NEGATIVE",
            "true_token_dist": None,
            "true_token_time": None,
            "patient_id": row["subject_id"],
            "prediction_time": row["prediction_time"],
            "data_idx": start_idx.item(),
            "boolean_value": row["boolean_value"],
        }
        return super().__getitem__(start_idx), y
