from datetime import timedelta
from pathlib import Path

import polars as pl
import torch as th

from .base import InferenceDataset


def _resolve_label_indices(dataset: InferenceDataset, labels_fp: str | Path) -> tuple[th.Tensor, list[dict]]:
    """Matches ACES/MEDS labels (subject_id, prediction_time, boolean_value) to the index of
    the last token at or before each label's prediction_time.

    Returns (start_indices, rows) for only the labels that could be matched; labels whose
    subject_id isn't in the dataset, or whose prediction_time falls before that patient's
    first recorded token, are skipped.
    """
    labels_fp = Path(labels_fp)
    labels_source = str(labels_fp / "**" / "*.parquet") if labels_fp.is_dir() else labels_fp
    labels_df = pl.read_parquet(labels_source).select(
        "subject_id", "prediction_time", "boolean_value"
    )

    pred_time_dtype = labels_df.schema["prediction_time"]
    if not isinstance(pred_time_dtype, pl.Datetime):
        raise TypeError(
            f"Expected 'prediction_time' in '{labels_fp}' to be a Datetime column, got "
            f"{pred_time_dtype}. It must be an actual timestamp so its unit can be rescaled to "
            "microseconds -- a raw integer column is ambiguous (unknown unit/epoch) and cannot "
            "be matched against the tokenized dataset's `times` safely."
        )

    labels_df = (
        labels_df
        # `times` is stored in microseconds since epoch (see TimelineDataset.tensorize), but a
        # MEDS label's prediction_time may be written with a different precision (e.g. pandas/
        # Spark commonly use nanoseconds) -- rescale explicitly rather than reinterpreting bits.
        .with_columns(
            prediction_time=pl.col("prediction_time").cast(pl.Datetime("us")).cast(pl.Int64)
        )
        .sort("subject_id", "prediction_time")
    )

    patient_starts = dataset.patient_offsets
    patient_ends = th.cat((patient_starts[1:], th.tensor([len(dataset.times)])))
    patient_range = {
        pid.item(): (start.item(), end.item())
        for pid, start, end in zip(dataset.patient_ids, patient_starts, patient_ends)
    }

    start_indices, kept_rows, skipped = [], [], 0
    for row in labels_df.iter_rows(named=True):
        pt_range = patient_range.get(row["subject_id"])
        if pt_range is None:
            skipped += 1
            continue
        start, end = pt_range
        times_slice = dataset.times[start:end]
        cutoff = th.tensor(row["prediction_time"], dtype=times_slice.dtype)
        offset = th.searchsorted(times_slice, cutoff, right=True).item() - 1
        if offset < 0:
            # prediction_time falls before this patient's first recorded token
            skipped += 1
            continue
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
        **kwargs,
    ):
        super().__init__(input_dir, n_positions, **kwargs)
        self.start_indices, self.labels = _resolve_label_indices(self, labels_fp)

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

    `outcome_stoken` names the special token that represents a positive outcome (e.g. ST.DEATH
    for a mortality label) -- the generation loop watches for it exactly like it does for the
    built-in tasks' outcome tokens. A label's `boolean_value` supplies the ground truth
    ("expected") that scoring compares the model's generated/sampled outcome against.
    """

    def __init__(
        self,
        input_dir: str | Path,
        labels_fp: str | Path,
        outcome_stoken: str,
        n_positions: int = 2048,
        time_limit_days: float | None = None,
        **kwargs,
    ):
        super().__init__(input_dir, n_positions, **kwargs)
        self.stop_stokens = [outcome_stoken] + [
            s for s in self.stop_stokens if s != outcome_stoken
        ]
        self._outcome_stoken = outcome_stoken
        if time_limit_days is not None:
            self.time_limit = timedelta(days=time_limit_days)
        self.start_indices, self.labels = _resolve_label_indices(self, labels_fp)

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx) -> tuple[th.Tensor, dict]:
        start_idx = self.start_indices[idx]
        row = self.labels[idx]
        y = {
            "expected": self._outcome_stoken if row["boolean_value"] else "NEGATIVE",
            "true_token_dist": None,
            "true_token_time": None,
            "patient_id": row["subject_id"],
            "prediction_time": row["prediction_time"],
            "data_idx": start_idx.item(),
            "boolean_value": row["boolean_value"],
        }
        return super().__getitem__(start_idx), y
