"""Prints a patient's entire tokenized trajectory from an ETHOS tokenized dataset.

The patient's static data (e.g., gender, date of birth) is printed first, followed by every token
of their timeline in order, with its timestamp.

Usage:
    python scripts/print_patient_trajectory.py $HOME/ethos-ares-output/train --patient-id 12345
    python scripts/print_patient_trajectory.py $HOME/ethos-ares-output/train --random
    python scripts/print_patient_trajectory.py $HOME/ethos-ares-output/train --random --json
"""

import argparse
import json
import pickle
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from safetensors import safe_open

STATIC_DATA_FN = "static_data.pickle"


def to_datetime(time_us: int) -> datetime:
    # times are stored as microseconds since the Unix epoch
    return datetime(1970, 1, 1) + timedelta(microseconds=int(time_us))


def load_vocab(data_dir: Path) -> list[str]:
    vocab_fps = list(data_dir.glob("vocab_t*.csv"))
    if len(vocab_fps) != 1:
        raise FileNotFoundError(f"Expected one vocab_t*.csv in {data_dir}, found {len(vocab_fps)}")
    return pl.read_csv(vocab_fps[0], has_header=False).to_series(0).to_list()


def find_patient(shard_fps: list[Path], patient_id: int | None) -> tuple[Path, int, int, int]:
    """Returns the shard, patient id and the [start, end) token range of the patient. A random
    patient is picked if `patient_id` is None."""
    if patient_id is None:
        shard_fps = [random.choice(shard_fps)]

    for shard_fp in shard_fps:
        with safe_open(shard_fp, framework="numpy") as f:
            patient_ids = f.get_tensor("patient_ids")
            if patient_id is None:
                pt_idx = random.randrange(len(patient_ids))
            elif (matches := np.flatnonzero(patient_ids == patient_id)).size:
                pt_idx = matches[0]
            else:
                continue

            offsets = f.get_tensor("patient_offsets")
            end = (
                offsets[pt_idx + 1]
                if pt_idx + 1 < len(offsets)
                else f.get_slice("tokens").get_shape()[0]
            )
            return shard_fp, int(patient_ids[pt_idx]), int(offsets[pt_idx]), int(end)

    raise ValueError(f"Patient {patient_id} not found in {shard_fps[0].parent}")


def get_static_data(data_dir: Path, patient_id: int) -> list[tuple[str, str]]:
    with (data_dir / STATIC_DATA_FN).open("rb") as f:
        pt_static_data = pickle.load(f).get(patient_id, {})
    return [
        (str(to_datetime(time)), code)
        for static_data_obj in pt_static_data.values()
        for code, time in zip(static_data_obj["code"], static_data_obj["time"])
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", type=Path, help="Tokenized dataset directory")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--patient-id", type=int, help="Id of the patient to print")
    group.add_argument("--random", action="store_true", help="Print a random patient")
    parser.add_argument("--json", action="store_true", help="Print the trajectory as JSON")
    args = parser.parse_args()

    shard_fps = sorted(args.data_dir.glob("[0-9]*.safetensors"))
    if not shard_fps:
        raise FileNotFoundError(f"No [0-9]*.safetensors files found in {args.data_dir}")

    shard_fp, patient_id, start, end = find_patient(shard_fps, args.patient_id)
    with safe_open(shard_fp, framework="numpy") as f:
        tokens = f.get_slice("tokens")[start:end]
        times = f.get_slice("times")[start:end]

    vocab = load_vocab(args.data_dir)
    trajectory = [(str(to_datetime(t)), vocab[token]) for t, token in zip(times, tokens)]
    static_data = get_static_data(args.data_dir, patient_id)

    if args.json:
        print(
            json.dumps(
                {"patient_id": patient_id, "static_data": static_data, "trajectory": trajectory},
                indent=2,
            )
        )
    else:
        print(f"Patient {patient_id} ({len(trajectory):,} tokens, {shard_fp.name})")
        print("Static data:")
        for time, code in static_data:
            print(f"  {time}  {code}")
        print("Trajectory:")
        for i, (time, token) in enumerate(trajectory):
            print(f"  {i:>6}  {time}  {token}")
