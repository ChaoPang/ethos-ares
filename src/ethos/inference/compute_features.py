import json
import os
from pathlib import Path

import hydra
import polars as pl
import torch as th
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from ..datasets import LabelledCohortDataset
from ..utils import load_model_checkpoint, setup_torch


def _prepare_resume(out_dir: Path, chunk_size: int, fingerprint: dict) -> int:
    """Returns the number of leading complete chunks already in `out_dir` so extraction can
    continue after them. Incomplete or unreadable trailing files are deleted, and resuming is
    refused if the earlier chunks came from a different checkpoint, labels or settings."""
    for fp in out_dir.glob("*.tmp"):
        fp.unlink()

    parts = sorted(out_dir.glob("part-*.parquet"))
    meta_fp = out_dir / "meta.json"
    if parts and meta_fp.exists():
        previous = json.loads(meta_fp.read_text())
        if previous != fingerprint:
            diff = {k: (previous.get(k), v) for k, v in fingerprint.items() if previous.get(k) != v}
            raise RuntimeError(
                f"'{out_dir}' holds features from a different run ({diff}). "
                "Delete it to start over."
            )
    elif parts:
        logger.warning(f"No meta.json in '{out_dir}', cannot verify the earlier chunks match.")

    n_ok = 0
    for i, fp in enumerate(parts):
        if fp.name != f"part-{i:05d}.parquet":
            break
        try:
            n_rows = pl.scan_parquet(fp).select(pl.len()).collect().item()
        except Exception:
            break
        if n_rows != chunk_size:  # only the final chunk of a finished split is shorter
            break
        n_ok += 1
    for fp in parts[n_ok:]:
        fp.unlink()

    meta_fp.write_text(json.dumps(fingerprint, indent=2))
    return n_ok


@hydra.main(version_base=None, config_path="../configs", config_name="linear_prob_features")
def main(cfg: DictConfig):
    device = cfg.device

    model_checkpoint = th.load(cfg.model_fp, map_location="cpu", mmap=True, weights_only=False)
    model_config = model_checkpoint["model_config"]
    if model_config.is_encoder_decoder:
        raise NotImplementedError("Linear probing currently supports decoder-only models only.")
    n_positions = model_config.n_positions

    dataset = LabelledCohortDataset(
        input_dir=cfg.input_dir,
        labels_fp=cfg.labels_fp,
        n_positions=n_positions,
    )
    logger.info(f"{dataset} initialized with {len(dataset):,} labelled examples.")

    model, _ = load_model_checkpoint(cfg.model_fp, map_location="cpu")
    model = model.eval().to(device)
    model = th.compile(model, disable=cfg.no_compile)

    autocast_context = setup_torch(device, dtype="bfloat16" if "cuda" in device else "float32")

    out_dir = Path(cfg.output_dir) / (cfg.output_fn or "features")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_stat = Path(cfg.model_fp).stat()
    fingerprint = {
        "model_fp": str(Path(cfg.model_fp).resolve()),
        "model_size": model_stat.st_size,
        "model_mtime_ns": model_stat.st_mtime_ns,
        "labels_fp": str(Path(cfg.labels_fp).resolve()),
        "input_dir": str(Path(cfg.input_dir).resolve()),
        "n_examples": len(dataset),
        "chunk_size": cfg.chunk_size,
        "average_over_sequence": bool(cfg.average_over_sequence),
    }
    n_chunks = _prepare_resume(out_dir, cfg.chunk_size, fingerprint)
    n_done = n_chunks * cfg.chunk_size
    if n_done:
        logger.info(f"Resuming after {n_chunks} finished chunk(s), {n_done:,} examples")

    rows, n_written = [], n_done

    def flush():
        nonlocal rows, n_chunks, n_written
        if not rows:
            return
        tmp_fp = out_dir / f"part-{n_chunks:05d}.parquet.tmp"
        pl.DataFrame(rows).write_parquet(tmp_fp)
        os.replace(tmp_fp, out_dir / f"part-{n_chunks:05d}.parquet")  # atomic, no torn files
        n_chunks += 1
        n_written += len(rows)
        rows = []

    with th.no_grad():
        for i in tqdm(range(n_done, len(dataset)), desc="Computing features",
                      initial=n_done, total=len(dataset)):
            x, y = dataset[i]
            x = x.unsqueeze(0).to(device, non_blocking=True)
            with autocast_context:
                output = model(x, output_hidden_states=True)
            hidden_states = output.hidden_states[0]  # (seq_len, n_embd)

            features = (
                hidden_states.mean(dim=0) if cfg.average_over_sequence else hidden_states[-1]
            )

            rows.append(
                {
                    "subject_id": y["subject_id"],
                    "prediction_time": y["prediction_time"],
                    "boolean_value": y["boolean_value"],
                    "features": features.float().cpu().numpy().tolist(),
                }
            )
            # features are held as python lists (~25 KB per row at n_embd=768), so flush often
            if len(rows) >= cfg.chunk_size:
                flush()
    flush()

    logger.info(f"Saved {n_written:,} feature rows in {n_chunks} file(s) to '{out_dir}'")


if __name__ == "__main__":
    main()
