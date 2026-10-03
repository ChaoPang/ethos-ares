from pathlib import Path

import hydra
import polars as pl
import torch as th
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from ..datasets import LabelledCohortDataset
from ..utils import load_model_checkpoint, setup_torch


@hydra.main(version_base=None, config_path="../configs", config_name="linear_prob_features")
def main(cfg: DictConfig):
    device = cfg.device

    model_checkpoint = th.load(cfg.model_fp, map_location="cpu", mmap=True, weights_only=False)
    model_config = model_checkpoint["model_config"]
    if model_config.is_encoder_decoder:
        raise NotImplementedError("Linear probing currently supports decoder-only models only.")
    n_positions = model_config.n_positions

    dataset = LabelledCohortDataset(
        input_dir=cfg.input_dir, labels_fp=cfg.labels_fp, n_positions=n_positions
    )
    logger.info(f"{dataset} initialized with {len(dataset):,} labelled examples.")

    model, _ = load_model_checkpoint(cfg.model_fp, map_location="cpu")
    model = model.eval().to(device)
    model = th.compile(model, disable=cfg.no_compile)

    autocast_context = setup_torch(device, dtype="bfloat16" if "cuda" in device else "float32")

    out_dir = Path(cfg.output_dir) / (cfg.output_fn or "features")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, n_chunks, n_written = [], 0, 0

    def flush():
        nonlocal rows, n_chunks, n_written
        if not rows:
            return
        pl.DataFrame(rows).write_parquet(out_dir / f"part-{n_chunks:05d}.parquet")
        n_chunks += 1
        n_written += len(rows)
        rows = []

    with th.no_grad():
        for x, y in tqdm(dataset, desc="Computing features", total=len(dataset)):
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
