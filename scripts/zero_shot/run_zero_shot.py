"""Zero-shot evaluation of a pretrained ETHOS checkpoint on MEDS label cohorts.

For every label (subject_id, prediction_time, boolean_value) the model is given the patient's
history up to prediction_time and generates `rep_num` future trajectories. A trajectory ends when
it produces an outcome token (an event), a stop token (death or end of the record) or when its
generated time passes the cohort's time limit. The share of trajectories that ended in an event is
the predicted risk, which is scored against boolean_value. This mirrors ETHOS's own 30-day
readmission benchmark, with the outcome tokens taken from the yaml files next to this script
(30_day_hf_readmission.yaml, 10_year_t2dm_hf.yaml, 1_year_cabg.yaml, one cohort each).

The labels are matched to the tokenized split they belong to, so use the split the label patients
come from (the zero-shot cohort samples are cut from held_out). A label table has subject_id,
prediction_time and boolean_value; the older person_id, index_date and label names are accepted
too (other columns, e.g. outcome_date, are ignored).

Usage:
    python scripts/zero_shot/run_zero_shot.py \
        --model-fp /path/to/best_model.pt \
        --cohorts-dir ~/ohdsi.../ethos_zero_shot_cohorts \
        --tokenized-dir /path/to/ethos-output \
        --out-dir /path/to/zero_shot \
        --cohorts hf_readmission_sample t2dm_hf_sample cad_cabg_sample

Rerunning continues where it stopped: the labels are cut into shards, finished shards are skipped
and the scores are recomputed from all of them. A shard that stopped before it finished keeps the
labels whose trajectories are all there (the results are written every --flush-labels labels) and
only the other labels of the shard are generated again. Rerunning with a different checkpoint,
labels, tokenized data or setting stops with an error instead of mixing results.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score, roc_auc_score

from ethos.constants import SpecialToken as ST
from ethos.datasets import MedsLabelledDataset
from ethos.datasets.meds_labels import read_label_table
from ethos.vocabulary import Vocabulary

STOP_REASONS_KEPT = ("token_of_interest", "time_limit")  # key_error: an undecodable token, token_limit: no end within the budget


def parquet_files(path: Path) -> list[Path]:
    path = path.expanduser()
    return sorted(path.rglob("*.parquet")) if path.is_dir() else [path]


def file_signature(paths: list[Path]) -> list[str]:
    return [f"{p.resolve()}|{p.stat().st_size}|{p.stat().st_mtime_ns}" for p in paths]


def hydra_list(values: list[str]) -> str:
    return "[" + ",".join(f"'{v}'" for v in values) + "]"


def resolve_outcome_tokens(vocab: Vocabulary, tokens: list[str], cohort: str) -> list[str]:
    """The outcome tokens that exist in the vocab.

    A token that is missing but has a spelling variant in the vocab (case, spaces) is an error.
    Otherwise it is genuinely absent, typically a rare code that `min_code_count` left out of the
    vocab, so the model can never generate it: it is dropped and listed.
    """
    missing = [t for t in tokens if t not in vocab.stoi]
    if not missing:
        return tokens
    normalized = {t.upper().replace(" ", "_"): t for t in vocab.stoi}
    typos = {t: normalized[t.upper().replace(" ", "_")] for t in missing
             if t.upper().replace(" ", "_") in normalized}
    if typos:
        lines = "\n".join(f"  {t}  (did you mean {v}?)" for t, v in typos.items())
        raise ValueError(f"[{cohort}] outcome tokens with another spelling in the vocab:\n{lines}")
    kept = [t for t in tokens if t in vocab.stoi]
    if not kept:
        raise ValueError(f"[{cohort}] none of the {len(tokens)} outcome tokens is in the vocab")
    print(
        f"[{cohort}] WARNING: {len(missing)} of {len(tokens)} outcome tokens are not in the vocab "
        f"and can never be generated, so they are not part of the event: {missing}",
        file=sys.stderr,
    )
    return kept


def build_eval_set(
    labels_fp: Path, tok_dir: Path, cohort_cfg: dict, max_labels: int, seed: int
) -> pl.DataFrame:
    """The labels that belong to the tokenized split, as they are matched to its patients."""
    ds = MedsLabelledDataset(
        input_dir=tok_dir,
        labels_fp=labels_fp,
        outcome_stoken=list(cohort_cfg["outcome_stokens"]),
        include_base_stop_stokens=False,
        strict_cutoff=bool(cohort_cfg.get("strict_cutoff", False)),
    )
    rows = pl.DataFrame(ds.labels)
    if max_labels and len(rows) > max_labels:
        keep = np.sort(np.random.default_rng(seed).permutation(len(rows))[:max_labels])
        rows = rows[keep.tolist()]
    return rows.with_columns(pl.from_epoch("prediction_time", time_unit="us"))


def write_shards(eval_set: pl.DataFrame, out: Path, shard_size: int) -> list[Path]:
    n_shards = max(1, math.ceil(len(eval_set) / shard_size))
    shard_dirs = []
    for i in range(n_shards):
        shard_dir = out / "shards" / f"shard_{i:04d}"
        labels_dir = shard_dir / "labels"
        if not labels_dir.exists():
            labels_dir.mkdir(parents=True)
            eval_set[i * shard_size : (i + 1) * shard_size].write_parquet(
                labels_dir / "labels.parquet"
            )
        shard_dirs.append(shard_dir)
    return shard_dirs


def result_files(shard_dir: Path) -> list[Path]:
    return sorted((shard_dir / "results").rglob("samples_*.parquet"))


LABEL_KEYS = ["subject_id", "prediction_time"]


def read_results(shard_dir: Path) -> pl.DataFrame | None:
    files = result_files(shard_dir)
    if not files:
        return None
    return pl.concat([pl.read_parquet(f, glob=False) for f in files], how="diagonal")


def labels_left(shard_dir: Path, labels: pl.DataFrame, rep_num: int) -> pl.DataFrame:
    """The labels of a shard that do not have all their trajectories yet.

    Trajectories of a label that are only partly there (the run stopped while the label was
    generated) are removed, so that nothing is counted twice when the label is generated again.
    The trajectories of the other labels are kept as they are.
    """
    results_dir, old, new = shard_dir / "results", shard_dir / "results.old", shard_dir / "results.new"
    if old.exists():  # an earlier run stopped while it replaced the results
        shutil.rmtree(old) if results_dir.exists() else old.rename(results_dir)
    shutil.rmtree(new, ignore_errors=True)

    results = read_results(shard_dir)
    if results is None:
        return labels
    expected = labels.group_by(LABEL_KEYS).len().with_columns(expected=pl.col("len") * rep_num)
    found = results.group_by("patient_id", "prediction_time").len().rename({"patient_id": "subject_id"})
    complete = found.join(expected.drop("len"), on=LABEL_KEYS).filter(pl.col("len") == pl.col("expected"))
    complete_keys = complete.select(LABEL_KEYS)

    kept = results.join(
        complete_keys.rename({"subject_id": "patient_id"}), on=["patient_id", "prediction_time"], how="semi"
    )
    if kept.height != results.height:  # some labels are only partly there: drop their trajectories
        if kept.height:
            new.mkdir()
            kept.write_parquet(new / "samples_kept.parquet")
            results_dir.rename(old)
            new.rename(results_dir)
            shutil.rmtree(old)
        else:
            shutil.rmtree(results_dir)
    return labels.join(complete_keys, on=LABEL_KEYS, how="anti")


def run_shard(shard_dir, gpu, args, cohort_cfg, model_fp, tok_dir, include_base) -> bool:
    if (shard_dir / ".done").exists():
        return True
    labels = pl.read_parquet(shard_dir / "labels" / "labels.parquet")
    todo = labels_left(shard_dir, labels, args.rep_num)  # keeps what an earlier run finished
    if todo.height == 0:
        (shard_dir / ".done").touch()
        return True

    todo_dir = shard_dir / "labels_todo"
    shutil.rmtree(todo_dir, ignore_errors=True)
    todo_dir.mkdir()
    todo.write_parquet(todo_dir / "labels.parquet")
    attempt = len(list((shard_dir / "results").glob("attempt_*")))  # an attempt never overwrites another

    cmd = [
        sys.executable, "-m", "ethos.inference.run_inference",
        "task=meds_label",
        f"model_fp={model_fp}",
        f"input_dir={tok_dir}",
        f"output_dir={shard_dir / 'results' / f'attempt_{attempt:03d}'}",
        "output_fn=run",
        f"result_chunk_size={args.rep_num * args.flush_labels}",
        *([f"max_new_tokens={args.max_new_tokens}"] if args.max_new_tokens else []),
        *([f"kv_slide={args.kv_slide}"] if args.kv_slide else []),
        f"batch_labels={args.batch_labels}",
        f"+dataset_kwargs.labels_fp={todo_dir}",
        f"+dataset_kwargs.outcome_stoken={hydra_list(cohort_cfg['outcome_stokens'])}",
        f"+dataset_kwargs.time_limit_days={cohort_cfg['time_limit_days']}",
        f"+dataset_kwargs.include_base_stop_stokens={str(include_base).lower()}",
        f"+dataset_kwargs.strict_cutoff={str(bool(cohort_cfg.get('strict_cutoff', False))).lower()}",
        f"device={'cpu' if gpu is None else 'cuda'}",
        f"n_jobs={args.jobs_per_gpu}",
        "n_gpus=1",
        f"rep_num={args.rep_num}",
        "no_compile=true",
        f"timeout={args.timeout}",
        f"seed={args.seed}",
        f"hydra.run.dir={shard_dir / 'hydra'}",
    ]
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with open(shard_dir / "run.log", "a") as log:
        log.write(f"\n=== attempt {attempt}: {todo.height} of {labels.height} labels ===\n")
        log.flush()
        returncode = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode

    # ethos_infer exits 0 even if a worker died and the progress queue timed out, so count the
    # labels that have all their trajectories
    left = labels_left(shard_dir, labels, args.rep_num)
    if left.height == 0:
        (shard_dir / ".done").touch()
        return True
    print(
        f"  {shard_dir}: {labels.height - left.height} of {labels.height} labels done, "
        f"{left.height} left, kept for the next run (exit code {returncode}), see {shard_dir / 'run.log'}",
        file=sys.stderr,
    )
    return False


def score(
    out: Path,
    eval_set: pl.DataFrame,
    labels_all: pl.DataFrame,
    cohort_cfg: dict,
    rep_num: int,
    shard_dirs: list[Path] | None = None,
    suffix: str = "",
) -> dict:
    """Scores the labels of `eval_set`, whose trajectories are in `shard_dirs` (default: all)."""
    shard_dirs = shard_dirs or sorted((out / "shards").glob("shard_*"))
    files = [f for shard in shard_dirs for f in result_files(shard)]
    results = pl.concat([pl.read_parquet(f, glob=False) for f in files], how="diagonal")
    if results.height != eval_set.height * rep_num:
        raise RuntimeError(
            f"{results.height} trajectories for {eval_set.height} labels x {rep_num}: incomplete"
        )
    kept = results.filter(pl.col("stop_reason").is_in(STOP_REASONS_KEPT))
    outcome = list(cohort_cfg["outcome_stokens"])

    predictions = (
        kept.group_by("patient_id", "prediction_time")
        .agg(
            risk=pl.col("actual").is_in(outcome).mean(),
            n_trajectories=pl.len(),
            y=pl.col("boolean_value").first(),
        )
        .rename({"patient_id": "subject_id"})
        .with_columns(pl.from_epoch("prediction_time", time_unit="us"))
    )
    extra = [c for c in ("time_to_event",) if c in labels_all.columns]
    if extra:
        predictions = predictions.join(
            labels_all.select("subject_id", "prediction_time", *extra).unique(
                ["subject_id", "prediction_time"]
            ),
            on=["subject_id", "prediction_time"],
            how="left",
        )
    predictions.write_parquet(out / f"predictions{suffix}.parquet")

    def metrics(df: pl.DataFrame) -> dict:
        y = df["y"].to_numpy()
        if len(set(y.tolist())) < 2:
            return {"n": len(y), "auroc": None, "auprc": None}
        return {
            "n": len(y),
            "prevalence": float(y.mean()),
            "auroc": float(roc_auc_score(y, df["risk"].to_numpy())),
            "auprc": float(average_precision_score(y, df["risk"].to_numpy())),
        }

    res = {
        "labels": eval_set.height,
        "trajectories": results.height,
        "trajectories_dropped_key_error": int((results["stop_reason"] == "key_error").sum()),
        "trajectories_dropped_token_limit": int((results["stop_reason"] == "token_limit").sum()),
        "labels_without_valid_trajectory": eval_set.height - predictions.height,
        "share_ended_by_time_limit": float(
            (kept["stop_reason"] == "time_limit").mean() if kept.height else float("nan")
        ),
        "mean_risk": float(predictions["risk"].mean()),
        "rep_num": rep_num,
        "shards_scored": len(shard_dirs),
        "all_labels": metrics(predictions),
    }
    # needs the follow-up of the negatives too; a label table with only the outcome date of the
    # positives cannot give it
    if (
        "time_to_event" in predictions.columns
        and predictions.filter(~pl.col("y"))["time_to_event"].null_count() == 0
    ):
        # like ETHOS's "Reduced" readmission score: a negative only counts if the patient was
        # followed for the whole horizon, otherwise a readmission may simply not be recorded
        horizon = cohort_cfg["time_limit_days"]
        res["fully_followed_labels"] = metrics(
            predictions.filter(pl.col("y") | (pl.col("time_to_event") >= horizon))
        )
    (out / f"metrics{suffix}.json").write_text(json.dumps(res, indent=2))
    return res


def run_cohort(cohort: str, cohort_cfg: dict, args, gpus: list) -> dict:
    out = Path(args.out_dir) / cohort
    out.mkdir(parents=True, exist_ok=True)
    tok_dir = Path(args.tokenized_dir) / args.split
    labels_fp = Path(args.cohorts_dir) / cohort
    model_fp = Path(args.model_fp)

    vocab = Vocabulary.from_path(tok_dir)
    train_vocab = sorted((Path(args.tokenized_dir) / "train").glob("vocab_t*.csv"))
    split_vocab = sorted(tok_dir.glob("vocab_t*.csv"))
    if train_vocab and (not split_vocab or split_vocab[0].read_bytes() != train_vocab[0].read_bytes()):
        raise ValueError(
            f"[{cohort}] the {args.split} split was not tokenized with the train vocab. "
            f"Retokenize it with vocab={Path(args.tokenized_dir) / 'train'}."
        )
    outcome = resolve_outcome_tokens(vocab, list(cohort_cfg["outcome_stokens"]), cohort)
    cohort_cfg = {**cohort_cfg, "outcome_stokens": outcome}
    base = [str(ST.DEATH), str(ST.TIMELINE_END)]
    include_base = all(t in vocab.stoi for t in base)
    if not include_base:
        print(f"[{cohort}] {base} are not all in the vocab, so they are not used as stop tokens")

    parts = {
        "model": file_signature([model_fp]),
        "labels": file_signature(parquet_files(labels_fp)),
        "tokenized": file_signature(sorted(tok_dir.glob("*.pickle")) + sorted(tok_dir.glob("vocab_t*.csv")))
        + [str(len(list(tok_dir.glob("[0-9]*.safetensors"))))],
        "cohort": OmegaConf.to_container(OmegaConf.create(cohort_cfg)),
        "rep_num": args.rep_num, "max_labels": args.max_labels,
        "shard_size": args.shard_size, "seed": args.seed, "split": args.split,
    }
    if args.kv_slide:  # only when used, so the signature of earlier runs stays valid
        parts["kv_slide"] = args.kv_slide
    signature = json.dumps(parts, sort_keys=True, indent=1)
    sig_fp = out / "signature"
    # older versions stored only the sha1 of the compact json
    legacy = hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()
    if sig_fp.exists() and sig_fp.read_text() not in (signature, legacy):
        try:
            old = json.loads(sig_fp.read_text())
            changed = [f"{k}: {old.get(k)!r} -> {parts.get(k)!r}"[:300]
                       for k in sorted(parts) if old.get(k) != parts[k]]
        except json.JSONDecodeError:  # written by an older version, which stored only a hash
            changed = ["not known: the signature was written by an older version"]
        raise RuntimeError(
            f"[{cohort}] {out} was made with a different checkpoint, labels, tokenized data or "
            "setting. Delete it or use another --out-dir. Different:\n  " + "\n  ".join(changed)
        )

    eval_fp = out / "eval_set.parquet"
    if eval_fp.exists():
        eval_set = pl.read_parquet(eval_fp)
    else:
        print(f"[{cohort}] matching labels to the {args.split} split")
        eval_set = build_eval_set(
            labels_fp, tok_dir, cohort_cfg, args.max_labels, args.seed
        )
        eval_set.write_parquet(eval_fp)
    n_label_rows = sum(
        pl.scan_parquet(f, glob=False).select(pl.len()).collect().item()
        for f in parquet_files(labels_fp)
    )
    match_rate = eval_set.height / max(n_label_rows, 1) if not args.max_labels else 1.0
    if match_rate < 0.98:
        print(
            f"[{cohort}] WARNING: only {eval_set.height:,} of {n_label_rows:,} labels "
            f"({match_rate:.1%}) belong to patients of the {args.split} split. If these labels "
            "were cut from that split, its tokenization is incomplete or it is the wrong split.",
            file=sys.stderr,
        )
    sig_fp.write_text(signature)
    print(f"[{cohort}] {eval_set.height:,} labels, {int(eval_set['boolean_value'].sum()):,} positive")

    shard_dirs = write_shards(eval_set, out, args.shard_size)
    pending = [d for d in shard_dirs if not (d / ".done").exists()]
    print(f"[{cohort}] {len(shard_dirs) - len(pending)}/{len(shard_dirs)} shards already done")

    labels_all = read_label_table(labels_fp).with_columns(
        pl.col("prediction_time").cast(pl.Datetime("us"))
    )
    if args.partial:
        done = [d for d in shard_dirs if (d / ".done").exists()]
        if not done:
            raise RuntimeError(f"[{cohort}] no finished shard to score yet")
        labels_done = pl.concat([pl.read_parquet(d / "labels" / "labels.parquet") for d in done])
        print(f"[{cohort}] scoring the {len(done)}/{len(shard_dirs)} finished shards "
              f"({labels_done.height:,} of {eval_set.height:,} labels)")
        return score(out, labels_done, labels_all, cohort_cfg, args.rep_num, done, "_partial")

    failed = []
    lock = threading.Lock()
    started = {}
    stop_heartbeat = threading.Event()

    def heartbeat():
        """Prints the progress bar line of every running shard, which ethos_infer only writes to
        the shard's run.log."""
        while not stop_heartbeat.wait(args.progress_every):
            for shard_dir in pending:
                log = shard_dir / "run.log"
                if shard_dir.name not in started or (shard_dir / ".done").exists() or not log.exists():
                    continue
                tail = log.read_bytes()[-4096:].decode(errors="ignore")
                bars = [seg for seg in re.split(r"[\r\n]", tail) if "Progress:" in seg]
                if bars:
                    minutes = int(time.time() - started[shard_dir.name]) // 60
                    line = " ".join(re.sub(r"\|[^|]*\|", " ", bars[-1]).split())
                    print(f"[{cohort}] {shard_dir.name} {minutes} min: {line}", flush=True)

    def worker(w: int):
        for shard_dir in pending[w :: len(gpus)]:
            started[shard_dir.name] = time.time()
            try:
                ok = run_shard(shard_dir, gpus[w], args, cohort_cfg, model_fp, tok_dir, include_base)
            except Exception as e:  # an exception in a thread would otherwise vanish silently
                print(f"  {shard_dir}: {type(e).__name__}: {e}", file=sys.stderr)
                ok = False
            with lock:
                print(f"[{cohort}] {shard_dir.name} {'done' if ok else 'FAILED'}", flush=True)
                if not ok:
                    failed.append(shard_dir.name)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(len(gpus))]
    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    for t in threads:
        t.start()
    heartbeat_thread.start()
    for t in threads:
        t.join()
    stop_heartbeat.set()
    unfinished = [d.name for d in shard_dirs if not (d / ".done").exists()]
    if failed or unfinished:
        raise RuntimeError(
            f"[{cohort}] unfinished shards: {sorted(set(failed) | set(unfinished))}. "
            "Rerun to retry them."
        )

    return score(out, eval_set, labels_all, cohort_cfg, args.rep_num)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-fp", required=True)
    parser.add_argument("--cohorts-dir", required=True, help="folder with one folder per cohort")
    parser.add_argument("--tokenized-dir", required=True, help="has the tokenized splits")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--config", nargs="+",
        default=[str(p) for p in sorted(Path(__file__).parent.glob("*.yaml"))],
        help="yaml file(s) with the outcome definition per cohort (default: all next to the script)",
    )
    parser.add_argument("--cohorts", nargs="+", help="default: every cohort in the config")
    parser.add_argument("--split", default="held_out", help="tokenized split the labels belong to")
    parser.add_argument("--max-labels", type=int, default=0, help="0: all labels of the cohort")
    parser.add_argument("--rep-num", type=int, default=10, help="trajectories per label")
    parser.add_argument("--shard-size", type=int, default=500, help="labels per ethos_infer run")
    parser.add_argument(
        "--gpu-ids",
        nargs="*",
        help="GPU ids of the machine; default: those in CUDA_VISIBLE_DEVICES, else all, else the CPU",
    )
    parser.add_argument("--jobs-per-gpu", type=int, default=2, help="processes sharing a GPU")
    parser.add_argument(
        "--flush-labels", type=int, default=20,
        help="results are written every this many labels, so a stopped shard keeps them",
    )
    parser.add_argument(
        "--kv-slide", type=int, default=0,
        help="generate with a cache of keys and values, which is much faster. When the context "
        "window is full it slides by this many tokens and the rest of it is computed again, so the "
        "window is between this many tokens shorter than the full one and the full one (e.g. 256 "
        "for a 2048 window). 0: compute the whole window for every token, exact but slow. It "
        "changes the trajectories slightly, so it is part of the run's signature",
    )
    parser.add_argument(
        "--batch-labels", type=int, default=1,
        help="labels generated together when their timelines have the same length, which keeps "
        "the GPU busier (rep-num x batch-labels sequences at once). Does not change the results",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=0,
        help="stop a trajectory after this many generated tokens (0: no limit). Such a trajectory "
        "is not scored, it is counted in trajectories_dropped_token_limit. Not part of the run's "
        "signature, so it can be added when a run is resumed",
    )
    parser.add_argument("--timeout", type=int, default=3600, help="seconds to wait for a result")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--partial", action="store_true",
        help="run nothing, score the shards that are finished (writes *_partial files)",
    )
    parser.add_argument(
        "--progress-every", type=int, default=60, help="seconds between progress lines of running shards"
    )
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)  # show progress in a redirected log right away
    sys.stderr.reconfigure(line_buffering=True)

    signal.signal(signal.SIGHUP, signal.SIG_IGN)  # a dropped SSH session must not stop the run

    config = {}
    for config_fp in args.config:
        for name, value in OmegaConf.to_container(OmegaConf.load(config_fp)).items():
            if name in config:
                raise SystemExit(f"Cohort {name} is defined in more than one config file")
            config[name] = value
    cohorts = args.cohorts or list(config)
    unknown = [c for c in cohorts if c not in config]
    if unknown:
        raise SystemExit(f"Not in {args.config}: {unknown}. Add their outcome definition first.")

    if args.gpu_ids is not None:
        gpus = [int(g) for g in args.gpu_ids] or [None]
    elif os.environ.get("CUDA_VISIBLE_DEVICES"):
        # each shard gets one of these ids as its own CUDA_VISIBLE_DEVICES, so they have to be the
        # ids of the machine (export CUDA_VISIBLE_DEVICES=0,2 means GPUs 0 and 2, not 0 and 1)
        gpus = [int(g) if g.strip().isdigit() else g.strip()
                for g in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if g.strip()]
    else:
        import torch

        gpus = list(range(torch.cuda.device_count())) or [None]
    print(f"Model: {args.model_fp}\nCohorts: {cohorts}\nGPUs: {gpus} (None = CPU)")

    summary, failures = [], []
    for cohort in cohorts:
        try:
            res = run_cohort(cohort, config[cohort], args, gpus)
        except Exception as e:  # keep going with the other cohorts
            failures.append(cohort)
            print(f"[{cohort}] FAILED: {e}", file=sys.stderr)
            continue
        summary.append({"cohort": cohort, **{k: v for k, v in res.items() if not isinstance(v, dict)},
                        **{f"{k}_{m}": v for k, d in res.items() if isinstance(d, dict) for m, v in d.items()}})

    if summary:
        df = pl.from_dicts(summary, infer_schema_length=None)
        df.write_csv(Path(args.out_dir) / ("summary_partial.csv" if args.partial else "summary.csv"))
        shown = [c for c in (
            "cohort", "labels", "all_labels_prevalence", "all_labels_auroc", "all_labels_auprc",
            "fully_followed_labels_auroc", "mean_risk", "trajectories_dropped_key_error",
            "labels_without_valid_trajectory") if c in df.columns]
        with pl.Config(tbl_cols=-1, tbl_width_chars=200):
            print(df.select(shown))
        print(f"Full table: {Path(args.out_dir) / ('summary_partial.csv' if args.partial else 'summary.csv')}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
