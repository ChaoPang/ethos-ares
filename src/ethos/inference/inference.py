from copy import copy
from pathlib import Path
from queue import Queue

import torch as th

from ..constants import SpecialToken as ST
from ..utils import load_model_checkpoint, setup_torch
from ..vocabulary import Vocabulary
from .constants import Reason, Task
from .utils import (
    KVCacheSampler,
    create_loader,
    get_dataset_cls,
    get_next_token,
    get_token_time,
    group_by_length,
)


def spawn_inference_worker(
    job_queue: Queue,
    model_fp: str | Path,
    task: Task,
    dataset_kwargs: dict,
    progress_queue: Queue,
    temperature: float = 1.0,
    rep_num: int = 1,
    device: str = "cuda",
    no_compile: bool = False,
    save_generated_tokens: bool = False,
    max_new_tokens: int | None = None,
    kv_slide: int | None = None,
    batch_labels: int = 1,
):
    if "cuda" in device:
        th.cuda.set_device(device)
        th.set_float32_matmul_precision("high")
    autocast_context = setup_torch(device, dtype="bfloat16" if "cuda" in device else "float32")

    model, _ = load_model_checkpoint(model_fp, map_location=device)
    model.to(device)
    model = th.compile(model, disable=no_compile)

    dataset_cls = get_dataset_cls(task)
    # TODO: Validate if the vocab in the dataset is the same as in the model
    dataset = dataset_cls(**dataset_kwargs)

    max_timeline_size = dataset_kwargs["n_positions"]
    ctx_size = dataset.context_size
    vocab: Vocabulary = dataset.vocab

    data_loader = create_loader(job_queue, dataset)

    stop_stokens = dataset.stop_stokens
    stop_tokens = th.tensor(vocab.encode(stop_stokens), dtype=th.long)

    time_limit = th.tensor(dataset.time_limit.total_seconds() * 1e6)

    for group in group_by_length(data_loader, batch_labels):
        ground_truths = [ground_truth for _, ground_truth in group]
        ctx = None
        if isinstance(group[0][0], tuple):  # one example, with the context of an encoder-decoder
            ctx, timeline = tuple(t.to(device, non_blocking=True) for t in group[0][0])
            ctx = ctx.repeat(rep_num, 1)
            timeline = timeline.repeat(rep_num, 1)
        else:
            timeline = th.cat(
                [tl.to(device, non_blocking=True).repeat(rep_num, 1) for tl, _ in group]
            )
        sampler = None
        if kv_slide:
            if ctx is not None:
                raise ValueError("kv_slide supports decoder-only models only")
            sampler = KVCacheSampler(model, max_timeline_size, ctx_size, kv_slide, temperature)
        row_ids = th.arange(timeline.size(0))  # the row of the group that each row stands for

        gen_token_num, offset = 0, 0
        gen_times = th.zeros(timeline.size(0), dtype=th.float64)
        generated_tokens = [] if save_generated_tokens else None
        while timeline.size(0):
            if task == Task.SOFA_PREDICTION and gen_token_num == 1:
                # append a sofa token to the timeline and continue generating
                next_token = th.tensor(
                    [vocab.encode([ST.SOFA])], device=timeline.device, dtype=th.long
                )
                next_token = next_token.repeat(timeline.size(0), 1)
            else:
                with autocast_context:
                    if sampler is not None:
                        next_token, probs, timeline = sampler.next_token(timeline)
                    else:
                        next_token, probs = get_next_token(
                            model, timeline, ctx=ctx, return_probs=True, temperature=temperature
                        )

            if generated_tokens is not None:
                generated_tokens.append(next_token)

            if sampler is not None:
                timeline = th.cat((timeline, next_token), dim=1)  # the sampler slides the window
            else:
                if not offset and timeline.size(1) == max_timeline_size:
                    offset = 1

                if ctx is not None:
                    new_timeline = (timeline[:, offset:], next_token)
                else:
                    new_timeline = (
                        timeline[:, :ctx_size],
                        timeline[:, ctx_size + offset :],
                        next_token,
                    )
                timeline = th.cat(new_timeline, dim=1)

            gen_token_num += 1

            new_token = next_token.cpu().view(-1)
            gen_times += get_token_time(new_token, vocab)

            completed_this_iter = th.isin(new_token, stop_tokens) | (gen_times > time_limit)

            if max_new_tokens is not None and gen_token_num >= max_new_tokens:
                completed_this_iter[:] = True  # the budget of tokens is used up

            if (task == Task.DRG_PREDICTION) or (
                task == Task.SOFA_PREDICTION and gen_token_num == 3
            ):
                completed_this_iter[:] = True

            if not completed_this_iter.any():
                continue

            for i in th.nonzero(completed_this_iter):
                stop_reason = Reason.GOT_TOKEN
                actual_token = next_token[i].item()

                token_time = gen_times[i]
                if token_time > time_limit:
                    stop_reason = Reason.TIME_LIMIT
                elif not th.isin(new_token[i], stop_tokens).item():
                    stop_reason = Reason.TOKEN_LIMIT

                if th.isinf(token_time):
                    actual_stoken = str(actual_token)
                    stop_reason = Reason.KEY_ERROR
                    token_time = None
                else:
                    actual_stoken = vocab.decode(actual_token)
                    token_time = round(token_time.item())

                gt = copy(ground_truths[int(row_ids[i]) // rep_num])

                results = {
                    "expected": gt.pop("expected"),
                    "actual": actual_stoken,
                    "stop_reason": stop_reason,
                    "actual_prob": probs[i, actual_token].item(),
                    **dict(zip(stop_stokens, probs[i, stop_tokens].tolist())),
                    "true_token_time": gt.pop("true_token_time"),
                    "token_time": token_time,
                    "true_token_dist": gt.pop("true_token_dist"),
                    "token_dist": gen_token_num,
                    **gt,
                }

                if generated_tokens is not None:
                    results["generated_tokens"] = [tokens[i].item() for tokens in generated_tokens]

                progress_queue.put(results)

            if completed_this_iter.all():
                break

            not_completed_mask = ~completed_this_iter
            timeline = timeline[not_completed_mask, :]
            gen_times = gen_times[not_completed_mask]
            row_ids = row_ids[not_completed_mask]
            if sampler is not None:
                sampler.select(not_completed_mask)
            if generated_tokens is not None:
                generated_tokens = [tokens[not_completed_mask, :] for tokens in generated_tokens]
