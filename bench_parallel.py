"""Benchmark: N parallel jobs on one Generator, aggregate + per-stream tok/s.

Usage:
    PYTHONPATH=. .venv/bin/python bench_parallel.py --model <dir> [--draft <same dir>] \
        --jobs N --runs R --label <name>
"""
import sys, os, time, argparse, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from exllamav3 import (
    Config, Model, Cache, CacheLayer_quant, Tokenizer, Generator, Job,
)
from exllamav3.generator.sampler.presets import ArgmaxSampler

PROMPT = (
    "Explain in detail, in about 300 words, how the RSA cryptosystem works "
    "and why factoring large numbers makes it hard to break."
)
CACHE_TOKENS = 16384
MAX_NEW_TOKENS = 320

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--label", default="par")
    args = ap.parse_args()

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    draft_model = draft_cache = None
    if args.draft:
        assert os.path.realpath(args.draft) == os.path.realpath(args.model)
        draft_model = Model.from_config(config, component="mtp")
        draft_cache = Cache(draft_model, max_num_tokens=CACHE_TOKENS, max_batch_size=args.jobs)
        draft_model.load()
    cache = Cache(
        model,
        max_num_tokens=CACHE_TOKENS,
        layer_type=CacheLayer_quant, k_bits=8, v_bits=8,
        max_batch_size=args.jobs,
        max_history=4 if draft_model else 0,
    )
    model.load()
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(
        model=model, cache=cache, tokenizer=tokenizer,
        draft_model=draft_model, draft_cache=draft_cache,
    )
    input_ids = tokenizer.hf_chat_template(
        [{"role": "user", "content": PROMPT}], enable_thinking=False,
    )

    aggs, streams = [], []
    for run in range(args.runs):
        jobs = [
            Job(
                input_ids=input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                stop_conditions=[tokenizer.eos_token_id, "<|im_end|>"],
                identifier=i,
                sampler=ArgmaxSampler(),
            )
            for i in range(args.jobs)
        ]
        for j in jobs:
            generator.enqueue(j)
        counts = {i: 0 for i in range(args.jobs)}
        t_first = {}
        t0 = time.perf_counter()
        while generator.num_remaining_jobs():
            for result in generator.iterate():
                i = result["identifier"]
                counts[i] += 1
                if counts[i] == 1:
                    t_first[i] = time.perf_counter() - t0
        wall = time.perf_counter() - t0
        per = [counts[i] / wall for i in range(args.jobs)]
        agg = sum(counts.values()) / wall
        aggs.append(agg)
        streams.append(statistics.median(per))
        print(f"[{args.label}] run {run + 1}: total {sum(counts.values())} tok in {wall:.1f}s | "
              f"aggregate {agg:.1f} tok/s | per-stream median {statistics.median(per):.1f} "
              f"(min {min(per):.1f} max {max(per):.1f}) | last TTFT {max(t_first.values()):.2f}s")

    print(f"[{args.label}] SUMMARY aggregate: median {statistics.median(aggs):.1f} tok/s  "
          f"(runs: {[f'{a:.1f}' for a in aggs]})  | per-stream median {statistics.median(streams):.1f} tok/s")

if __name__ == "__main__":
    main()
