"""Benchmark: Qwen3.8-27B EXL3 decode speed, median of N runs.

Usage:
    PYTHONPATH=. .venv/bin/python bench_qwen38.py --model <dir> [--runs 3] [--label <name>]
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
    ap.add_argument("--draft", default=None, help="draft/MTP model dir")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--label", default="bench")
    args = ap.parse_args()

    assert torch.version.hip, "expected ROCm build of torch"
    props = torch.cuda.get_device_properties(0)
    print(f"[{args.label}] torch {torch.__version__} hip {torch.version.hip} dev {props.gcnArchName}")

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    draft_model = draft_cache = None
    if args.draft:
        # MTP drafting: draft dir == model dir, draft = embedded MTP head
        assert os.path.realpath(args.draft) == os.path.realpath(args.model), \
            "--draft must point at the same SC checkpoint dir (component=mtp)"
        draft_model = Model.from_config(config, component="mtp")
        draft_cache = Cache(draft_model, max_num_tokens=CACHE_TOKENS, max_batch_size=1)
        draft_model.load()
    cache = Cache(
        model,
        max_num_tokens=CACHE_TOKENS,
        layer_type=CacheLayer_quant, k_bits=8, v_bits=8,
        max_batch_size=1,
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
    n_prompt = input_ids.shape[-1]
    print(f"[{args.label}] prompt tokens: {n_prompt}")

    speeds, ttfts, ntoks = [], [], []
    for run in range(args.runs):
        job = Job(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            stop_conditions=[tokenizer.eos_token_id, "<|im_end|>"],
            identifier=0,
            sampler=ArgmaxSampler(),
        )
        generator.enqueue(job)
        ntok, t_first, t_last = 0, None, None
        t0 = time.perf_counter()
        while generator.num_remaining_jobs():
            for _ in generator.iterate():
                ntok += 1
                if ntok == 1:
                    t_first = time.perf_counter()
                t_last = time.perf_counter()
        gen_time = t_last - t_first
        tps = (ntok - 1) / gen_time if ntok > 1 else 0.0
        speeds.append(tps)
        ttfts.append(t_first - t0)
        ntoks.append(ntok)
        print(f"[{args.label}] run {run + 1}: {ntok} tok, TTFT {t_first - t0:.2f}s, decode {tps:.1f} tok/s")

    print(f"[{args.label}] SUMMARY decode tok/s: median {statistics.median(speeds):.1f}  "
          f"min {min(speeds):.1f}  max {max(speeds):.1f}  (runs: {[f'{s:.1f}' for s in speeds]})")
    print(f"[{args.label}] SUMMARY TTFT: median {statistics.median(ttfts):.2f}s")

if __name__ == "__main__":
    main()
