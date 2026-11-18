"""Smoke test: load Qwen3.8-27B EXL3 3.00bpw on ROCm gfx1200, generate one answer.

Run from the repo root:
    PYTHONPATH=. .venv/bin/python smoke_qwen38.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from exllamav3 import (
    Config, Model, Cache, CacheLayer_quant, Tokenizer, Generator, Job, TopPSampler,
)
try:
    from exllamav3.generator.sampler.presets import ArgmaxSampler
    SAMPLER = ArgmaxSampler()
except ImportError:
    SAMPLER = TopPSampler(temperature=0.0, top_p=0.95, temperature_last=True)

MODEL_DIR = os.path.expanduser(
    "~/Projects/exllamav3-data/models/Qwen3.8-27B-exl3-3.00bpw"
)
CACHE_TOKENS = 16384
MAX_NEW_TOKENS = 400

PROMPT = (
    "Give one short paragraph: what is the Collatz conjecture, and why is it hard to prove?"
)

def format_prompt(p: str) -> str:
    # Model's own chat_template.jinja via Tokenizer.hf_chat_template
    # (transformers renders it; enable_thinking=False prefills the empty
    # think block). Placeholder kept for reference; unused.
    return f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"

def main():
    assert torch.version.hip, "expected ROCm build of torch"
    props = torch.cuda.get_device_properties(0)
    print(f"torch {torch.__version__}  hip {torch.version.hip}  device {props.gcnArchName}")
    free0 = torch.cuda.mem_get_info()[0] / 2**30
    print(f"VRAM free before load: {free0:.2f} GiB")

    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    cache = Cache(
        model,
        max_num_tokens=CACHE_TOKENS,
        layer_type=CacheLayer_quant, k_bits=8, v_bits=8,
        max_batch_size=1,
    )
    t0 = time.time()
    model.load(progressbar=True)
    print(f"load time: {time.time() - t0:.1f}s")
    free1 = torch.cuda.mem_get_info()[0] / 2**30
    print(f"VRAM free after load: {free1:.2f} GiB (used {free0 - free1:.2f} GiB)")

    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

    job = Job(
        input_ids=tokenizer.hf_chat_template(
            [{"role": "user", "content": PROMPT}], enable_thinking=False,
        ),
        max_new_tokens=MAX_NEW_TOKENS,
        stop_conditions=[tokenizer.eos_token_id, "<|im_end|>"],
        identifier=0,
    )
    generator.enqueue(job)

    out, ntok = "", 0
    t0 = time.time()
    while generator.num_remaining_jobs():
        for result in generator.iterate():
            text = result.get("text", "")
            out += text
            ntok += 1
    dt = time.time() - t0
    print(f"\n--- response ({ntok} tok, {ntok / dt:.1f} tok/s) ---")
    print(out.strip())
    print(f"VRAM free after gen: {torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB")

if __name__ == "__main__":
    main()
