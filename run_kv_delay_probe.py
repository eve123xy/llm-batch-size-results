#!/usr/bin/env python3
"""Test whether a delay between agent turns changes resumed-turn cost.

The experiment fixes the question, retrieved passages, model, concurrency, and
generation settings.  It randomizes the order of delay conditions, enables
vLLM automatic prefix caching, and records both whole-batch and turn-2 timing
and energy from 10-ms NVML power samples.
"""
import argparse
import asyncio
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

MODEL = "/scratch/xl5874/hf/hub/models--hugging-quants--Meta-Llama-3.1-8B-Instruct-AWQ-INT4/snapshots/db1f81ad4b8c7e39777509fac66c652eb0a52f91"
CORPUS = Path("/scratch/xl5874/AI-Inference/prompt_diversity_probe/data/alice_wonderland.txt")
QUESTION = "Why does the Cheshire Cat say that everyone in Wonderland is mad, and what happens to the Cat at the end of its conversation with Alice?"
SYSTEM = "You answer questions using the supplied local-book text. First state a short search query. After you receive text, answer only from that text and say when the evidence is insufficient."
sys.path.insert(0, "/scratch/xl5874/agent_probe_alice_20260908")
from power_sampler import PowerSampler


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def passages(text):
    chunks = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    query = {"cheshire", "cat", "mad", "disappear", "grin", "alice"}
    ranked = sorted(enumerate(chunks), key=lambda x: (-len(query & set(re.findall(r"[a-z]+", x[1].lower()))), x[0]))
    return [{"id": i, "text": p} for i, p in ranked[:3]]


def encode(tokenizer, messages):
    return tokenizer.encode(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), add_special_tokens=False)


async def generate(engine, prompt_ids, sampling, request_id):
    tokens = []
    async for update in engine.generate({"prompt_token_ids": prompt_ids}, sampling, request_id):
        if update.outputs:
            tokens.extend(update.outputs[0].token_ids)
    return tokens


def load_power(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def power_at(rows, timestamp):
    if timestamp <= rows[0]["t_sample_monotonic"]:
        return rows[0]["power_w"]
    for left, right in zip(rows, rows[1:]):
        if left["t_sample_monotonic"] <= timestamp <= right["t_sample_monotonic"]:
            span = right["t_sample_monotonic"] - left["t_sample_monotonic"]
            fraction = (timestamp - left["t_sample_monotonic"]) / span
            return left["power_w"] + fraction * (right["power_w"] - left["power_w"])
    return rows[-1]["power_w"]


def energy_j(rows, start, end):
    if end <= start:
        return 0.0
    points = [(start, power_at(rows, start))]
    points += [(r["t_sample_monotonic"], r["power_w"]) for r in rows if start < r["t_sample_monotonic"] < end]
    points += [(end, power_at(rows, end))]
    return sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(points, points[1:]))


async def main(args):
    from transformers import AutoTokenizer
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    import pynvml

    root = Path(args.results)
    root.mkdir(parents=True, exist_ok=True)
    (root / "power").mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tool_result = passages(CORPUS.read_text())
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=MODEL, quantization="awq_marlin", tensor_parallel_size=1,
        gpu_memory_utilization=.85, max_model_len=4096,
        enable_prefix_caching=True, disable_log_stats=True,
    ))
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    conditions = [(delay, rep) for rep in range(args.repeats) for delay in args.delays]
    random.Random(args.seed).shuffle(conditions)
    config = {"workload": "KV-cache delay probe", "N": args.n, "delays_s": args.delays,
              "repeats": args.repeats, "seed": args.seed, "power_sample_interval_s": .01,
              "enable_prefix_caching": True, "fixed_retrieved_passages": len(tool_result),
              "model": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"}
    dump(root / "config.json", config)
    sampling = SamplingParams(max_tokens=128, temperature=0.0, output_kind=RequestOutputKind.DELTA)
    rows = []
    for ordinal, (delay, rep) in enumerate(conditions):
        batch_id = f"delay{delay:g}s_rep{rep:02d}"
        power_path = root / "power" / f"{batch_id}.jsonl"
        sampler = PowerSampler(handle, .01, str(power_path))
        messages = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION}] for _ in range(args.n)]
        prompts_1 = [encode(tokenizer, message) for message in messages]
        sampler.start()
        t_start = time.monotonic()
        t1_start = time.monotonic()
        outputs_1 = await asyncio.gather(*[generate(engine, prompt, sampling, f"{batch_id}-a{i}-t1") for i, prompt in enumerate(prompts_1)])
        t1_end = time.monotonic()
        tool_start = time.monotonic()
        prompts_2 = []
        for message, output in zip(messages, outputs_1):
            resumed = message + [
                {"role": "assistant", "content": tokenizer.decode(output, skip_special_tokens=True)},
                {"role": "user", "content": "Tool results from search_corpus and read_passage:\n" + json.dumps(tool_result) + "\nAnswer the question."},
            ]
            prompts_2.append(encode(tokenizer, resumed))
        tool_end = time.monotonic()
        delay_start = time.monotonic()
        await asyncio.sleep(delay)
        delay_end = time.monotonic()
        t2_start = time.monotonic()
        outputs_2 = await asyncio.gather(*[generate(engine, prompt, sampling, f"{batch_id}-a{i}-t2") for i, prompt in enumerate(prompts_2)])
        t2_end = time.monotonic()
        samples = sampler.stop()
        rows.append({"batch_id": batch_id, "ordinal": ordinal, "delay_s": delay, "repeat": rep, "N": args.n,
                     "samples": samples, "input_tokens_turn1": len(prompts_1[0]), "input_tokens_turn2": len(prompts_2[0]),
                     "output_tokens_turn1_mean": sum(map(len, outputs_1)) / args.n,
                     "output_tokens_turn2_mean": sum(map(len, outputs_2)) / args.n,
                     "batch_ms": (t2_end - t_start) * 1000, "turn1_ms": (t1_end - t1_start) * 1000,
                     "tool_ms": (tool_end - tool_start) * 1000, "wait_ms": (delay_end - delay_start) * 1000,
                     "turn2_ms": (t2_end - t2_start) * 1000, "t_start": t_start, "t_turn2_start": t2_start,
                     "t_end": t2_end, "power_file": str(power_path)})
        print(batch_id, f"turn2_ms={(t2_end-t2_start)*1000:.1f}", flush=True)
    for row in rows:
        samples = load_power(Path(row["power_file"]))
        row["energy_j"] = energy_j(samples, row["t_start"], row["t_end"])
        row["turn2_energy_j"] = energy_j(samples, row["t_turn2_start"], row["t_end"])
    dump(root / "batches.json", rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["delay_s"]].append(row)
    summary = [{"delay_s": delay, "runs": len(group),
                "batch_ms_mean": sum(x["batch_ms"] for x in group) / len(group),
                "turn2_ms_mean": sum(x["turn2_ms"] for x in group) / len(group),
                "energy_j_mean": sum(x["energy_j"] for x in group) / len(group),
                "turn2_energy_j_mean": sum(x["turn2_energy_j"] for x in group) / len(group)}
               for delay, group in sorted(grouped.items())]
    dump(root / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--delays", type=float, nargs="+", default=[0, .25, 1, 5])
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260909)
    asyncio.run(main(parser.parse_args()))
