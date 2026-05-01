#!/usr/bin/env python3
"""Cross-validate our vLLM inference against honest_b200_h200.jsonl using distance2.

Reads N items from honest_b200_h200.jsonl, replays each prompt through our vLLM
with the SAME sampling params (temp=0.7, seed=1, top_k=40, top_p=0.95) and the
SAME enforced_tokens (so our logprobs are computed on identical token sequence).

Computes per-item distance2 and reports mean / median / p95.

Gate from gonka: mean L2 (distance2) <= 0.2; honest exp2 baseline mean = 0.0232.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import requests

sys.stdout.reconfigure(line_buffering=True)


def token_distance2_pos(inf_lp: dict, val_lp: dict) -> tuple[float, int]:
    """Per-position distance2 — exact gonka formula."""
    if not val_lp:
        return float(len(inf_lp)), 0
    sorted_vals = sorted(val_lp.values())
    if len(sorted_vals) >= 2:
        min1, min2 = sorted_vals[0], sorted_vals[1]
    else:
        min1 = sorted_vals[0]
        min2 = min1 - 1.0
    dist = 0.0
    n_matches = 0
    for token, inf_logprob in inf_lp.items():
        if token in val_lp:
            val_logprob = val_lp[token]
            n_matches += 1
        else:
            val_logprob = min1 - (min2 - min1)
        denom = 1e-10 + abs(inf_logprob) + abs(val_logprob)
        dist += abs(inf_logprob - val_logprob) / denom / 2.0
    return dist, n_matches


def distance2(inf_results: list, val_results: list, top_k: int) -> tuple[float, int]:
    """Smoothed normalized distance: (sum + 1) / (max(100, n_pos) * top_k + 1)."""
    n = min(len(inf_results), len(val_results))
    total_dist = 0.0
    total_matches = 0
    for i in range(n):
        d, m = token_distance2_pos(inf_results[i]["logprobs"], val_results[i]["logprobs"])
        total_dist += d
        total_matches += m
    return (total_dist + 1.0) / (max(100, n) * top_k + 1.0), total_matches


def call_vllm(url: str, prompt: str, enforced_tokens: list[str] | None) -> dict:
    """Send a single chat completion request with optional enforced_tokens."""
    payload = {
        "model": "moonshotai/Kimi-K2.6",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.7,
        "seed": 1,
        "logprobs": True,
        "top_logprobs": 4,
        "skip_special_tokens": False,
        "repetition_penalty": 1.2,
        "top_k": 40,
        "top_p": 0.95,
    }
    if enforced_tokens:
        payload["enforced_tokens"] = {
            "tokens": [{"token": tok} for tok in enforced_tokens],
        }
    r = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


def extract_results(content_lps: list) -> list:
    """Normalize chat-completion logprobs into the gonka-style results list."""
    results = []
    for pos in content_lps:
        token_str = pos["token"]
        lp = {token_str: pos["logprob"]}
        for tl in pos.get("top_logprobs", []):
            lp[tl["token"]] = tl["logprob"]
        results.append({"token": token_str, "logprobs": lp})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5001")
    ap.add_argument("--honest", required=True, help="path to honest_b200_h200.jsonl")
    ap.add_argument("--n", type=int, default=50, help="how many items to revalidate")
    ap.add_argument("--output", required=True, help="output report json")
    ap.add_argument("--top-k", type=int, default=4)
    args = ap.parse_args()

    print(f"Loading honest dataset from {args.honest} (n={args.n})...", flush=True)
    items = []
    with open(args.honest) as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            items.append(json.loads(line))
    print(f"Loaded {len(items)} items", flush=True)

    distances = []
    per_item = []
    n_token_mismatches = 0
    t0 = time.time()
    for i, item in enumerate(items):
        prompt = item["prompt"]
        # Strip alpaca framing if present so chat template doesn't double-wrap.
        # Actually the honest dataset stores the alpaca-style prompt verbatim;
        # we replay it as the user message — vLLM will apply chat template.
        honest_results = item["inference_result"]["results"]
        honest_tokens = [r["token"] for r in honest_results]

        try:
            resp = call_vllm(args.url, prompt, enforced_tokens=honest_tokens)
            content_lps = resp["choices"][0]["logprobs"]["content"]
            our_results = extract_results(content_lps)
            our_tokens = [r["token"] for r in our_results]
            mismatches = sum(1 for a, b in zip(our_tokens, honest_tokens) if a != b)
            d, n_matches = distance2(our_results, honest_results, args.top_k)
            distances.append(d)
            per_item.append({
                "index": i,
                "language": item.get("language"),
                "n_tokens_honest": len(honest_results),
                "n_tokens_ours": len(our_results),
                "token_mismatches": mismatches,
                "distance2": d,
                "n_logprob_matches": n_matches,
            })
            if mismatches > 0:
                n_token_mismatches += 1
            if (i + 1) % 5 == 0 or i < 3:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(items) - i - 1) / rate if rate else 0
                print(f"  [{i+1}/{len(items)}] lang={item.get('language')} d={d:.4f} mismatches={mismatches} | {rate:.2f} req/s | ETA {eta:.0f}s", flush=True)
        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}", flush=True)
            per_item.append({"index": i, "error": str(e)})

    if not distances:
        print("ERROR: no distances computed", flush=True)
        sys.exit(1)

    distances.sort()
    n = len(distances)
    mean = sum(distances) / n
    median = distances[n // 2]
    p05 = distances[max(0, int(n * 0.05))]
    p95 = distances[min(n - 1, int(n * 0.95))]

    elapsed = time.time() - t0
    report = {
        "n": n,
        "elapsed_sec": elapsed,
        "distance2_mean": mean,
        "distance2_median": median,
        "distance2_p05": p05,
        "distance2_p95": p95,
        "distance2_min": min(distances),
        "distance2_max": max(distances),
        "items_with_token_mismatch": n_token_mismatches,
        "honest_baseline_mean": 0.0232,
        "gate_mean_le_0.2": mean <= 0.2,
        "soft_pass_mean_le_0.05": mean <= 0.05,
        "per_item": per_item,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78, flush=True)
    print(f"  DISTANCE2 SUMMARY  ({n} items, {elapsed:.0f}s)", flush=True)
    print("=" * 78, flush=True)
    print(f"  mean   = {mean:.4f}    (honest exp2 baseline: 0.0232)", flush=True)
    print(f"  median = {median:.4f}", flush=True)
    print(f"  p05    = {p05:.4f}", flush=True)
    print(f"  p95    = {p95:.4f}", flush=True)
    print(f"  min    = {min(distances):.4f}    max = {max(distances):.4f}", flush=True)
    print(f"  token mismatches: {n_token_mismatches}/{n}", flush=True)
    print(f"  GATE (mean <= 0.2):       {'PASS' if mean <= 0.2 else 'FAIL'}", flush=True)
    print(f"  SOFT (mean <= 0.05):      {'PASS' if mean <= 0.05 else 'FAIL'}", flush=True)
    print("=" * 78, flush=True)
    print(f"\nReport saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
