#!/usr/bin/env python3
"""Serving benchmark for DeepSeek-V4-Flash-0731: DSpark on/off A/B.

Mirrors the four scenario shapes of the earlier compressa-perf runs
(s1..s4: prompt length x concurrency x decode length) but measures them
directly off the streaming OpenAI endpoint, so both arms of the A/B are
produced by the identical instrument.

Metrics per scenario: TTFT (mean/p95), end-to-end latency, TPOT,
output tokens/s aggregate, request throughput, failures.
"""
import argparse
import asyncio
import json
import random
import statistics
import string
import time

SCENARIOS = [
    # name, prompt_chars, num_tasks, concurrency, max_tokens
    ("s1_long_prompt_sequential_short_decode", 20000, 5, 1, 300),
    ("s2_short_prompt_high_concurrency", 2000, 200, 30, 300),
    ("s3_very_long_sequential_long_decode", 45000, 5, 1, 1000),
    ("s4_very_long_max_concurrency", 45000, 40, 20, 1000),
]

WORDS = [
    "system", "network", "protocol", "memory", "kernel", "vector", "matrix",
    "cluster", "gradient", "latency", "throughput", "cache", "token", "weight",
    "inference", "quantise", "schedule", "pipeline", "attention", "expert",
]


def make_prompt(nchars, rng):
    out = []
    n = 0
    while n < nchars:
        w = rng.choice(WORDS)
        out.append(w)
        n += len(w) + 1
    return " ".join(out) + "\n\nSummarise the text above in detail:\n"


async def one_request(session, url, model, prompt, max_tokens, results):
    import aiohttp  # noqa

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    ntok = 0
    nchunk = 0
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                body = (await resp.text())[:200]
                results.append({"ok": False, "err": f"http{resp.status}:{body}"})
                return
            async for raw in resp.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except Exception:
                    continue
                ch = chunk.get("choices") or []
                if ch and ch[0].get("text"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    nchunk += 1
                # Speculative decoding packs several tokens into one SSE chunk,
                # so chunks are NOT tokens. Take the authoritative count.
                u = chunk.get("usage")
                if u and u.get("completion_tokens"):
                    ntok = u["completion_tokens"]
    except Exception as exc:  # noqa
        results.append({"ok": False, "err": repr(exc)[:200]})
        return
    lat = time.perf_counter() - t0
    if ttft is None or nchunk == 0:
        results.append({"ok": False, "err": "no tokens"})
        return
    results.append(
        {
            "ok": True,
            "ttft": ttft,
            "latency": lat,
            "ntok": ntok or nchunk,
            "nchunk": nchunk,
            "tpot": (lat - ttft) / max(1, (ntok or nchunk) - 1),
        }
    )


async def run_scenario(base_url, model, name, chars, ntasks, conc, max_tokens, seed):
    import aiohttp

    rng = random.Random(seed)
    prompts = [make_prompt(chars, rng) for _ in range(min(ntasks, 50))]
    url = base_url.rstrip("/") + "/v1/completions"
    results = []
    sem = asyncio.Semaphore(conc)

    timeout = aiohttp.ClientTimeout(total=1800)
    conn = aiohttp.TCPConnector(limit=conc + 5)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:

        async def guarded(i):
            async with sem:
                await one_request(
                    session, url, model, prompts[i % len(prompts)], max_tokens, results
                )

        t0 = time.perf_counter()
        await asyncio.gather(*[guarded(i) for i in range(ntasks)])
        wall = time.perf_counter() - t0

    ok = [r for r in results if r["ok"]]
    fails = [r for r in results if not r["ok"]]
    if not ok:
        return {
            "scenario": name,
            "failed_requests": len(fails),
            "errors": fails[:3],
            "wall_s": round(wall, 2),
        }

    def pct(xs, q):
        xs = sorted(xs)
        k = min(len(xs) - 1, int(round(q * (len(xs) - 1))))
        return xs[k]

    ttfts = [r["ttft"] for r in ok]
    lats = [r["latency"] for r in ok]
    tpots = [r["tpot"] for r in ok]
    toks = sum(r["ntok"] for r in ok)
    chunks = sum(r["nchunk"] for r in ok)
    return {
        "scenario": name,
        "prompt_chars": chars,
        "num_tasks": ntasks,
        "concurrency": conc,
        "max_tokens": max_tokens,
        "wall_s": round(wall, 2),
        "TTFT": round(statistics.mean(ttfts), 4),
        "TTFT_95": round(pct(ttfts, 0.95), 4),
        "LATENCY": round(statistics.mean(lats), 4),
        "LATENCY_95": round(pct(lats, 0.95), 4),
        "TPOT": round(statistics.mean(tpots), 5),
        "OUTPUT_TOKENS": toks,
        "OUTPUT_CHUNKS": chunks,
        "TOKENS_PER_CHUNK": round(toks / max(1, chunks), 3),
        "OUTPUT_TOK_PER_S": round(toks / wall, 2),
        "RPS": round(len(ok) / wall, 4),
        "FAILED_REQUESTS": len(fails),
        "errors": fails[:3],
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="arm label, e.g. dspark_off")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--only", default="", help="comma-separated scenario prefixes")
    args = ap.parse_args()

    picked = SCENARIOS
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        picked = [s for s in SCENARIOS if any(s[0].startswith(k) for k in keys)]

    out = {"tag": args.tag, "model": args.model, "scenarios": []}
    for name, chars, ntasks, conc, mt in picked:
        print(f"=== SERVING {args.tag} {name} ===", flush=True)
        r = await run_scenario(
            args.url, args.model, name, chars, ntasks, conc, mt, args.seed
        )
        print(json.dumps(r, indent=2), flush=True)
        out["scenarios"].append(r)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
    print(f"=== SERVING {args.tag} DONE ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
