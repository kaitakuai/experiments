#!/usr/bin/env python3
"""Is DSpark's output what the target model would have produced on its own?

Comparing generated *texts* between the two arms does not work: the engine is
nondeterministic run to run, so texts diverge even with speculation off.

This does teacher forcing instead. Phase 'gen' records greedy completions.
Phase 'check' replays prompt+completion through an engine WITHOUT speculation,
asking for per-token logprobs with echo, and counts how often the recorded
token was the target model's argmax at that position.

  arm under test : completions recorded with DSpark on
  control        : completions recorded with DSpark off

Both are replayed on the same non-speculative engine, so whatever numerical
noise teacher forcing itself has, it applies equally. A DSpark deficit above
the control's is evidence that acceptance is lossy.
"""
import argparse
import json
import urllib.request

PROMPTS = [
    "Explain how a TCP three-way handshake works, step by step:\n",
    "List the first 30 prime numbers, comma separated:\n",
    "Write a technical paragraph about GPU memory bandwidth:\n",
    "def quicksort(arr):\n",
    "The capital of France is",
    "Summarise the causes of the 1929 financial crash:\n",
    "Translate to German: 'The server is currently unavailable.'\n",
    "Write a bash one-liner that finds the ten largest files under /var:\n",
]


def post(url, payload, timeout=900):
    req = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def gen(url, model, out, max_tokens):
    recs = []
    for p in PROMPTS:
        d = post(url + "/v1/completions",
                 {"model": model, "prompt": p, "max_tokens": max_tokens,
                  "temperature": 0, "stream": False})
        recs.append({"prompt": p, "text": d["choices"][0]["text"],
                     "completion_tokens": d["usage"]["completion_tokens"]})
        print(f"  gen {len(recs)}/{len(PROMPTS)} tokens={recs[-1]['completion_tokens']}",
              flush=True)
    json.dump(recs, open(out, "w"), indent=2)
    print("GEN_SAVED", out)


def check(url, model, rec_path, out):
    recs = json.load(open(rec_path))
    total = argmax_hits = 0
    per = []
    for r in recs:
        full = r["prompt"] + r["text"]
        d = post(url + "/v1/completions",
                 {"model": model, "prompt": full, "max_tokens": 0, "echo": True,
                  "logprobs": 1, "temperature": 0, "stream": False})
        lp = d["choices"][0]["logprobs"]
        toks = lp["tokens"]
        tops = lp["top_logprobs"]
        # locate where the completion starts: replay the prompt's token count by
        # re-echoing the prompt alone
        dp = post(url + "/v1/completions",
                  {"model": model, "prompt": r["prompt"], "max_tokens": 0,
                   "echo": True, "logprobs": 1, "temperature": 0, "stream": False})
        n_prompt = len(dp["choices"][0]["logprobs"]["tokens"])
        hits = n = 0
        for i in range(n_prompt, len(toks)):
            top = tops[i]
            if not top:
                continue
            best = max(top, key=top.get)
            n += 1
            hits += int(best == toks[i])
        per.append({"prompt": r["prompt"][:40], "positions": n, "argmax_hits": hits,
                    "pct": round(100.0 * hits / n, 3) if n else None})
        total += n
        argmax_hits += hits
        print(f"  check {len(per)}/{len(recs)}  {hits}/{n} argmax", flush=True)
    res = {"source": rec_path, "positions": total, "argmax_hits": argmax_hits,
           "pct": round(100.0 * argmax_hits / total, 4) if total else None,
           "per_prompt": per}
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("positions", "argmax_hits", "pct")}))
    print("CHECK_SAVED", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["gen", "check"])
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--records", help="for check: the gen output to replay")
    ap.add_argument("--max-tokens", type=int, default=200)
    a = ap.parse_args()
    if a.phase == "gen":
        gen(a.url, a.model, a.out, a.max_tokens)
    else:
        check(a.url, a.model, a.records, a.out)
