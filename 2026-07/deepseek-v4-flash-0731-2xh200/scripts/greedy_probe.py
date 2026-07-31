#!/usr/bin/env python3
"""Capture greedy completions for fixed prompts.

With temperature 0, speculative decoding must reproduce the non-speculative
output token for token. Running this in both arms and diffing is the
correctness oracle for the DSpark arm.
"""
import json, sys, urllib.request

URL, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PROMPTS = [
    "Explain in detail how a TCP three-way handshake works:\n",
    "List the first 20 prime numbers, comma separated:\n",
    "Write a short technical paragraph about GPU memory bandwidth:\n",
    "def quicksort(arr):\n",
    "The capital of France is",
]
out = []
for p in PROMPTS:
    body = json.dumps({"model": MODEL, "prompt": p, "max_tokens": 200,
                       "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request(URL + "/v1/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    out.append({"prompt": p,
                "text": d["choices"][0]["text"],
                "completion_tokens": d["usage"]["completion_tokens"],
                "finish_reason": d["choices"][0].get("finish_reason")})
    print(f"  {len(out)}/{len(PROMPTS)}  tokens={out[-1]['completion_tokens']} "
          f"finish={out[-1]['finish_reason']}", flush=True)
json.dump(out, open(OUT, "w"), indent=2)
print("GREEDY_PROBE_SAVED", OUT)
