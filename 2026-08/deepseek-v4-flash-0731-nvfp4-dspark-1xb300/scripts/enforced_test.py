#!/usr/bin/env python3
"""Exercise the Gonka validation replay path (enforced_tokens).

gen   : greedy completion with top_logprobs -> build the enforced payload
replay: send it back and check the engine reproduces exactly those tokens
"""
import json, sys, urllib.request

URL = "http://127.0.0.1:8081/v1/chat/completions"
MODEL = "/models/nvfp4"
PROMPT = "List three prime numbers and explain briefly why each is prime."


def post(payload):
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def gen():
    d = post({"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
              "max_tokens": 64, "temperature": 0.0, "logprobs": True,
              "top_logprobs": 4, "seed": 1})
    content = d["choices"][0]["logprobs"]["content"]
    json.dump(content, open("/tmp/enforced_payload.json", "w"))
    print("GEN_TOKENS", len(content))
    print("GEN_TEXT", repr(d["choices"][0]["message"]["content"][:90]))
    return content


def replay(content):
    tokens = [{"token": p["token"], "top_tokens": [x["token"] for x in p["top_logprobs"]]}
              for p in content]
    d = post({"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
              "max_tokens": len(tokens), "temperature": 0.0,
              "logprobs": True, "top_logprobs": 4, "seed": 1,
              "enforced_tokens": {"tokens": tokens}})
    out = d["choices"][0]["logprobs"]["content"]
    got = [p["token"] for p in out]
    want = [p["token"] for p in content]
    n = min(len(got), len(want))
    match = sum(1 for i in range(n) if got[i] == want[i])
    print("REPLAY_TOKENS", len(got), "of", len(want))
    print("REPLAY_MATCH %d/%d" % (match, n))
    print("REPLAY_LOGPROBS_PRESENT", bool(out and out[0].get("logprob") is not None))
    print("REPLAY_OK" if (len(got) == len(want) and match == n) else "REPLAY_MISMATCH")


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen()
    else:
        replay(json.load(open("/tmp/enforced_payload.json")))
