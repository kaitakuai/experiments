#!/usr/bin/env python3
"""
DeepSeek-V4-Flash inference validation, gonka methodology.

  generate : executor side. Normal chat completion, stores logprobs + enforced_tokens.
  replay   : validator side. Re-runs the SAME token sequence via enforced_tokens,
             stores validator logprobs and the gonka distance2.

Both sides must use identical sampling params; only --logprobs-mode differs per pass.
"""

import argparse
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

SYSTEM_PROMPT = "You are a helpful assistant. Response clear, correct and complete."


def build_payload(model, prompt, args, enforced_tokens=None):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "stream": False,
        "n": 1,
        "logprobs": True,
        "top_logprobs": args.top_logprobs,
        "skip_special_tokens": False,
        "repetition_penalty": args.repetition_penalty,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "logprobs_mode": args.logprobs_mode,
    }
    if enforced_tokens is not None:
        payload["enforced_tokens"] = enforced_tokens
    return payload


def post(url, payload, timeout, retries=3):
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(url.rstrip("/") + "/chat/completions",
                              json=payload, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last = RuntimeError("HTTP %d: %s" % (r.status_code, r.text[:400]))
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(1.5 * (attempt + 1))
    raise last


def extract(resp):
    """-> {text, results:[{token, logprobs:{tok: lp}}]}"""
    choice = resp["choices"][0]
    content = choice["logprobs"]["content"]
    results = []
    for pos in content:
        lps = {}
        for lp in pos["top_logprobs"]:
            lps[str(lp["token"])] = lp["logprob"]
        results.append({"token": str(pos["token"]), "logprobs": lps})
    return {"text": choice["message"]["content"], "results": results}


def enforced_from(resp):
    """-> {tokens:[{token, top_tokens:[...]}]}  (gonka enforced_tokens format)"""
    content = resp["choices"][0]["logprobs"]["content"]
    toks = []
    for pos in content:
        toks.append({
            "token": pos["token"],
            "top_tokens": [lp["token"] for lp in pos["top_logprobs"]],
        })
    return {"tokens": toks}


def token_distance2_pos(inf_lp, val_lp):
    if not val_lp:
        return float(len(inf_lp)), 0
    sorted_vals = sorted(val_lp.values())
    if len(sorted_vals) >= 2:
        min1, min2 = sorted_vals[0], sorted_vals[1]
    else:
        min1, min2 = sorted_vals[0], sorted_vals[0] - 1.0
    dist = 0.0
    n_match = 0
    for token, inf_logprob in inf_lp.items():
        if token in val_lp:
            val_logprob = val_lp[token]
            n_match += 1
        else:
            val_logprob = min1 - (min2 - min1)
        denom = 1e-10 + abs(inf_logprob) + abs(val_logprob)
        dist += abs(inf_logprob - val_logprob) / denom / 2.0
    return dist, n_match


def distance2(inf_results, val_results):
    n = min(len(inf_results), len(val_results))
    if n == 0:
        return None, 0, 0
    top_k = len(inf_results[0]["logprobs"]) or 1
    total = 0.0
    matches = 0
    for i in range(n):
        d, m = token_distance2_pos(inf_results[i]["logprobs"],
                                   val_results[i]["logprobs"])
        total += d
        matches += m
    return (total + 1.0) / (max(100, n) * top_k + 1.0), n, matches


def sentinel_stats(results):
    """Fraction of positions carrying a -9999 sentinel."""
    if not results:
        return 0.0
    hits = 0
    for pos in results:
        if any(v <= -9000 for v in pos["logprobs"].values()):
            hits += 1
    return hits / len(results)


def load_prompts(path, n):
    items = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items.append({"index": i,
                          "prompt": d.get("text") or d.get("prompt"),
                          "language": d.get("language", "en")})
    return items[:n] if n else items


def done_indices(path):
    seen = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["index"])
                except Exception:  # noqa: BLE001
                    pass
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["generate", "replay"])
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--prompts", help="prompts jsonl (generate mode)")
    ap.add_argument("--input", help="executor jsonl (replay mode)")
    ap.add_argument("--n-prompts", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--logprobs-mode", default="processed_logprobs",
                    choices=["raw_logprobs", "processed_logprobs"])
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=1000)
    ap.add_argument("--top-logprobs", type=int, default=4)
    ap.add_argument("--repetition-penalty", type=float, default=1.2)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--tag", default="", help="free-form label stored in each row")
    args = ap.parse_args()

    if args.mode == "generate":
        items = load_prompts(args.prompts, args.n_prompts)
    else:
        items = []
        with open(args.input, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        if args.n_prompts:
            items = items[:args.n_prompts]

    skip = done_indices(args.output)
    todo = [x for x in items if x["index"] not in skip]
    print("total=%d already_done=%d todo=%d mode=%s logprobs_mode=%s"
          % (len(items), len(skip), len(todo), args.mode, args.logprobs_mode),
          flush=True)
    if not todo:
        return

    lock = threading.Lock()
    out = open(args.output, "a", encoding="utf-8")
    state = {"ok": 0, "err": 0, "t0": time.time(),
             "dsum": 0.0, "dn": 0, "mismatch": 0, "sent": 0.0}

    def work(item):
        if args.mode == "generate":
            resp = post(args.url, build_payload(args.model, item["prompt"], args),
                        args.timeout)
            inf = extract(resp)
            row = {
                "index": item["index"],
                "language": item["language"],
                "prompt": item["prompt"],
                "logprobs_mode": args.logprobs_mode,
                "tag": args.tag,
                "model": args.model,
                "inference_result": inf,
                "enforced_tokens": enforced_from(resp),
                "n_tokens": len(inf["results"]),
                "sentinel_frac": sentinel_stats(inf["results"]),
            }
            return row

        # replay
        resp = post(args.url,
                    build_payload(args.model, item["prompt"], args,
                                  enforced_tokens=item["enforced_tokens"]),
                    args.timeout)
        val = extract(resp)
        inf = item["inference_result"]
        d, n, matches = distance2(inf["results"], val["results"])
        row = dict(item)
        row.pop("enforced_tokens", None)
        row["validation_result"] = val
        row["validator_model"] = args.model
        row["validator_tag"] = args.tag
        row["distance"] = d
        row["n_compared"] = n
        row["len_mismatch"] = len(inf["results"]) - len(val["results"])
        row["val_sentinel_frac"] = sentinel_stats(val["results"])
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, it): it for it in todo}
        for fut in as_completed(futs):
            it = futs[fut]
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001
                with lock:
                    state["err"] += 1
                print("  [%d] %s ERROR %s" % (it["index"], it.get("language"), e),
                      flush=True)
                continue
            with lock:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                state["ok"] += 1
                if args.mode == "replay":
                    if row["distance"] is not None:
                        state["dsum"] += row["distance"]
                        state["dn"] += 1
                    if row["len_mismatch"] != 0:
                        state["mismatch"] += 1
                    state["sent"] += row["val_sentinel_frac"]
                else:
                    state["sent"] += row["sentinel_frac"]
                done = state["ok"]
                if done % 25 == 0 or done == len(todo):
                    el = time.time() - state["t0"]
                    rate = done / el * 60 if el > 0 else 0
                    msg = ("  %d/%d ok err=%d %.1f/min sentinel=%.1f%%"
                           % (done, len(todo), state["err"], rate,
                              100.0 * state["sent"] / max(1, done)))
                    if args.mode == "replay" and state["dn"]:
                        msg += (" mean_d=%.4f mismatch=%d"
                                % (state["dsum"] / state["dn"], state["mismatch"]))
                    print(msg, flush=True)

    out.close()
    el = time.time() - state["t0"]
    print("DONE ok=%d err=%d in %.1fs (%.1f/min)"
          % (state["ok"], state["err"], el, state["ok"] / el * 60 if el else 0),
          flush=True)
    if args.mode == "replay" and state["dn"]:
        print("mean_distance2=%.6f over %d items, len_mismatches=%d"
              % (state["dsum"] / state["dn"], state["dn"], state["mismatch"]),
              flush=True)


if __name__ == "__main__":
    main()
