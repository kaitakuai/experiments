#!/usr/bin/env python3
"""Regenerate every table in this experiment's README from the committed artifacts.

Usage:  python3 scripts/summarize.py artifacts > artifacts/summary.json

L2 arithmetic is identical to the chain (vllm/poc/data.py): fp16 little-endian
decoded to fp32, float64 norm, strict '>' against the distance threshold.
"""
import base64
import glob
import json
import os
import re
import statistics
import sys

import numpy as np

DIST_THRESHOLD = 0.40


def vec(b64):
    return np.frombuffer(base64.b64decode(b64), dtype="<f2").astype(np.float32)


def load(path):
    """Same shape as l2_crossval.py: {nonce_index: base64 fp16 vector}."""
    d = json.load(open(path))
    return {a["nonce"]: a["vector_b64"] for a in d["artifacts"]}


def compare(a_path, b_path):
    a, b = load(a_path), load(b_path)
    common = sorted(set(a) & set(b))
    d = sorted(float(np.linalg.norm(vec(a[k]) - vec(b[k]))) for k in common)
    k = sum(1 for x in d if x > DIST_THRESHOLD)
    return {
        "n": len(common),
        "median": round(statistics.median(d), 6) if d else None,
        "p95": round(d[int(0.95 * (len(d) - 1))], 6) if d else None,
        "max": round(d[-1], 6) if d else None,
        "mismatches": k,
        "mismatch_pct": round(100.0 * k / len(d), 2) if d else None,
        "bit_identical": sum(1 for x in d if x == 0.0),
    }


def sweep(path):
    rows = {}
    for line in open(path, errors="ignore"):
        m = re.match(r"\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)", line)
        if m:
            rows[int(m.group(1))] = int(m.group(3))
    return rows


def main():
    art = sys.argv[1] if len(sys.argv) > 1 else "artifacts"
    j = lambda *p: os.path.join(art, *p)
    out = {"dist_threshold": DIST_THRESHOLD}

    out["poc_sweep_nonces_per_min"] = {
        "v1_off": sweep(j("logs", "sweep_v1_off.log")),
        "v2_off": sweep(j("logs", "sweep_v2_off.log")),
        "v2_dspark": sweep(j("logs", "sweep_v2_dspark.log")),
    }

    seeds = ["s1", "s2", "s3"]
    out["l2"] = {
        "runner_v1_vs_v2_no_spec": {
            s: compare(j(f"nonces_v1_off_{s}.json"), j(f"nonces_v2_off_{s}.json"))
            for s in seeds
        },
        "dspark_on_vs_off_same_runner": {
            s: compare(j(f"nonces_v2_off_{s}.json"), j(f"nonces_v2_dspark_{s}.json"))
            for s in seeds
        },
        "dspark_on_vs_v1_baseline": {
            s: compare(j(f"nonces_v1_off_{s}.json"), j(f"nonces_v2_dspark_{s}.json"))
            for s in seeds
        },
        "independent_seeds_control": {
            "s1_vs_s2": compare(j("nonces_v1_off_s1.json"), j("nonces_v1_off_s2.json"))
        },
    }

    old = sorted(glob.glob(os.path.join(art, "..", "..", "deepseek-v4-flash-2xh200",
                                        "artifacts", "nonces_eager.json")))
    if old:
        out["l2"]["checkpoint_0731_vs_old_flash"] = {
            "legacy_seed": compare(j("nonces_v1_off_legacyseed.json"), old[0]),
            "note": "old set collected on image k4; see README caveat",
        }

    serv = {}
    for tag in ("dspark_off", "dspark_on"):
        d = json.load(open(j(f"serving_{tag}.json")))
        serv[tag] = {
            s["scenario"]: {
                k: s[k]
                for k in ("TTFT", "LATENCY", "TPOT", "OUTPUT_TOK_PER_S",
                          "TOKENS_PER_CHUNK", "OUTPUT_TOKENS", "FAILED_REQUESTS")
                if k in s
            }
            for s in d["scenarios"]
        }
    out["serving"] = serv
    out["serving_speedup"] = {
        k: {
            "tok_per_s_x": round(serv["dspark_on"][k]["OUTPUT_TOK_PER_S"]
                                 / serv["dspark_off"][k]["OUTPUT_TOK_PER_S"], 3),
            "tpot_x": round(serv["dspark_off"][k]["TPOT"] / serv["dspark_on"][k]["TPOT"], 3),
        }
        for k in serv["dspark_off"]
    }

    g_on = json.load(open(j("greedy_dspark_on.json")))
    g_off = json.load(open(j("greedy_dspark_off.json")))
    g_off2 = json.load(open(j("greedy_dspark_off_run2.json")))
    out["greedy_equivalence"] = {
        "dspark_on_vs_off_identical": sum(
            a["text"] == b["text"] for a, b in zip(g_on, g_off)),
        "off_vs_off_identical_control": sum(
            a["text"] == b["text"] for a, b in zip(g_off, g_off2)),
        "n_prompts": len(g_on),
        "note": "control < n means the engine is nondeterministic run to run, "
                "so the text diff cannot decide output equivalence",
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
