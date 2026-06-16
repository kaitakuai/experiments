"""decode-PoC live test driver. Runs inside the container against localhost vLLM.

Usage: python3 driver.py <phase> [port]
  phases: golden | regression | decode
Artifacts persisted under /hf/poc-tests/.
"""
import json
import os
import sys
import time
import urllib.request

PORT = 8000
BASE = f"http://127.0.0.1:{PORT}/api/v1/pow"
OUT = "/hf/poc-tests"
os.makedirs(OUT, exist_ok=True)

BH = "deadbeef" * 8
PK = "cafebabe" * 8
NONCES = list(range(int(os.environ.get("N_NONCES", "8"))))
MODEL = "MiniMaxAI/MiniMax-M2.7"
SEQ_LEN = 256
MAX_TOKENS = int(os.environ.get("STEPS", "8"))


def post(path, payload, timeout=1800):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    out["_elapsed_s"] = round(time.time() - t0, 2)
    return out


def gen_payload(max_tokens=0, inf_map=None, validation=None):
    params = {"model": MODEL, "seq_len": SEQ_LEN, "k_dim": 12}
    if max_tokens:  # omit when 0: unpatched server's PoCParamsModel forbids extras
        params["max_tokens"] = max_tokens
    p = {
        "block_hash": BH, "block_height": 1, "public_key": PK,
        "node_id": 0, "node_count": 1, "nonces": NONCES,
        "wait": True,
        "batch_size": len(NONCES),
        "params": params,
    }
    if inf_map is not None:
        p["inference_k_points_steps"] = inf_map
    if validation is not None:
        p["validation"] = validation
    return p


def by_nonce(arts):
    return {a["nonce"]: a for a in arts}


def check(name, cond, detail=""):
    print(("OK   " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILURES.append(name)


FAILURES = []


def phase_v2(outfile):
    r1 = post("/generate", gen_payload(max_tokens=0))
    arts = r1.get("artifacts", [])
    check("v2: 8 artifacts", len(arts) == 8, f"{r1['_elapsed_s']}s")
    check("v2: shape (no decode keys)",
          all(set(a) == {"nonce", "vector_b64"} for a in arts))
    json.dump(arts, open(outfile, "w"))
    r2 = post("/generate", gen_payload(max_tokens=0))
    check("v2: repeat run byte-identical", by_nonce(arts) == by_nonce(r2.get("artifacts", [])))
    print(f"V2_DONE -> {outfile}", "FAIL" if FAILURES else "PASS")


def phase_regression(file_a, file_b):
    a = by_nonce(json.load(open(file_a)))
    b = by_nonce(json.load(open(file_b)))
    check("regression: same nonce set", set(a) == set(b))
    mism = [n for n in a if a[n].get("vector_b64") != b.get(n, {}).get("vector_b64")]
    check(f"regression: {file_a} == {file_b} byte-identical", not mism, f"diff={mism}")
    print("REGRESSION_DONE", "FAIL" if FAILURES else "PASS")


def phase_decode():
    # A: generation mode
    ra = post("/generate", gen_payload(max_tokens=MAX_TOKENS))
    aa = by_nonce(ra.get("artifacts", []))
    check("decode A: 8 artifacts", len(aa) == 8, f"{ra['_elapsed_s']}s")
    check("decode A: k_points_steps len == 1+max_tokens",
          all(len(a.get("k_points_steps", [])) == 1 + MAX_TOKENS for a in aa.values()))
    check("decode A: k ids in [0,16)",
          all(0 <= k < 16 for a in aa.values() for k in a.get("k_points_steps", [])))
    check("decode A: n_sphere_mismatches == -1 (generation)",
          all(a.get("n_sphere_mismatches") == -1 for a in aa.values()))
    json.dump(ra.get("artifacts", []), open(f"{OUT}/decode_a.json", "w"))

    # B: repeat -> identical trajectories
    rb = post("/generate", gen_payload(max_tokens=MAX_TOKENS))
    bb = by_nonce(rb.get("artifacts", []))
    check("decode B: repeat k-trajectories identical",
          all(aa[n]["k_points_steps"] == bb[n]["k_points_steps"] for n in aa))
    check("decode B: v2 vectors identical too",
          all(aa[n]["vector_b64"] == bb[n]["vector_b64"] for n in aa))

    # C: self-validation (teacher forcing on own trajectories) -> 0 mismatches
    inf_map = {str(n): aa[n]["k_points_steps"] for n in aa}
    rc = post("/generate", gen_payload(max_tokens=MAX_TOKENS, inf_map=inf_map))
    cc = by_nonce(rc.get("artifacts", []))
    check("decode C: self-validation 0 mismatches",
          all(a.get("n_sphere_mismatches") == 0 for a in cc.values()),
          str({n: a.get("n_sphere_mismatches") for n, a in cc.items()}))

    # D: validation mode (own v2 vectors as reference) -> verdict + sphere_mismatches
    try:
        pd = gen_payload(
            max_tokens=MAX_TOKENS, inf_map=inf_map,
            validation={"artifacts": [{"nonce": n, "vector_b64": aa[n]["vector_b64"]}
                                      for n in aa]})
        pd["stat_test"] = {"dist_threshold": 0.02, "p_mismatch": 0.001,
                           "fraud_threshold": 0.01}
        rd = post("/generate", pd)
        check("decode D: fraud_detected == False", rd.get("fraud_detected") is False,
              f"n_mismatch={rd.get('n_mismatch')}")
        sm = rd.get("sphere_mismatches")
        check("decode D: sphere_mismatches present and all 0",
              isinstance(sm, dict) and len(sm) == 8 and all(v == 0 for v in sm.values()),
              str(sm))
    except Exception as e:
        check("decode D: validation-mode response", False, f"exc={type(e).__name__}:{e}")

    # E: negative control — corrupt one reference k -> mismatches detected
    bad_map = {n: list(v) for n, v in inf_map.items()}
    first = sorted(bad_map)[0]
    bad_map[first][3] = (bad_map[first][3] + 1) % 16
    re_ = post("/generate", gen_payload(max_tokens=MAX_TOKENS, inf_map=bad_map))
    ee = by_nonce(re_.get("artifacts", []))
    bad_n = int(first)
    others_clean = all(a.get("n_sphere_mismatches") == 0
                       for n, a in ee.items() if n != bad_n)
    check("decode E: corrupted nonce has mismatches >= 1",
          ee[bad_n].get("n_sphere_mismatches", 0) >= 1,
          f"got={ee[bad_n].get('n_sphere_mismatches')}")
    check("decode E: other nonces stay 0", others_clean)

    json.dump({n: a.get("k_points_steps") for n, a in aa.items()},
              open(f"{OUT}/trajectories_b300.json", "w"), indent=1)
    print("DECODE_DONE", "FAIL" if FAILURES else "PASS")


def phase_crosshw(ref_file, label):
    """Validation against a reference trajectory from OTHER hardware.

    Teacher-forced: each step seeds the next with the reference k, so there is
    NO cascade — n_sphere_mismatches counts only steps where THIS hardware's
    quantized k diverges from the reference. Pure honest cross-hardware signal.
    """
    ref = json.load(open(ref_file))            # {nonce_str: [k0..kN]}
    inf_map = {str(int(n)): v for n, v in ref.items()}
    steps_per = len(next(iter(ref.values())))
    r = post("/generate", gen_payload(max_tokens=steps_per - 1, inf_map=inf_map))
    arts = by_nonce(r.get("artifacts", []))
    check(f"crosshw[{label}]: 8 artifacts", len(arts) == 8, f"{r['_elapsed_s']}s")
    # own trajectory this hardware computed (free-running, for the record)
    own = {str(n): a.get("k_points_steps") for n, a in arts.items()}
    json.dump(own, open(f"{OUT}/trajectories_{label}.json", "w"), indent=1)
    mm = {n: a.get("n_sphere_mismatches") for n, a in arts.items()}
    total_mm = sum(v for v in mm.values() if isinstance(v, int) and v >= 0)
    total_steps = len(arts) * steps_per
    print(f"CROSSHW[{label}] mismatches per nonce: {mm}")
    print(f"CROSSHW[{label}] TOTAL: {total_mm}/{total_steps} step-comparisons "
          f"diverge ({100.0*total_mm/total_steps:.1f}% honest cross-hw mismatch)")
    check(f"crosshw[{label}]: mismatch fraction < 40% (honest, not fraud)",
          total_mm / total_steps < 0.40, f"{total_mm}/{total_steps}")
    print(f"CROSSHW_DONE[{label}]", "FAIL" if FAILURES else "PASS")


def _validate_mm(inf_map):
    """Run teacher-forced validation, return list of n_sphere_mismatches."""
    r = post("/generate", gen_payload(max_tokens=MAX_TOKENS, inf_map=inf_map))
    arts = by_nonce(r.get("artifacts", []))
    return [arts[n].get("n_sphere_mismatches") for n in sorted(arts)], r["_elapsed_s"]


def _kl_bern(p_hi, p_lo):
    """KL(Bern(p_hi) || Bern(p_lo)) in bits — evidence per trial."""
    import math
    eps = 1e-9
    p_hi = min(max(p_hi, eps), 1 - eps)
    p_lo = min(max(p_lo, eps), 1 - eps)
    return (p_hi * math.log2(p_hi / p_lo)
            + (1 - p_hi) * math.log2((1 - p_hi) / (1 - p_lo)))


def phase_separation(crosshw_ref):
    positions = (1 + MAX_TOKENS)
    total_pos = len(NONCES) * positions
    print(f"SEPARATION: {len(NONCES)} nonces x {positions} positions "
          f"= {total_pos} comparisons/arm")

    # honest reference on THIS hardware (generation, free-running)
    rg = post("/generate", gen_payload(max_tokens=MAX_TOKENS))
    own = {str(a["nonce"]): a["k_points_steps"] for a in rg.get("artifacts", [])}
    json.dump(own, open(f"{OUT}/sep_ref_self.json", "w"))
    check("separation: reference generated", len(own) == len(NONCES), f"{rg['_elapsed_s']}s")

    arms = {}
    # FLOOR: self-validation (same model, same hw) -> expect 0
    self_mm, t = _validate_mm({n: own[n] for n in own})
    arms["floor_self"] = self_mm
    # CEILING: cross-nonce (validate nonce i vs neighbor i+1 reference) -> ~15/16
    keys = sorted(own, key=int)
    shifted = {keys[i]: own[keys[(i + 1) % len(keys)]] for i in range(len(keys))}
    ceil_mm, _ = _validate_mm(shifted)
    arms["ceiling_xnonce"] = ceil_mm
    # HONEST cross-hw: validate THIS hw vs OTHER hw reference (same nonces/steps)
    if crosshw_ref and os.path.exists(crosshw_ref):
        ref = json.load(open(crosshw_ref))
        ref = {str(int(k)): v for k, v in ref.items()}
        common = [n for n in own if n in ref and len(ref[n]) == positions]
        if common:
            cross_mm, _ = _validate_mm({n: ref[n] for n in common})
            arms["honest_crosshw"] = cross_mm

    print("\n=== SEPARATION RESULTS (per-step mismatch probability p) ===")
    stats = {}
    for name, mm in arms.items():
        mm = [m for m in mm if isinstance(m, int) and m >= 0]
        tot = sum(mm)
        p = tot / (len(mm) * positions) if mm else 0.0
        stats[name] = p
        print(f"  {name:16s}: p = {p:.4f}  ({tot}/{len(mm)*positions})  "
              f"per-nonce mean={tot/max(len(mm),1):.2f}/{positions}")
    json.dump({"stats": stats, "arms": arms, "positions": positions,
               "n_nonces": len(NONCES)}, open(f"{OUT}/separation.json", "w"), indent=1)

    # discrimination: bits/step for each fraud-like arm vs the honest floor
    base = stats.get("honest_crosshw", stats.get("floor_self", 0.0))
    print(f"\n=== DISCRIMINATION (KL bits/step vs honest baseline p={base:.4f}) ===")
    for name in ("ceiling_xnonce",):
        if name in stats:
            kl = _kl_bern(stats[name], base)
            steps_for_1e6 = (13.8 / kl) if kl > 0 else float("inf")  # ln(1e6)≈13.8
            print(f"  {name} vs honest: {kl:.3f} bits/step  "
                  f"-> ~{steps_for_1e6:.1f} steps for 1e-6 false-accept")
    print("SEPARATION_DONE", "FAIL" if FAILURES else "PASS")


def phase_genref(outfile):
    r = post("/generate", gen_payload(max_tokens=MAX_TOKENS))
    own = {str(a["nonce"]): a["k_points_steps"] for a in r.get("artifacts", [])}
    json.dump(own, open(outfile, "w"))
    check(f"genref: {len(own)} nonces x {1+MAX_TOKENS} steps -> {outfile}",
          len(own) == len(NONCES), f"{r['_elapsed_s']}s")
    print("GENREF_DONE", "FAIL" if FAILURES else "PASS")


def phase_profile(ref_file):
    """Per-step honest mismatch profile (teacher-forced vs OTHER-hw reference).

    Returns H100's teacher-forced k per step in the artifact; compare
    element-wise to the reference to see if mismatch GROWS with step
    (KV-history accumulation) or is FLAT (per-step projection noise).
    """
    ref = json.load(open(ref_file))
    ref = {str(int(k)): v for k, v in ref.items()}
    inf_map = {n: ref[n] for n in ref if len(ref[n]) == (1 + MAX_TOKENS)}
    r = post("/generate", gen_payload(max_tokens=MAX_TOKENS, inf_map=inf_map))
    arts = by_nonce(r.get("artifacts", []))
    pos = 1 + MAX_TOKENS
    # per-step mismatch fraction across nonces
    prof = []
    for s in range(pos):
        diff = 0
        tot = 0
        for n in arts:
            own = arts[n].get("k_points_steps")
            rk = ref[str(n)]
            if own and len(own) > s and len(rk) > s:
                tot += 1
                if own[s] != rk[s]:
                    diff += 1
        prof.append(diff / tot if tot else 0.0)
    json.dump(prof, open(f"{OUT}/profile.json", "w"))
    print(f"PROFILE ({len(arts)} nonces, {pos} positions):")
    # print step buckets
    buckets = [(0, 1), (1, 9), (9, 17), (17, 33), (33, 49), (49, 65)]
    for a, b in buckets:
        seg = prof[a:b]
        if seg:
            print(f"  steps {a:2d}-{b-1:2d}: mean mismatch = {sum(seg)/len(seg):.3f}")
    print(f"  step 0 (prefill): {prof[0]:.3f}")
    print(f"  step 1          : {prof[1]:.3f}" if pos > 1 else "")
    print(f"  step {pos-1:2d} (last)  : {prof[-1]:.3f}")
    # linear slope (growth signature)
    import statistics
    xs = list(range(pos))
    mx = sum(xs) / pos
    my = sum(prof) / pos
    slope = sum((xs[i]-mx)*(prof[i]-my) for i in range(pos)) / sum((x-mx)**2 for x in xs)
    print(f"  slope = {slope:.5f} mismatch/step  ({'GROWS->KV accumulation' if slope > 0.001 else 'FLAT->per-step noise'})")
    print("PROFILE_DONE")


def phase_spherevecs(outfile, ref_file=None):
    """Generate (or teacher-force) and save raw sphere vectors per step.

    Server must run with GONKA_POC_RETURN_SPHERE set. With ref_file the run is
    teacher-forced (validator processes the SAME inputs as the reference) so the
    sphere vectors are directly comparable for sign-sketch / angle analysis.
    """
    inf_map = None
    if ref_file:
        ref = json.load(open(ref_file))
        inf_map = {str(int(k)): v["k"] if isinstance(v, dict) else v
                   for k, v in ref.items()}
    r = post("/generate", gen_payload(max_tokens=MAX_TOKENS, inf_map=inf_map))
    arts = by_nonce(r.get("artifacts", []))
    out = {str(n): {"k": arts[n].get("k_points_steps"),
                    "sphere": arts[n].get("sphere_vecs")} for n in arts}
    json.dump(out, open(outfile, "w"))
    have = sum(1 for v in out.values() if v["sphere"])
    check(f"spherevecs: {len(arts)} nonces, {have} with sphere -> {outfile}",
          have == len(arts), f"{r['_elapsed_s']}s")
    print("SPHEREVECS_DONE", "FAIL" if FAILURES else "PASS")


def phase_pechat(outfile):
    """Prefill-only run; collect full-volume sign-sketches (per nonce, per layer).

    Server must run with GONKA_POC_FULLVOL=1. Inputs are seed-deterministic, so
    no teacher-forcing needed — honest = same model diff hw, fraud = diff model.
    """
    r = post("/generate", gen_payload(max_tokens=0))
    arts = by_nonce(r.get("artifacts", []))
    out = {str(n): arts[n].get("fullvol_sketch") for n in arts}
    json.dump(out, open(outfile, "w"))
    have = sum(1 for v in out.values() if v)
    check(f"pechat: {len(arts)} nonces, {have} with sketch -> {outfile}",
          have == len(arts), f"{r['_elapsed_s']}s")
    print("PECHAT_DONE", "FAIL" if FAILURES else "PASS")


cmd = sys.argv[1]
if cmd == "pechat":
    phase_pechat(sys.argv[2])
elif cmd == "spherevecs":
    phase_spherevecs(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
elif cmd == "profile":
    phase_profile(sys.argv[2])
elif cmd == "genref":
    phase_genref(sys.argv[2])
elif cmd == "separation":
    phase_separation(sys.argv[2] if len(sys.argv) > 2 else None)
elif cmd == "v2":
    phase_v2(sys.argv[2])
elif cmd == "regression":
    phase_regression(sys.argv[2], sys.argv[3])
elif cmd == "decode":
    phase_decode()
elif cmd == "crosshw":
    phase_crosshw(sys.argv[2], sys.argv[3])
else:
    raise SystemExit(f"unknown phase {cmd}")
sys.exit(1 if FAILURES else 0)
