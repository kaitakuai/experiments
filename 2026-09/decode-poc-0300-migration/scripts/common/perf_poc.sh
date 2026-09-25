#!/usr/bin/env bash
# Performance at the known profile, no sweeps: PoC nonces/s with all nonces submitted at once
# (engine-managed admission), then chat req/s at one concurrency. Env: OUT, KITDIR (/root/r-kit or /root/d-kit),
# TAG, NPOC (nonces, default 3000), WARM (warm-up nonces, default 0), CHAT_CONC, SERVED.
set -u
OUT=${OUT:?}; KITDIR=${KITDIR:-/root/r-kit}; TAG=${TAG:?}; NPOC=${NPOC:-3000}; WARM=${WARM:-0}; CHAT_CONC=${CHAT_CONC:?}
VENV=/opt/gv30; PY=$VENV/bin/python; LOG=$OUT/perf_${TAG}.log; say(){ echo "[$(date -u +%T)] $*" | tee -a "$LOG"; }
export KIT=$KITDIR VENV OUT ${SERVED:+SERVED_MODEL=$SERVED}
say "perf start tag=$TAG npoc=$NPOC warm=$WARM chat_conc=$CHAT_CONC"
POC=$(cd $KITDIR/scripts && env N=$NPOC WARM=$WARM $PY - <<'PY'
import os, rlib, json
warm = int(os.environ['WARM']); n = int(os.environ['N'])
if warm:
    rlib.generate(range(warm), bh=rlib.HASHES['h02'], batch_size=warm)
r, dt = rlib.generate(range(n), bh=rlib.HASHES['h01'], batch_size=n)
arts = r.get('artifacts') or []; full = sum(1 for a in arts if len(a.get('k_points_steps') or []) == 257)
nan = sum(int(a.get('n_nan_steps') or 0) for a in arts)
print(f"{n} nonces in {dt:.1f} s = {n/dt:.2f} nonce/s; complete {full}/{len(arts)}; nan steps {nan}")
json.dump({"tag": os.environ.get('TAG'), "nonces": n, "seconds": dt, "nonces_per_s": n/dt, "complete": full, "artifacts": len(arts), "nan_steps": nan}, open(os.path.join(rlib.OUT, f"perf_poc_{os.environ.get('TAG')}.json"), "w"))
PY
)
say "PoC: $POC"
rm -f $OUT/r_chat_sweep_chat_${TAG}*.json
(cd $KITDIR/scripts && env WAVES=3 IGNORE_EOS=1 timeout 2400 $PY -u r_chat_sweep.py chat_$TAG "$CHAT_CONC" > $OUT/chat_${TAG}.log 2>&1)
CH=$($PY -c "import json,glob;d=json.load(open(glob.glob('$OUT/r_chat_sweep_chat_${TAG}*.json')[0]));print([(p['concurrency'],round(p['req_s'],2),p.get('tpot_p50_ms')) for p in d['points']])" 2>&1 | tail -1)
say "chat (conc, req/s, tpot_p50_ms): $CH"
say "perf done"
