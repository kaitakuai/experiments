#!/usr/bin/env bash
# Steps 2 and 4 of stage_mm.sh (corpora + self-validation), for a re-run after the boot provenance was fixed.
set -u
OUT=${OUT:-/root/rout/mm}; TAG=${TAG:-mm030}; HASHES=${HASHES:-h01,h02,h03}
KIT=/root/r-kit; VENV=/opt/gv30; PY=$VENV/bin/python; LOG=$OUT/stage_${TAG}.log; say(){ echo "[$(date -u +%T)] $*" | tee -a "$LOG"; }
export KIT VENV OUT
say "--- 2. corpora (GEN_BATCH=250) re-run"; (cd $KIT/scripts && GEN_BATCH=250 $PY r_corpora.py honest $TAG "$HASHES" 2>&1 | tail -8 | tee -a "$LOG")
ls -la $OUT/corp_honest_h0*_gen.json 2>/dev/null | awk '{print "   ", $5, $9}' | tee -a "$LOG"
mkdir -p $OUT/vs_self; cp $OUT/current_boot.json $OUT/vs_self/
say "--- 4. validate this server's own corpora (same boot)"
(cd $KIT/scripts && OUT=$OUT/vs_self CORP_DIR=$OUT PART=250 RUN_LABEL=self $PY r2_validate.py "honest_h0*" $TAG 2>&1 | grep -vE "^\s*$" | tail -8 | tee -a "$LOG")
say "rest done"
