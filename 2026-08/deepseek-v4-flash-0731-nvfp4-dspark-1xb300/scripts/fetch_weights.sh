#!/bin/bash
# Fetch the NVFP4 checkpoint into a plain directory with curl.
#
# Two reasons not to use the hf CLI here: the host outside the container has no
# Python packages, and the published shard sizes are the only trustworthy
# integrity reference — `conversion-receipt.json` in that repo records byte
# counts and sha256 from *before* upload and does not match the published files
# (differences of 128 B to 337 KB per shard). We verify against the HF API.
set -u
REPO=${REPO:-MJPansa/DeepSeek-V4-Flash-0731-NVFP4}
REV=${REV:-64d64cd89bc63a66aa46506da89d7821f7491c62}
DEST=${DEST:-/root/nvfp4}
mkdir -p "$DEST"

curl -s "https://huggingface.co/api/models/$REPO?blobs=true&revision=$REV" -o /tmp/repo.json
python3 - <<'PY' > /tmp/files.txt
import json
KEEP = {"config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "model.safetensors.index.json",
        "hf_quant_config.json"}
d = json.load(open("/tmp/repo.json"))
for s in d["siblings"]:
    f = s["rfilename"]
    if f.endswith(".safetensors") or f in KEEP:
        print(f)
PY
echo "files: $(wc -l < /tmp/files.txt)"

fetch() {
  local f="$1" out="$DEST/$1"
  [ -s "$out" ] && return 0
  curl -sfL --retry 5 --retry-delay 5 \
    "https://huggingface.co/$REPO/resolve/$REV/$f" -o "$out.part" \
    && mv "$out.part" "$out" || echo "FAIL $f"
}
export -f fetch; export DEST REPO REV
xargs -a /tmp/files.txt -P 8 -I{} bash -c 'fetch "$@"' _ {}

echo "=== verify shard sizes against the HF API ==="
python3 - <<'PY' > /tmp/sizes.txt
import json
d = json.load(open("/tmp/repo.json"))
for s in d["siblings"]:
    if s["rfilename"].endswith(".safetensors"):
        print(s["rfilename"], s.get("size") or s["lfs"]["size"])
PY
BAD=0
while read -r f b; do
  a=$(stat -c %s "$DEST/$f" 2>/dev/null || echo 0)
  [ "$a" = "$b" ] || { echo "MISMATCH $f got=$a want=$b"; BAD=1; }
done < /tmp/sizes.txt
[ $BAD -eq 0 ] && echo "ALL_SHARDS_MATCH_HF ($(du -sh "$DEST" | cut -f1))"
