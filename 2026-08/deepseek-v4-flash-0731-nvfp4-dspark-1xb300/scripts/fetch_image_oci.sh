#!/bin/bash
# Pull an image from ghcr when `docker pull` will not.
#
# ghcr throttles layer downloads per IP. Past a few tens of GB, docker sits on
# "Pulling fs layer" with no "Downloading" lines and no disk growth, while the
# same host still gets 27-35 MB/s from HuggingFace and ghcr manifests answer in
# 0.2 s. The daemon is not hung; it is waiting for bytes nobody is sending.
#
# This builds an OCI layout by hand and hands it to `docker load`. Three details
# are load-bearing:
#   * a fresh token per blob - ghcr pull tokens expire in ~5 min and a 4 GB
#     layer outlives them
#   * explicit `Range` offsets, appending to .part - `curl -C -` does NOT work,
#     because the blob URL 302s to the CDN and the Range header is lost on the
#     redirect, so every retry silently restarts from zero
#   * one stream at a time - parallel fetches share the same throttled budget
#     and measure slower (5 MB/s at 4 streams vs 19 MB/s at one)
set -u
IMG=${IMG:-kaitakuai/mlnode-b300-deepseek-v4-flash-0731}
TAG=${TAG:-3.0.14-post2-vllm0.25.1-rc3-overlay-k4}
OUT=${OUT:-/root/oci}
STALL_MB=${STALL_MB:-32}      # a round gaining less than this means throttled
BACKOFF=${BACKOFF:-600}
REF="ghcr.io/$IMG:$TAG"

tok() { curl -s "https://ghcr.io/token?scope=repository:$IMG:pull&service=ghcr.io" \
        | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])' 2>/dev/null; }

mkdir -p "$OUT/blobs/sha256"
echo '{"imageLayoutVersion": "1.0.0"}' > "$OUT/oci-layout"

T=$(tok)
curl -s -H "Authorization: Bearer $T" \
  -H "Accept: application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json" \
  "https://ghcr.io/v2/$IMG/manifests/$TAG" -o "$OUT/index_raw.json"
MD=$(python3 -c "
import json;m=json.load(open('$OUT/index_raw.json'))
print([x['digest'] for x in m['manifests']
       if x.get('platform',{}).get('architecture')=='amd64'][0])")
curl -s -H "Authorization: Bearer $T" -H "Accept: application/vnd.oci.image.manifest.v1+json" \
  "https://ghcr.io/v2/$IMG/manifests/$MD" -o "$OUT/blobs/sha256/${MD#sha256:}"

python3 - "$MD" "$REF" "$OUT" <<'PY'
import json, os, sys
md, ref, out = sys.argv[1], sys.argv[2], sys.argv[3]
mpath = os.path.join(out, "blobs", "sha256", md.split(":", 1)[1])
json.dump({"schemaVersion": 2,
           "mediaType": "application/vnd.oci.image.index.v1+json",
           "manifests": [{"mediaType": "application/vnd.oci.image.manifest.v1+json",
                          "digest": md, "size": os.path.getsize(mpath),
                          "annotations": {"org.opencontainers.image.ref.name": ref}}]},
          open(os.path.join(out, "index.json"), "w"))
m = json.load(open(mpath))
with open(os.path.join(out, "blobs.txt"), "w") as fh:
    fh.write("%s %d\n" % (m["config"]["digest"], m["config"]["size"]))
    for layer in m["layers"]:
        fh.write("%s %d\n" % (layer["digest"], layer["size"]))
print("blobs: %d, compressed %.2f GB"
      % (1 + len(m["layers"]),
         (m["config"]["size"] + sum(l["size"] for l in m["layers"])) / 1e9))
PY

while read -r D WANT; do
  F="$OUT/blobs/sha256/${D#sha256:}"
  [ -s "$F" ] && [ "$(stat -c %s "$F")" = "$WANT" ] && { echo "have ${D:7:12}"; continue; }
  P="$F.part"; [ -f "$P" ] || : > "$P"
  for _ in $(seq 1 200); do
    HAVE=$(stat -c %s "$P" 2>/dev/null || echo 0)
    [ "$HAVE" -ge "$WANT" ] && break
    T=$(tok); [ -n "$T" ] || { sleep 20; continue; }
    curl -sfL -H "Authorization: Bearer $T" -r "${HAVE}-" \
         --speed-time 45 --speed-limit 131072 \
         "https://ghcr.io/v2/$IMG/blobs/$D" >> "$P" 2>/dev/null
    NOW=$(stat -c %s "$P" 2>/dev/null || echo 0)
    GAIN=$(( (NOW - HAVE) / 1048576 ))
    echo "$(date -u +%H:%M:%S) ${D:7:12} $((NOW/1048576))/$((WANT/1048576)) MB (+${GAIN} MB)"
    [ "$GAIN" -lt "$STALL_MB" ] && { echo "  throttled, sleeping ${BACKOFF}s"; sleep "$BACKOFF"; }
  done
  [ "$(stat -c %s "$P" 2>/dev/null || echo 0)" = "$WANT" ] \
    && { mv "$P" "$F"; echo "DONE ${D:7:12}"; } || echo "SHORT ${D:7:12}"
done < "$OUT/blobs.txt"

echo "=== verify ==="
BAD=0
while read -r D _; do
  F="$OUT/blobs/sha256/${D#sha256:}"
  [ -s "$F" ] || { echo "MISSING $D"; BAD=1; continue; }
  A=$(sha256sum "$F" | cut -d' ' -f1)
  [ "sha256:$A" = "$D" ] || { echo "DIGEST_MISMATCH $D"; BAD=1; }
done < "$OUT/blobs.txt"
[ $BAD -eq 0 ] || { echo "VERIFY_FAILED"; exit 1; }
echo "ALL_BLOBS_VERIFIED"

rm -f "$OUT/index_raw.json" "$OUT/blobs.txt"
tar cf - -C "$OUT" . | docker load
echo "FETCH_DONE"
