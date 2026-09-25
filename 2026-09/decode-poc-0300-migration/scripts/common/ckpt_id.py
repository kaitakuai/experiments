import glob, hashlib, json, os, re, sys
d = sys.argv[1]
rev = None
m = re.search(r"snapshots/([0-9a-f]{40})", os.path.realpath(d) + "/")
if m: rev = m.group(1)
h = hashlib.sha256()
files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
for f in files:
    h.update(os.path.basename(f).encode()); h.update(str(os.path.getsize(f)).encode())
    with open(f, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        if 0 < n <= 100 << 20: h.update(fh.read(n))
cfgp = os.path.join(d, "config.json"); ch = ""
if os.path.exists(cfgp):
    b = open(cfgp, "rb").read(); h.update(b); ch = hashlib.sha256(b).hexdigest()[:16]
print("dir", d, "realpath", os.path.realpath(d))
print("files", len(files), "total_bytes", sum(os.path.getsize(f) for f in files))
print("rev", rev, "ckpt_sha", h.hexdigest()[:16], "cfg_sha", ch)
# hf download metadata (commit hash) if present
for meta in glob.glob(os.path.join(d, ".cache/huggingface/download/*.metadata"))[:3]:
    print("meta", os.path.basename(meta), open(meta).read().replace("\n"," | ")[:200])
for extra in ("generation_config.json","tokenizer_config.json","model.safetensors.index.json"):
    p=os.path.join(d,extra)
    if os.path.exists(p): print(extra, hashlib.sha256(open(p,'rb').read()).hexdigest()[:16], os.path.getsize(p))
