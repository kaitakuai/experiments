import re, sys, os
R = "/app/packages/api/src/api/inference/vllm/runner.py"
TP  = os.environ.get("TP", "4")
SPEC = os.environ.get("SPEC", "")          # "" = выключено, иначе JSON
MNBT = os.environ.get("MNBT", "65536")
MML  = os.environ.get("MML", "262144")
GMU  = os.environ.get("GMU", "0.90")

src = open(R).read()
if not os.path.exists(R + ".hy3.bak"):
    open(R + ".hy3.bak", "w").write(src)
else:
    src = open(R + ".hy3.bak").read()      # всегда патчим от оригинала

forced = [
    ('--tensor-parallel-size', TP),
    ('--gpu-memory-utilization', GMU),
    ('--max-model-len', MML),
    ('--max-num-batched-tokens', MNBT),
    ('--kv-cache-dtype', 'fp8'),
    ('--logprobs-mode', 'processed_logprobs'),
    ('--worker-extension-cls', 'gonka_poc.worker.PoCWorkerExtension'),
    ('--tool-call-parser', 'hy_v3'),
    ('--reasoning-parser', 'hy_v3'),
]
if SPEC:
    forced.append(('--speculative-config', SPEC))

body = "\n".join(f"            ({f!r}, {v!r})," for f, v in forced)
new_list = "_dsv4_0731_forced = [\n" + body + "\n        ]"
src, n1 = re.subn(r"_dsv4_0731_forced = \[.*?\n        \]", new_list, src, flags=re.S)

new_flags = ("_dsv4_0731_flags = [\n"
             "            '--trust-remote-code',\n"
             "            '--enable-auto-tool-choice',\n"
             "            '--enable-expert-parallel',\n"
             "        ]")
src, n2 = re.subn(r"_dsv4_0731_flags = \[.*?\n        \]", new_flags, src, flags=re.S)

assert n1 == 1 and n2 == 1, (n1, n2)
open(R, "w").write(src)
print(f"patched: TP={TP} MML={MML} MNBT={MNBT} GMU={GMU} SPEC={SPEC or 'off'}")
