# GLM-5.3-Flash — 2×B300 — NVFP4 fraud, re-analysed: the 0.711 figure does not transfer

**Data collected:** 2026-08-26 — **nothing was re-run for this folder.** The nonce sets are
byte-identical copies of
[`../../2026-08/glm53-flash-nvfp4-libertai-2xb300/`](../../2026-08/glm53-flash-nvfp4-libertai-2xb300/).
What is new is the comparison against the September 4×B200 run of the **same checkpoint**.

**Model (fraud):** [`LibertAIDAI/GLM-5.3-Flash-NVFP4`](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)
— NVFP4 via ModelOpt, snapshot `11d73216cd636238e82e1d77fe1042ffab36e7fa`
**Model (reference):** `zai-org/GLM-5.3-Flash`, native FP8, same box
**Hardware:** 2× NVIDIA B300 SXM6 AC, TP=2, driver 610.57.04
**Image:** `ghcr.io/kaitakuai/vllm-poc:glm53-poc-v4-ed8873884` (old profile, FlashInfer 0.6.17)

## The correction this folder exists to record

The August report called this **"the loudest fraud we have measured"** — median L2 **0.711**,
97.3 % of nonces past the 0.40 gate. That number is reproduced here from the committed sets:

| seed | median L2 vs honest | past 0.40 |
|---|---:|---:|
| s1 | 0.7138 | 96.7 % |
| s2 | 0.7132 | 97.3 % |
| s3 | 0.7050 | 97.8 % |

**It is not a property of the attack.** The identical checkpoint on 4×B200 with the `k3` image
measures **0.379 / 43 %** — half as loud
([`../glm53-flash-nvfp4-libertai-4xb200/`](../glm53-flash-nvfp4-libertai-4xb200/)). And the two
fraud runs disagree with *each other* more than either disagrees with its honest baseline:

| pair | median L2 | past 0.40 | what differs |
|---|---:|---:|---|
| this NVFP4 ↔ its own honest arm | 0.7132 | 97.3 % | checkpoint |
| **this NVFP4 ↔ September NVFP4 on B200** | **0.7368** | **98.0 %** | hardware, image, TP |
| September NVFP4 ↔ its own honest arm | 0.3791 | 43.2 % | checkpoint |

Two runs of one quantised checkpoint are further apart (0.737) than fraud is from honest on
either box. Quantised numerics depend on which kernels execute them far more than FP8 numerics
do — for contrast, the honest arms across the same platform move only 0.27.

**Consequence for validation:** a published NVFP4 signature is a property of the platform it
was measured on. Detection has to be framed as *far from honest*, never as *close to a known
fraud fingerprint*. The margin that actually matters is against honest cross-generation noise
(16–17 %), and against that this fraud sits at 43 % on B200 and 97 % on B300 — a 2.5× to 6×
margin depending on where the prover runs.

*Caveat:* architecture, image and TP all differ between the two NVFP4 runs, so the split
between them is unknown. A 2×B300 run on `k3` would decompose it; that was planned and
dropped, because the actionable conclusion — the quantity is not stable — does not depend on
the decomposition.

## Files

| path | what |
|---|---|
| `artifacts/nonces_nvfp4_{s1,s2,s3}.json` | the August fraud sets, copied verbatim |
| `artifacts/ref_nonces_honest_{s1,s2,s3}.json` | the August honest sets, same box |
| [`artifacts/summary.json`](artifacts/summary.json) | L2 tables and the batch split, regenerated |
| [`artifacts/matrix.json`](artifacts/matrix.json) | full pairwise matrix across all six arms |
| [`scripts/summarize.py`](scripts/summarize.py), [`scripts/matrix.py`](scripts/matrix.py) | regenerate every table |
| [`scripts/poc_seeds.json`](scripts/poc_seeds.json) | the fixed seed set |

## Reproduce the analysis

```bash
python3 scripts/summarize.py artifacts > artifacts/summary.json
python3 scripts/matrix.py               > artifacts/matrix.json
```

For the original measurement (engine flags, +36 % throughput, TP=1 impossibility) see the
August folder; this one deliberately does not restate it.

## Gotchas

- **Give nonce files distinct basenames before comparing.** `compare_nonces.py` labels pairs by
  basename and silently compares a file with itself when two inputs share a name, reporting a
  false `L2 = 0.0000`. That happened during this analysis and was caught only by checking the
  `PAIR:` / `vs:` lines. `scripts/matrix.py` keys by folder and arm instead of by filename.

## Related

- the same checkpoint on 4×B200, `k3` image: [`../glm53-flash-nvfp4-libertai-4xb200/README.md`](../glm53-flash-nvfp4-libertai-4xb200/README.md)
- honest arm on this box: [`../glm53-flash-fp8-2xb300/README.md`](../glm53-flash-fp8-2xb300/README.md)
- original measurement: [`../../2026-08/glm53-flash-nvfp4-libertai-2xb300/README.md`](../../2026-08/glm53-flash-nvfp4-libertai-2xb300/README.md)
- campaign summary: [`../glm53-flash-cross-hardware-summary/README.md`](../glm53-flash-cross-hardware-summary/README.md)

## Reproducibility checklist

- [x] A reader with only this folder can reproduce every table by running the two scripts.
- [x] Hardware, image and digest are stated exactly.
- [x] Provenance is explicit: data collected 2026-08-26, no new run.
- [x] Every script referenced is committed under `scripts/`.
- [x] No links to `.claude/...` and no paths into sibling repos.
- [x] All artifacts referenced exist in `artifacts/`, including the honest reference sets.
- [x] The known analysis trap (identical basenames) is documented.
- [ ] Engine logs are not duplicated here; they live in the August folder.
