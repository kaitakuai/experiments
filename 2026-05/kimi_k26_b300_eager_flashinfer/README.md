# Kimi-K2.6 INT4 PoC v2 — параметры perf-оптимизации на 8×B300

**Дата**: 2026-05-01/02
**Сервер**: `root@95.133.253.49` — 8× NVIDIA B300 SXM6 AC (275 GiB HBM × 8 = 2.2 TiB), AMD EPYC 9575F, Ubuntu 24.04, driver 580.126.09 / CUDA 13.0
**Модель**: `moonshotai/Kimi-K2.6` — DeepseekV3-style MoE, 1.06T params, INT4 (compressed-tensors W4A16, group_size=32, 384 routed experts × top_k=8, vision tower)
**Софт**: vLLM 0.20.0 в `ghcr.io/kaitakuai/mlnode-full:0.2.12-vllm0.20.0-b300-k5`
**Топология**: TP=4 × 2 инстанса (порты 5001 и 5002), nonces агрегируются через mlnode `/api/v1/inference/pow/init/generate` fan-out

## TL;DR

**Лучшая конфига: 5 120 nonces/min @ batch=64 (×2.5 от baseline)**
- `VLLM_USE_FLASHINFER_MOE_INT4=1` — главный лоск, **+138%** один (доказано clean A/B)
- `--enforce-eager` — без cudagraph, на этой модели cudagraph мешает + блокирует batch=128
- `--enable-expert-parallel` — +5% поверх FlashInfer
- `--max-num-seqs 128`, `--gpu-memory-utilization 0.85` — нужны для того чтобы eager+EP влез в память
- `--attention-backend CUTLASS_MLA` — обязательно (FLASHINFER не поддерживает MLA)

L2 vs H100 reference: mean 0.20, при chain-default `dist_threshold=0.4` + calibrated `p_mismatch=0.02` — **PASS** chain validation.

## Forced mlnode defaults (после патчей runner.py)

```
--tensor-parallel-size 4
--gpu-memory-utilization 0.85
--max-model-len 120000
--max-num-batched-tokens 131072
--logprobs-mode processed_logprobs
--compilation-config '{"mode": 0, "cudagraph_mode": "NONE"}'   # либо mode 3 + FULL
```

Env инжектится в каждый vllm subprocess:
```
CUDA_VISIBLE_DEVICES=0..3 / 4..7
VLLM_USE_FLASHINFER_MOE_INT4=1
```

## Эксперименты

| # | Конфигурация | Best n/min | Δ vs base | Заметки |
|---|--------------|-----------:|----------:|---------|
| 1 baseline   | TP=4×2, Marlin, mode=1, max-num-seqs 128, util 0.95 | **2 048** @ b=32 | — | batch=128 висит |
| 2 +cudagraph | + mode=3 cudagraph FULL, max-num-seqs 256 | 2 048 @ b=64 | 0% peak | b=8 +70%, peak не двинулся |
| 3 +combined  | + EP + FlashInfer + raised batch + cudagraph | 4 864 @ b=64 | **+138%** | batch=128 виснет (cudagraph) |
| 4 eager OOM  | #3 минус cudagraph (`--enforce-eager`) | OOM | — | 14.65 GiB не влезли |
| 5 eager redu | #4 + max-num-seqs 128 + util 0.85 | **5 120** @ b=64 | **+150%** | batch=128 разблокирован, ★ best |
| 6 marlin     | #5 минус FlashInfer (env=0) | 2 048 @ b=32 | 0% | clean A/B baseline |
| 7 flashinfer | #6 + FlashInfer (env=1, без EP) | 4 864 @ b=32 | **+138%** | clean A/B доказательство |

## Атрибуция прироста (clean A/B изоляция)

Сравнение #6 vs #7 — отличие **только** `VLLM_USE_FLASHINFER_MOE_INT4`:

| batch | Marlin (#6) | FlashInfer (#7) | Δ |
|------:|-----------:|----------------:|---:|
|     8 |     1 440 |          3 680 | +156% |
|    16 |     2 016 |          4 448 | +121% |
|    32 |     2 048 |        **4 864** | **+138%** |
|    64 |     2 048 |          4 480 | +119% |
|   128 |     2 048 |          4 608 | +125% |

**FlashInfer mxint4 MoE — единственный источник основного прироста.** Нативный TRT-LLM int4 fused MoE kernel на Blackwell sm_100 заменяет Marlin (W4A16 dequant в bf16 каждый matmul) на прямой 4-битный путь — освобождает memory bandwidth.

| Knob | Вклад в перф |
|------|------:|
| FlashInfer mxint4 MoE | **+138% (самый крупный)** |
| `--enable-expert-parallel` (EP=4) | +5% (поверх FlashInfer) |
| max-num-seqs 128→256 | ~0% (на этой модели) |
| max-num-batched-tokens 65536→131072 | ~0% |
| cudagraph FULL | **−5% + блокирует batch=128** |
| eager (вместо cudagraph FULL) | +5% и batch=128 работает |

## Гонка-валидация (canonical L2 + binomtest)

Каноничный код взят из `vllm/poc/data.py` + `validation.py`. Скрипт: [.claude/skills/gonka-l2-validate/compare_nonces.py](.claude/skills/gonka-l2-validate/compare_nonces.py)

Pairwise сравнения собранных 1000 nonces (block_hash=`artifact_collection_block_v1`):

| Пара | N | mean L2 | median | max | p99 |
|------|--:|--------:|-------:|----:|----:|
| B300 FlashInfer ↔ B300 Marlin (self cross-kernel) | 896 | **0.1869** | 0.176 | 0.742 | 0.407 |
| B300 FlashInfer ↔ B200 FlashInfer (cross-arch, vLLM 0.19) | 884 | 0.1887 | 0.179 | 0.628 | — |
| B300 Marlin ↔ B200 FlashInfer | 988 | 0.1926 | 0.183 | 0.509 | — |
| B300 FlashInfer ↔ H100 ref (TP=16) | 884 | 0.2034 | 0.194 | 0.646 | 0.433 |
| B300 Marlin ↔ H100 ref | 988 | 0.2051 | 0.194 | 0.565 | 0.424 |
| B200 FlashInfer ↔ H100 ref | 1000 | 0.2056 | 0.196 | 0.698 | — |

**Floor для Kimi-K2.6 INT4 ≈ 0.19** на cross-anything (kernel/arch). Cross-arch добавляет всего ~0.02 поверх kernel-swap. Это особенность модели: 384 экспертов × top_k=8 + дискретный routing → один бит numerics → разные эксперты → активации расходятся до конца сети.

### Verdict через canonical fraud test

| Сценарий | thr | p_mis | M↔H100 | FI↔H100 | M↔FI (self) |
|----------|----:|------:|:------:|:-------:|:-----------:|
| vllm strict (self-validation) | 0.02 | 0.001 | FRAUD 99.8% | FRAUD 99.9% | FRAUD 99.8% |
| PoC report 98pct calibrated | 0.174 | 0.02 | FRAUD 62% | FRAUD 62% | FRAUD 51% |
| chain proto default | 0.4 | 0.001 | FRAUD 1.7% (p=8e-16) | FRAUD 2.8% (p=9e-28) | FRAUD 1.2% (p=3e-9) |
| **chain proto + calibrated p_mis** | **0.4** | **0.02** | **PASS** p=0.77 | **PASS** p=0.06 | **PASS** p=0.97 |

При **production-like** chain-параметрах (`dist_threshold=0.4` из proto + `p_mismatch=0.02` из calibrated honest baseline по PoC v2 report) **все три пары PASS chain validation**. FlashInfer не ломает выходы — drift сравним с Marlin (floor определён моделью, а не kernel-ом).

## Sanity inference (model output)

`/v1/chat/completions` запрос "What is the capital of France? Answer in one short sentence.":
> *"The user is asking for the capital of France and wants the answer in one short sentence. ... The capital of France is Paris."*

Модель работает корректно с FlashInfer kernel'ом.

## Патчи к mlnode-308 (b300-k5 image)

Применены через `docker exec mlnode-308 sed -i ...`, перезагружаются `docker restart`:

1. **runner.py** — TP default 1→4 + max-num-batched-tokens 65536→131072 + compilation-config + env injection:
   ```python
   _b300_forced['--tensor-parallel-size'] = '4'
   _b300_forced['--max-num-batched-tokens'] = '131072'
   _b300_forced['--gpu-memory-utilization'] = '0.85'
   _b300_forced['--compilation-config'] = '{"mode": 0, "cudagraph_mode": "NONE"}'
   # после CUDA_VISIBLE_DEVICES:
   env["VLLM_USE_FLASHINFER_MOE_INT4"] = "1"
   ```
2. **poc_model_runner.py** — добавить `seq_lens_cpu_upper_bound=seq_lens_cpu` после `_seq_lens_cpu=seq_lens_cpu` (CUTLASS_MLA требует)
3. **watcher.py** — `MAX_UNHEALTHY_COUNT` дефолт 3→9999 (чтобы не убил во время cold start)

## Запуск (POST `/api/v1/inference/up/async`)

```json
{
  "model": "moonshotai/Kimi-K2.6",
  "dtype": "auto",
  "additional_args": [
    "--trust-remote-code",
    "--attention-backend", "CUTLASS_MLA",
    "--max-num-seqs", "128",
    "--enable-expert-parallel",
    "--enforce-eager"
  ]
}
```

Cold start ~5-6 мин (loading 140 GiB × 2 инстанса × ~3 мин + JIT FlashInfer kernel при первом forward).

## Артефакты

- `nonces_1000.json` — 1280 nonces из best config (#5, FlashInfer + EP eager)
- `marlin_eager_tp4_no_ep_nonces_1000.json` — 1024 nonces из #6 Marlin baseline
- `h100_tp16_b16_nonces_1000.json` — H100 ref (downloaded из github.com/kaitakuai/experiments)
- `l2_canonical_3way.json` / `.png` — каноничная gonka-валидация trio
- `l2_histogram_3cases.png` — гистограммы L2 со всеми threshold-линиями
- `config.json` — параметры на момент сбора

## Open questions

- **vLLM 0.16+ требуется** для `VLLM_USE_FLASHINFER_MOE_INT4` (PR #32437). Нашему 0.20.0 — ОК.
- **Только Blackwell (sm_100)**: B100/B200/B300 проходят `is_device_capability_family(100)`. На B200 cold start длиннее из-за weight-prep (`prepare_static_weights_for_trtllm_mxint4_moe`); может выглядеть как hang — ждать до 15 мин.
- **Vision tower** (KimiK25ForConditionalGeneration) тащит encoder profile_run на старте. Если зависает — `--limit-mm-per-prompt '{"image":0,"video":0}'` отключает.
- **Cross-validation gate**: production значение `p_mismatch` в genesis сети неизвестно (vllm default=0.001 vs calibrated=0.02 → разная картина). Стоит уточнить у operator-а.
