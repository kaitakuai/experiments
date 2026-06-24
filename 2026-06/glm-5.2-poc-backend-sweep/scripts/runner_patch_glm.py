"""GLM-5.2 hardcodes for MLNode's in-image runner.py.

Same additive/idempotent pattern as the foundry image's own
`tools/runner-patches/b300-kimi.py`: insert a flag block right after
`self.additional_args = additional_args or []` in `VLLMRunner.__init__`,
so the image runs OUR GLM-5.2 config regardless of what chain epoch_models
broadcasts. No upstream lines are removed; fails loud if the marker is gone
(= upstream refactored, re-verify).

Unlike the baked Kimi patches, this one is PARAMETERIZED via environment
variables so a single committed script covers every cell of the sweep
(FP8 vs AWQ, TP=8 vs TP=4, deepgemm vs auto vs cutlass). The wrapper
`02_start_vllm.sh` exports these before invoking the patch.

Inputs (env, with defaults matching scripts/config.env):
    TP                    tensor-parallel size            (default 8)
    GPU_MEM_UTIL          --gpu-memory-utilization        (default 0.92)
    MAX_MODEL_LEN         --max-model-len                 (default 131072)
    MAX_NUM_SEQS          --max-num-seqs                  (default 128)
    KV_CACHE_DTYPE        --kv-cache-dtype                (default fp8_e4m3)
    GLM_TOOL_PARSER       --tool-call-parser              (default glm45)  TODO verify in-image
    GLM_REASONING_PARSER  --reasoning-parser              (default glm45)  TODO verify in-image
    MODEL_ROLE            "fp8" | "awq"   (awq adds --quantization awq_marlin)
    RUNNER_FILE           path to runner.py in the image
                          (default /app/packages/api/src/api/inference/vllm/runner.py)

MoE / attention backend selection is done via ENVIRONMENT VARIABLES at the
process level (VLLM_MOE_USE_DEEP_GEMM, VLLM_USE_FLASHINFER_MOE_FP8,
VLLM_ATTENTION_BACKEND), NOT here — 02_start_vllm.sh exports them. This patch
only injects the vLLM CLI flags that cannot be expressed as env vars.

Idempotent: re-running on an already-patched file is a no-op.
"""

from __future__ import annotations

import os
import re
import sys

FILE = os.environ.get(
    "RUNNER_FILE", "/app/packages/api/src/api/inference/vllm/runner.py"
)
MARKER = "self.additional_args = additional_args or []"
TAG = "Kaitaku GLM-5.2 hardcodes"
BEGIN = f"# --- {TAG} (scripts/runner_patch_glm.py) ---"
END = f"# --- end {TAG} ---"
# This image bakes a b300-kimi hardcode block AFTER the marker. Our block MUST
# run after it so GLM flags (TP, parsers, attention) win. We anchor on the kimi
# block's end marker; fall back to the plain marker if it's absent.
KIMI_END_RE = re.compile(r"# --- end Kaitaku B300-[A-Za-z0-9]+.*?hardcodes ---")
INDENT = " " * 8


def _flags() -> tuple[list[tuple[str, str]], list[str], list[str]]:
    tp = os.environ.get("TP", "8")
    gmu = os.environ.get("GPU_MEM_UTIL", "0.92")
    mml = os.environ.get("MAX_MODEL_LEN", "131072")
    seqs = os.environ.get("MAX_NUM_SEQS", "128")
    kv = os.environ.get("KV_CACHE_DTYPE", "fp8_e4m3")
    tool = os.environ.get("GLM_TOOL_PARSER", "glm45")
    reason = os.environ.get("GLM_REASONING_PARSER", "glm45")
    role = os.environ.get("MODEL_ROLE", "fp8")
    comp = os.environ.get("COMPILATION_CONFIG", "").strip()
    mnbt = os.environ.get("MAX_NUM_BATCHED_TOKENS", "").strip()

    forced = [
        ("--tensor-parallel-size", tp),
        ("--gpu-memory-utilization", gmu),
        ("--max-model-len", mml),
        ("--max-num-seqs", seqs),
        ("--kv-cache-dtype", kv),
        ("--logprobs-mode", "processed_logprobs"),  # PoC v2 correctness
        ("--tool-call-parser", tool),
        ("--reasoning-parser", reason),
    ]
    # Override the image's baked compilation-config (Kimi block hardcodes eager
    # mode 0). When COMPILATION_CONFIG is set we force CUDA graphs instead.
    if comp:
        forced.append(("--compilation-config", comp))
    # Override the image's baked max-num-batched-tokens (Kimi: 131072). On memory-
    # tight nodes (H200 141 GiB + bf16 KV for GLM DSA) the profiling peak scales
    # with this; lowering it frees KV cache (avoids "Available KV cache: -X GiB").
    if mnbt:
        forced.append(("--max-num-batched-tokens", mnbt))
    # NOTE: cyankiwi/GLM-5.2-AWQ-INT4 is packaged as *compressed-tensors* (W4A16),
    # not classic AWQ. vLLM auto-detects the quant method from the model's
    # config.json, so we must NOT force --quantization (forcing awq_marlin raises
    # "quant method (compressed-tensors) does not match (awq_marlin)"). Remove any
    # stale --quantization the base image may have injected.
    remove_quant = role == "awq"

    flags_only = [
        "--trust-remote-code",
        "--enable-auto-tool-choice",
        "--enable-expert-parallel",  # EP for the 256-expert MoE
    ]
    # Strip Kimi-specific flags that don't apply to GLM-5.2 (text-only, not MLA):
    #   --attention-backend CUTLASS_MLA  -> GLM is not MLA; backend via env
    #   --mm-encoder-tp-mode            -> no vision encoder
    # NOTE: --worker-extension-cls gonka_poc.worker.PoCWorkerExtension (set by the
    # baked Kimi block) is REQUIRED for PoC generation — we intentionally keep it.
    remove = [
        "--attention-backend", "CUTLASS_MLA",
        "--enforce-eager",
        "--mm-encoder-tp-mode", "data",
    ]
    if remove_quant:
        remove += ["--quantization", "awq_marlin", "compressed-tensors"]
    return forced, flags_only, remove


def _injection() -> str:
    forced, flags_only, remove = _flags()
    lines = ["", BEGIN]
    lines.append("_glm_forced = [")
    for k, v in forced:
        lines.append(f"    ({k!r}, {v!r}),")
    lines.append("]")
    lines.append(f"_glm_flags = {flags_only!r}")
    lines.append(f"_glm_remove = {remove!r}")
    lines.append("for _f, _v in _glm_forced:")
    lines.append("    if _f in self.additional_args:")
    lines.append("        self.additional_args[self.additional_args.index(_f) + 1] = _v")
    lines.append("    else:")
    lines.append("        self.additional_args.extend([_f, _v])")
    lines.append("for _f in _glm_flags:")
    lines.append("    if _f not in self.additional_args:")
    lines.append("        self.additional_args.append(_f)")
    lines.append("for _r in _glm_remove:")
    lines.append("    while _r in self.additional_args:")
    lines.append("        self.additional_args.pop(self.additional_args.index(_r))")
    lines.append(END)
    return "".join((INDENT + ln + "\n") if ln else "\n" for ln in lines)


def _strip_existing(src: str) -> str:
    """Remove a previously-injected GLM block (self-healing / order fix)."""
    if BEGIN not in src:
        return src
    out, skipping = [], False
    for line in src.splitlines(keepends=True):
        if BEGIN in line:
            skipping = True
            # also drop a single blank line we inserted before BEGIN
            if out and out[-1].strip() == "":
                out.pop()
            continue
        if skipping:
            if END in line:
                skipping = False
            continue
        out.append(line)
    return "".join(out)


def main() -> int:
    with open(FILE) as f:
        src = f.read()

    if MARKER not in src:
        sys.stderr.write(
            f"ERROR: GLM patch: marker {MARKER!r} not found in {FILE}. "
            "Upstream runner.py may have been refactored — re-verify the patch.\n"
        )
        return 1

    src = _strip_existing(src)  # remove any stale GLM block, then re-insert correctly

    # Insert AFTER the baked Kimi hardcode block so GLM flags win; else after marker.
    anchor_end = None
    for m in KIMI_END_RE.finditer(src):
        anchor_end = m  # last kimi end marker
    out, inserted = [], False
    if anchor_end:
        # insert right after the line containing the kimi end marker
        idx = src.index("\n", anchor_end.end())
        out = [src[: idx + 1], _injection(), src[idx + 1 :]]
        inserted = True
    else:
        lines = []
        for line in src.splitlines(keepends=True):
            lines.append(line)
            if not inserted and MARKER in line:
                lines.append(_injection())
                inserted = True
        out = lines
    if not inserted:
        sys.stderr.write("ERROR: GLM patch: no insertion anchor found\n")
        return 1

    with open(FILE, "w") as f:
        f.write("".join(out))
    where = "after Kimi block" if anchor_end else "after marker"
    print(f"runner.py patched for GLM-5.2 ({where}; role={os.environ.get('MODEL_ROLE','fp8')}, "
          f"TP={os.environ.get('TP','8')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
