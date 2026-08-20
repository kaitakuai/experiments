#!/usr/bin/env python3
"""V1: a replaying request must not be scheduled with draft tokens.

The V1 RejectionSampler has no enforced-token hook at all -- on the speculative
path only the *bonus* token passes through the plain Sampler (which does apply
enforced ids), so accepted drafts are emitted unenforced ahead of it. Worse,
when a draft is accepted the engine books two emitted tokens while the reply
carries one, so len(req_output_token_ids) -- the index
_build_enforced_tensor reads -- runs ahead of the output and the replay skips
an enforced id.

Post-processing the sampler output cannot fix that: the accept count is
produced inside rejection_sample and is what the bookkeeping follows. Measured,
two such attempts landed at 0.136 (one token always dropped) and 0.166 (the
error smeared across the sequence), against a 0.0185 floor.

So take the drafts away from replaying requests instead, at scheduling time,
before they enter the forward pass. Then num_draft_tokens is 0 for that
request, it takes the plain Sampler path where enforced ids are applied, and
the accept count and the emitted count agree by construction. Draft counts are
per-request on V1 -- the runner's own example is [3, 0, 2, 0, 1] -- so other
requests in the same batch keep speculating at full speed.

Zeroing num_draft_tokens later, in gpu_model_runner, would NOT work: the draft
positions are already in the forward pass by then and logits_indices would
misalign.
"""

import py_compile
import sys

SCHED = "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"

bak = SCHED + ".nospecbak"
try:
    with open(bak) as f:
        src = f.read()
except FileNotFoundError:
    with open(SCHED) as f:
        src = f.read()
    with open(bak, "w") as f:
        f.write(src)


def sub(s, old, new, label):
    if old not in s:
        sys.exit(f"ANCHOR NOT FOUND: {label}")
    if s.count(old) != 1:
        sys.exit(f"ANCHOR NOT UNIQUE ({s.count(old)}x): {label}")
    return s.replace(old, new, 1)


# 1. running requests: drop proposed drafts for replaying requests
OLD1 = """            # Speculative decode related.
            if request.spec_token_ids:"""

NEW1 = """            # Speculative decode related.
            # gonka PoC v2: a replaying request never speculates. The V1
            # RejectionSampler has no enforced-token hook, and an accepted
            # draft desynchronises the replay index from the emitted output.
            if request.spec_token_ids and _gonka_is_replay(request):
                request.spec_token_ids = []
            if request.spec_token_ids:"""

src = sub(src, OLD1, NEW1, "running-request spec attach")

# 2. the chunked-prefill padding path must skip them too
OLD2 = """                if pad_spec_decode:
                    scheduled_spec_decode_tokens[request_id] = [
                        -1
                    ] * self.num_spec_tokens"""

NEW2 = """                if pad_spec_decode and not _gonka_is_replay(request):
                    scheduled_spec_decode_tokens[request_id] = [
                        -1
                    ] * self.num_spec_tokens"""

src = sub(src, OLD2, NEW2, "pad_spec_decode path")

# 3. the predicate itself, module level
OLD3 = """class Scheduler(SchedulerInterface):"""
NEW3 = '''def _gonka_is_replay(request) -> bool:
    """True when the request replays a pinned token sequence (validation)."""
    params = getattr(request, "sampling_params", None)
    return bool(getattr(params, "enforced_token_ids", None))


class Scheduler(SchedulerInterface):'''

src = sub(src, OLD3, NEW3, "predicate definition")

with open(SCHED, "w") as f:
    f.write(src)
py_compile.compile(SCHED, doraise=True)
print(f"patched V1 scheduler: replaying requests get no drafts -- {SCHED}")
