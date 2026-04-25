"""Data quality inspection for K2.6 exp1 (no sentinels)."""
import json
from collections import Counter
import numpy as np

PATH = "artifacts/honest_b200_h200.jsonl"

n_items = 0
n_token_mismatch = 0
n_positions_inf = 0
n_positions_sentinel = 0
n_positions_zero_chosen = 0
n_positions_normal = 0
all_chosen_lps = []
all_alt_lps = []
langs = Counter()

sample_items = []
with open(PATH, "r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        n_items += 1
        langs[d.get("language", "?")] += 1
        inf = d.get("inference_result", {})
        val = d.get("validation_result", {})
        inf_toks = [p["token"] for p in inf.get("results", [])]
        val_toks = [p["token"] for p in val.get("results", [])]
        if inf_toks != val_toks:
            n_token_mismatch += 1
        for pos in inf.get("results", []):
            lps = list(pos["logprobs"].values())
            chosen = pos["logprobs"].get(pos["token"])
            alts = [v for k, v in pos["logprobs"].items() if k != pos["token"]]
            if chosen is not None:
                all_chosen_lps.append(chosen)
            all_alt_lps.extend(alts)
            if any(v == float("-inf") for v in lps):
                n_positions_inf += 1
            elif any(v <= -9000 for v in lps):
                n_positions_sentinel += 1
            elif chosen == 0.0:
                n_positions_zero_chosen += 1
            else:
                n_positions_normal += 1
        if len(sample_items) < 2:
            sample_items.append(d)

print(f"== Data quality report (K2.6 exp1 no-sentinels) ==")
print(f"Items: {n_items}")
print(f"Languages: {dict(langs)}")
print(f"Token mismatches: {n_token_mismatch} ({100*n_token_mismatch/n_items:.1f}%)")
total_pos = n_positions_inf + n_positions_sentinel + n_positions_zero_chosen + n_positions_normal
print(f"\nTotal positions: {total_pos}")
print(f"  With -inf logprob: {n_positions_inf} ({100*n_positions_inf/total_pos:.1f}%)")
print(f"  With sentinel -9999: {n_positions_sentinel} ({100*n_positions_sentinel/total_pos:.1f}%)")
print(f"  With chosen=0.0 exactly: {n_positions_zero_chosen} ({100*n_positions_zero_chosen/total_pos:.1f}%)")
print(f"  Normal (chosen != 0, no sentinel): {n_positions_normal} ({100*n_positions_normal/total_pos:.1f}%)")

print(f"\n== Chosen token logprob distribution (n={len(all_chosen_lps)}) ==")
arr = np.array(all_chosen_lps)
print(f"  mean={arr.mean():.4f}  median={np.median(arr):.4f}  min={arr.min():.4f}  max={arr.max():.4f}")
print(f"  % at 0.0 exactly: {100*(arr == 0.0).sum()/len(arr):.1f}%")
print(f"  % above -0.01 (very confident): {100*(arr > -0.01).sum()/len(arr):.1f}%")
print(f"  % below -5 (unconfident): {100*(arr < -5).sum()/len(arr):.1f}%")

print(f"\n== Alternative tokens logprob distribution (n={len(all_alt_lps)}) ==")
arr = np.array(all_alt_lps)
print(f"  mean={arr.mean():.4f}  median={np.median(arr):.4f}  min={arr.min():.4f}  max={arr.max():.4f}")
print(f"  percentiles: 1%={np.percentile(arr, 1):.2f}  5%={np.percentile(arr, 5):.2f}  25%={np.percentile(arr, 25):.2f}  50%={np.percentile(arr, 50):.2f}  75%={np.percentile(arr, 75):.2f}  95%={np.percentile(arr, 95):.2f}")

print(f"\n== Sample 1 first 2 positions ==")
for i, pos in enumerate(sample_items[0]["inference_result"]["results"][:2]):
    print(f"  pos[{i}] token='{pos['token']}' logprobs={pos['logprobs']}")
