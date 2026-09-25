#!/usr/bin/env python3
"""Per-corpus point rate and per-nonce flagged fraction on the tau grid, from the kit's val_/npz_ files.
usage: tau_cells.py <dir> [<dir> ...]"""
import glob, json, os, sys
import numpy as np
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "val_*_v*.json"))):
        v = json.load(open(f)); r = v["results"]
        nz = f.replace("/val_", "/npz_").replace(".json", ".npz")
        line = f"  {os.path.basename(d):8s} {v['meta'].get('ref', '?'):16s} points {r['total_points']:6d} diff {r['total_diff']:5d} = {r['rate_pct']:6.3f}%  step0 {r.get('step0_pct', 0):5.2f}%"
        if os.path.exists(nz):
            z = np.load(nz); taus = [round(float(t), 3) for t in z["taus"]]; c = z["counts"]
            fr = []
            for t in (0.0, 0.01, 0.02, 0.025, 0.03, 0.05):
                fr.append(f"{100 * float((c[taus.index(t)] > 0).mean()):5.1f}" if t in taus else "  n/a")
            line += "  nonces>tau @0/0.01/0.02/0.025/0.03/0.05: " + " ".join(fr)
        print(line)
