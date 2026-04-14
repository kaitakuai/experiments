# experiments

Public repository for experiment artifacts produced by the [kaitaku.ai](https://github.com/kaitakuai) team.

## Purpose

This repository stores reproducible, shareable outputs of our GPU and inference experiments so that they can be referenced from pull requests, reports, and governance proposals without relying on ephemeral storage (Google Drive, pastebins, chat attachments).

Typical contents:

- Benchmark reports (PoC throughput, inference latency, cross-GPU comparisons)
- Configuration snapshots and startup commands used to reproduce a run
- Small artifacts referenced from reports: nonce vectors, logprobs samples, hardware/software environment dumps
- Research notes tied to a specific experiment (migration designs, regression investigations)

Large binary artifacts (>100 MB) are attached to GitHub Releases of this repository rather than committed to the tree.

## Layout

Experiments are grouped by month:

```
YYYY-MM/
  <experiment-name>/
    README.md       # report
    artifacts/      # optional: small data files
```

## Contributing

Any member of the kaitaku.ai team can open a PR with a new experiment folder. Reports should be self-contained: hardware, software versions, commands, and results in one place so that a reader does not need context from chat.
