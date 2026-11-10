# BigCodeBench (Singularity)

`inspect_evals/bigcodebench_singularity`: upstream BigCodeBench (completion) with the generated code executed in a
Singularity sandbox instead of Docker. It reuses upstream's dataset and `verify()` scorer and runs the tests inside
the same prebuilt image upstream's Docker sandbox uses, `ghcr.io/arcadiaimpact/inspect-eval-bigcodebench:latest`.
See the module docstring for why upstream's `bigcodebench()` cannot be called directly.

## Pre-pulling the image (do this once)

On first use the Singularity provider pulls the image into `$INSPECT_SINGULARITY_SIF_DIR` (see
[sandboxed/README.md](../../sandboxed/README.md)). Letting the eval do this on a system using distributed storage,
such as Lustre, tends to fail: the image unpacks to about 5.5 GB of small files, and `singularity pull` unpacks and
repacks it (`mksquashfs`) in `$SINGULARITY_TMPDIR`, this might be a bootleneck.

Pull it once by hand instead, to have better control on how the image is pulled and unpacked, using a command like:

```bash
uv run python -m custom.bigcodebench_singularity.bigcodebench_singularity
```

while being sure to set the right folder for `$INSPECT_SINGULARITY_SIF_DIR` and `$SINGULARITY_TMPDIR`.
