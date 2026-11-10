# SWE-bench (Singularity)

`inspect_evals/swe_bench_singularity`: upstream SWE-bench, with each instance run in a Singularity container built
from its DockerHub image. Images are cached as `.sif` in `$INSPECT_SINGULARITY_SIF_DIR` (or the `sif_dir` task arg).

## Pre-pulling the images

500 images, about 1.1 GB each. Pre-pulling them ensures that the evaluation runs smoothly without repeatedly
downloading the images or encountering delays during the evaluation.

```bash
uv run -m custom.swe_bench_singularity.swe_bench_singularity --jobs 6 2>&1 | tee -a swebench_sif_pull.log
```
while being sure to set the right folder for `$INSPECT_SINGULARITY_SIF_DIR`.
