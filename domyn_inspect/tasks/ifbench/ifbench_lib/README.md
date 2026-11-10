# ifbench_lib

Scoring glue for the IFBench task (`../ifbench.py`), taken from [allenai/IFBench](https://github.com/allenai/IFBench)
at commit [`1c40f0c`](https://github.com/allenai/IFBench/tree/1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d) (Apache-2.0,
see `../LICENSE`).

- The checkers (`instructions`, `instructions_registry`, `instructions_util`) are not copied here. They come from the
  upstream `ifbench` package, installed from git at that commit (`pyproject.toml`, `[tool.uv.sources]`).
- `evaluation_lib.py` is upstream's `evaluation_lib.py`, which is not part of the package, cut down to the strict and
  loose scoring functions. It has two local edits:
  1. it imports `instructions_registry` from the `ifbench` package;
  2. checker calls go through `_check_following`, which counts a `csv.Error` as "not followed". The error comes from
     a malformed CSV, e.g. a field longer than csv's 128 KiB limit (a degenerate, repeating generation). Upstream lets
     the exception through, which aborts its run.
