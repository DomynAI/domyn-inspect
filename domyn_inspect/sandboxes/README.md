# `sandboxes`

Sandbox providers for `inspect_ai`. Importing `domyn_inspect.sandboxes` (done by `domyn_inspect/cli.py`) registers the
provider in each `sandboxes/<provider>/` package. Today that is `singularity`, for clusters without Docker.

```python
task = humaneval(sandbox="singularity")
task = humaneval(sandbox=("singularity", "docker://python:3.12-slim-bookworm"))
```

## Layout

| File | Purpose |
|---|---|
| `singularity/config.py` | Config model (`SingularitySandboxConfig`) and `resolve_config()`. |
| `singularity/images.py` | Pull + cross-run `.sif` cache, file-locked. |
| `singularity/environment.py` | Provider registration, CLI invocation, lifecycle, `exec`, file transfer. |

Tasks using this provider live in `domyn_inspect/tasks/` (e.g. `swe_bench_singularity`, `scicode_singularity`).

## Lifecycle of a sandbox

One Singularity **instance** per sample, started from a read-only image and given two throwaway writable areas: a
writable layer over `/`, and a work directory for `/tmp` and `/var/tmp`. Both are deleted when the sample ends, so
nothing a sample writes reaches the image or another sample.

1. **Image (once per task).** `task_init` resolves the image to a local file. `sif=` uses a `.sif` or sandbox directory
   as is, and fails the task if it is missing. `image="docker://..."` is pulled once into the `.sif` cache and reused by
   every later sample and run. Pulls are file-locked, and written to `.partial.<pid>` then renamed, so a killed pull
   never leaves a broken `.sif`. The cache key is the reference string: set `INSPECT_SINGULARITY_SIF_REFRESH=1` to
   re-pull a moved `:latest` tag.
2. **Writable areas (per sample).** A layer over `/` (`--writable-tmpfs`, or an ext3 overlay file of `overlay_size_mb`
   MB) and a work directory for `/tmp` and `/var/tmp` (`--workdir`), both in `$INSPECT_SINGULARITY_OVERLAY_DIR`; see
   [Where writes go](#where-writes-go). Leftovers from a killed predecessor with the same name are removed first.
3. **Copy-out.** Each `copy_out` path is copied out of the image into the work directory and bind-mounted back over
   itself, so the host user can write it even when the image was built for root.
4. **Start.** `singularity instance start --containall --workdir <work> (--overlay <img> | --writable-tmpfs)
   [--bind <copy>:<path> ...] [--net --network none] <sif> <instance>`. `--containall` hides the host's `$HOME` and
   shared filesystems and starts from a clean environment. `--network none` (unless `allow_internet`) leaves only
   `lo`.
5. **Setup.** Each `setup` command runs once via `sh -c`. A failure stops the instance and fails the sample.
6. **Exec.** Every `exec`, `read_file` and `write_file` is a `singularity exec --cleanenv instance://<instance> ...`.
   `--cleanenv` is needed on each call: `--containall` only cleans the environment at start, and without it a host
   variable such as `TMPDIR=/opt/...` leaks in and points at a path that does not exist inside. The
   sandbox's own `env` is passed as `SINGULARITYENV_*`.
7. **Cleanup.** `sample_cleanup` stops the instance, then deletes the overlay file and the work directory. At task end,
   any instance still running is stopped; `inspect sandbox cleanup singularity` sweeps any that outlived their process.

### Where writes go

| Path in the container | `overlay_size_mb = None` (default) | `overlay_size_mb = N` | Sandbox directory on Lustre |
|---|---|---|---|
| `/` (e.g. `/testbed`) | `--writable-tmpfs`, **64 MB** | ext3 overlay, **N MB** | **read-only** |
| `/tmp`, `/var/tmp` | work dir | work dir | work dir |

- **`writable-tmpfs` size limit.** `--writable-tmpfs` lives in Singularity's session directory, capped by
  `sessiondir max size` in `singularity.conf` and not settable per run. The capped size might be fine for writing and running
  one file, but too small for installing packages or running a project's test suite, which then fail with ENOSPC. Set
  `overlay_size_mb` for those (SWE-bench uses 4096).
- **`/tmp` and `/var/tmp` need their own work dir.** Under `--containall` they are mounted separately from `/`, so a
  layer over `/` never covers them: without `--workdir` they share the same 64 MB, even when `/` has a large overlay.
  Inspect's tool helper (`inspect-sandbox-tools`, ~15 MB, unpacking ~20 MB into `/tmp` per call) then does not fit, and
  the bash tool fails with "Failed to extract libpython".
- **A sandbox directory on Lustre gets no writable layer.** The kernel's overlayfs cannot use Lustre as its lower layer,
  so Singularity silently drops the overlay ("Fallback to underlay layer") and `/` is read-only in either mode. Only
  `/tmp` and `/var/tmp` stay writable. `scicode_singularity` relies on this: its scorer only execs, from `/`.
- **`/dev/shm` is RAM.** Everything a sample writes to its overlay or `/tmp` uses node memory until cleanup. With many
  concurrent samples, cap them with `INSPECT_SINGULARITY_MAX_SANDBOXES`, or point `INSPECT_SINGULARITY_OVERLAY_DIR` at
  local disk.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `INSPECT_SINGULARITY_SIF_DIR` | `./.cache/singularity_sif` | `.sif` cache, shared across runs. |
| `INSPECT_SINGULARITY_SIF_REFRESH` | unset | `1` re-pulls images already in the cache. |
| `INSPECT_SINGULARITY_OVERLAY_DIR` | `/dev/shm` | Per-sample overlay files and work directories. |
| `INSPECT_SINGULARITY_MAX_SANDBOXES` | inspect's default | Maximum number of concurrent sandboxes. |

## Tests

```bash
uv run pytest tests/singularity/test_sandbox.py -q
```

Live, opt-in checks against a real `singularity` binary and an image with `python3` (`test_testbed_is_writable` also
needs a SWE-bench image):

```bash
SINGULARITY_LIVE_TEST_SIF=/path/to/some.sif uv run pytest tests/singularity/test_live.py -q
```
