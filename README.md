<div align="center">

<!-- TODO: add the logos to static/ and drop the title below.
<img src="static/domyn-inspect-logo-primary.svg#gh-light-mode-only" alt="Domyn Inspect" height="100">
<img src="static/domyn-inspect-logo-white.svg#gh-dark-mode-only" alt="Domyn Inspect" height="100">
-->

# domyn-inspect

**Custom [inspect-ai](https://inspect.aisi.org.uk/) tasks, ready to run from domyn-eval.**

domyn-inspect makes it easier to integrate custom inspect-ai tasks with domyn-eval.

[![CI](https://github.com/DomynAI/domyn-inspect/actions/workflows/ci.yaml/badge.svg)](https://github.com/DomynAI/domyn-inspect/actions/workflows/ci.yaml)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-brightgreen)](pyproject.toml)
[![inspect-evals](https://img.shields.io/badge/inspect--evals-0.22.0-orange)](https://pypi.org/project/inspect-evals/0.22.0/)
[![License - Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://github.com/microsoft/pyright)

<a href="#quick-start">Quick start</a> •
<a href="#configuration">Configuration</a> •
<a href="#custom-tasks">Custom tasks</a> •
<a href="#offline-clusters">Offline clusters</a> •
<a href="#viewing-results">Viewing results</a>

</div>

<table>
<tr>
<td width="50%" valign="top">

### Configure

- **domyn-eval's model config**, read as is: endpoint, judge and think/nothink sampling
- **Judge endpoint** wired to the grader and user-simulator roles
- **Custom tasks** registered by adding a folder under `domyn_inspect/tasks/`
- **Dual scoring** (opt-in): upstream strict extraction next to a robust one

</td>
<td width="50%" valign="top">

### Run

- **Context-aware budgets**: `max_tokens` cut to what the prompt leaves of the window
- **Long-context truncation** inside the documents, keeping the question intact
- **Offline-ready**: prefetch datasets and check they load without internet
- **Docker-free sandboxes** on Singularity

</td>
</tr>
</table>

---

## Quick start

```bash
uv sync
source .venv/bin/activate
export INSPECT_LOG_DIR=results
```

### 1. Describe the model

Start from [`tests/fixtures/dummy_model.toml`](tests/fixtures/dummy_model.toml) and point it at your endpoint:

```toml
[model]
id = "domyn-edge"                # names the results directory
path = "domyn/Domyn-Edge-SFT"    # the served model name
url = "http://localhost:8000/v1"

[model.generation.context]
max_model_len = 32768            # required: the server's --max-model-len
```

### 2. Pick the tasks

```toml
[general]
name = "smoke"                   # names the results subdirectory

[[tasks]]
task = "inspect_evals/hello_world"

[[tasks]]
task = "inspect_evals/aime2024"
```

### 3. Check, then run

Without `--run`, domyn-inspect resolves every task and prints what it would run:

```console
$ python -m domyn_inspect --model model.toml --task tasks.toml
[DRY-RUN]  The following tasks will be passed to eval_set():

    1. inspect_evals/hello_world
    2. inspect_evals/aime2024

  Model:           domyn-edge
  Log dir:         results/domyn-edge/smoke
  Max tasks:       10
  Max connections: 64
  ...

Pass --run to execute.
```

Then run it. `--limit` caps the samples per task, which is useful for a first sanity check:

```bash
python -m domyn_inspect --model model.toml --task tasks.toml --run --limit 10
```

### 4. Look at the results

Logs land in `$INSPECT_LOG_DIR/<model id>/<task set name>/`. Progress is printed by inspect's `log` display, at log
level `info` unless `INSPECT_LOG_LEVEL` is set; scores are in the `.eval` logs. Open them with `inspect view`, as in
[Viewing results](#viewing-results), or load them with `inspect_ai.analysis.evals_df()`.

---

## Configuration

### Model config

The endpoint and generation settings. When run from domyn-eval, this is the run's config snapshot.

```toml
[general]
# request_timeout = 3600         # per-request read and connect timeout (s)
# max_retries = 0                # inspect's retries of a failed request
# num_threads = 64               # concurrent requests; default for [inspect].num_threads

[model]
id = "domyn-edge"                # results subdirectory
path = "domyn/Domyn-Edge-SFT"    # served model name; also the tokenizer, unless tokenizer_path is set
url = "http://<node>:9000/v1"

[model.generation]
reasoning = true                 # picks [model.generation.think] or [model.generation.nothink]

[model.generation.think]
temperature = 0.6
top_p = 0.95

[model.generation.nothink]
temperature = 0.7
top_p = 0.8

[model.generation.context]
max_model_len = 65536            # required: the server's --max-model-len
max_tokens = 0                   # cap sent with every request; 0 lets the server budget each one
# tokenizer_path = "..."         # counts prompt tokens for long-context truncation (default model.path)
# token_safety_margin = 256      # tokens kept free when estimating the available budget
# completion_reserve = 8192      # output room kept when long-context tasks truncate (default: 8192 think, 4096 nothink)

[judge]                          # optional: the grader, and the user simulator unless [user] is set
path = "Qwen/Qwen3.5-9B"
url = "http://<node>:9001/v1"

# [judge.generation.grader]      # the model-graded scorers' grader, over the judge's mode; the keys of the modes
# temperature = 0.0              # the default; a table keeps it unless it sets its own
#
# [judge.generation.context]
# max_tokens = 4096              # the default cap on each role's generation; 0 lets the server budget each

# [user]                         # optional: another model for tau2's user simulator
# path = "Qwen/Qwen3.5-4B"
# url = "http://<node>:9002/v1"
#
# [user.generation.user]         # the user simulator, over this model's mode
# temperature = 0.0              # the default, as tau2 asks of its user
# chat_template_kwargs = { enable_thinking = false }

[inspect]
num_threads = 100                # concurrent requests to the endpoint
```

The user simulator is played by `[user]`, else `[judge]`, else the model under test (self-play); the grader by
`[judge]`, else by the model under test. Each samples with its role's table of its model's generation, e.g.
`[user.generation.user]`, over that model's mode, capped by its `context.max_tokens`: 4096 by default for `[judge]` and
`[user]`, but the model's own in self-play.

Long-context tasks cut the middle of their documents when a prompt does not fit. The prompt may use
`max_model_len - token_safety_margin`, minus `max_tokens` when that is set below the window, else minus
`completion_reserve`.

<details>
<summary>Timeouts</summary>

There is no separate connect timeout: inspect uses the request timeout for both, i.e. `[inspect] request_timeout` of the
task config, else `general.request_timeout` of the model config (default 3600s). The effective value is
`max(request timeout, $INSPECT_HTTP_CONNECT_TIMEOUT)`, so:

- to change it, change `request_timeout`, which also changes how long a request may run;
- to make it *longer* than the request timeout, set `INSPECT_HTTP_CONNECT_TIMEOUT` (seconds);
- a connect timeout *shorter* than the request timeout is not possible without a custom `http_client`.

The judge roles get the same request and connect timeouts as the model under test.

</details>

<details>
<summary>CLI reference</summary>

```console
$ python -m domyn_inspect --help
Usage: python -m domyn_inspect [OPTIONS]

  Launch inspect-ai evaluations from split TOML configs.

Options:
  --task PATH         Path to the tasks TOML config.  [required]
  --model PATH        Path to the model TOML config.  [required]
  --run               Actually execute (default is dry-run).
  --limit INTEGER     Override the sample limit for all tasks.
  --run-id TEXT       Ignored. Still accepted because the parent driver passes
                      it; it used to name the results CSV.
  --log-dir TEXT      Where eval logs and results go (default:
                      $INSPECT_LOG_DIR/<model>/<task set name>).
  --inspect-key TEXT  Model-TOML section layered over [inspect] (e.g.
                      inspect_swebench), so a suite's own
                      num_threads/max_tasks apply. Passed by the parent
                      driver.
  --help              Show this message and exit.
```

</details>

---

## Custom tasks

Any `@task` in a folder under `domyn_inspect/tasks/` is registered at startup and can be listed in a task config by its
name:

```python
# domyn_inspect/tasks/hello_world/hello_world.py
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate


@task(name="inspect_evals/hello_world")
def hello_world():
    return Task(
        dataset=[Sample(input="Just reply with Hello World", target="Hello World")],
        solver=[generate()],
        scorer=exact(),
    )
```

Export it from the folder's `__init__.py`, and add its tests under `tests/custom_tasks/`. Tasks that need a sandbox can
use the Docker-free Singularity backend in [`domyn_inspect/sandboxes/`](domyn_inspect/sandboxes/README.md).

---

## Offline clusters

Compute nodes often have no internet access. Prefetch the datasets of a task set from a node that has it:

```bash
python -m domyn_inspect.prefetch.prefetch_datasets --tasks tasks.toml
```

Each task is built once online, which caches its dataset, then again with the Hugging Face offline flags set, as on a
compute node. The script exits 1 if any task fails either pass, so a job is not submitted only to fail later. Assets a
task fetches while it runs, rather than when it is built, are not covered.

---

## Viewing results

Browse the logs with the [inspect-ai log viewer](https://inspect.aisi.org.uk/log-viewer.html): start it with
`inspect view --log-dir results/`, or open the `.eval` files in VS Code with the
[Inspect AI extension](https://marketplace.visualstudio.com/items?itemName=ukaisi.inspect-ai).

## Contributing

Issues and pull requests are welcome. Install the hooks once per clone with `uv run pre-commit install`: each commit
then runs ruff and pyright on the changed files and the test suite (`uv run pytest`).
