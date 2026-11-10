# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""The tasks TOML and model TOML configs, and their loaders."""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


@dataclass
class TaskEntry:
    """A single benchmark task with optional task-level arguments.

    Attributes:
        task: Registry name, e.g. ``"inspect_evals/mmlu_0_shot"``.
        task_args: Keyword arguments forwarded to the ``@task`` function
            (equivalent to ``-T key=val`` on the CLI), passed with their TOML types.
        display_name: Optional human-readable label shown in logs and summaries.
            Useful when the same task appears multiple times with different args
            (e.g. MMLU language variants).
        generate: Optional per-task generation overrides.  Keys are the same
            as those accepted by ``inspect_ai.GenerateConfig``
            (temperature, top_p, max_tokens, …).
        message_limit: Optional cap on messages per sample, set on the ``Task``.
        pass_k: Optional k for pass@k.  Runs ``k`` epochs per sample and reduces
            them with ``mean`` plus ``pass_at_<v>`` for v in 1, 2, 4, …, k.
        dual_score: Add a ``loose_<scorer>`` scorer with robust answer extraction
            next to each upstream scorer that has one (see scoring.py).
    """

    task: str
    task_args: dict[str, Any] = field(default_factory=dict)
    display_name: str = ""
    generate: dict[str, object] = field(default_factory=dict)
    message_limit: int | None = None
    pass_k: int | None = None
    dual_score: bool = False


@dataclass
class TasksGeneralConfig:
    """Settings that govern how the eval-set is executed."""

    name: str
    max_tasks: int = 10
    retry_attempts: int = 0
    retry_wait: int = 30
    log_shared: bool = True
    fail_on_error: bool | float = True
    # Passed to inspect_ai.eval_set(fail_on_error=...):
    #   True  -> abort task on first sample/scorer error (inspect default)
    #   False -> never abort; errored samples are excluded from the metric
    #   0<x<1 -> abort only if that *proportion* of samples errors
    #   x>=1  -> abort only if that *count* of samples errors


@dataclass
class TasksInspectConfig:
    """Optional [inspect] block inside a tasks TOML (runner overrides)."""

    request_timeout: float | None = None  # request + connect timeout (s); overrides generation.request_timeout
    # Temperature for the tau2 "user" simulator role
    user_sim_temperature: float = 0.7
    # Per-recipe cap on generation max_tokens
    max_tokens: int | None = None


@dataclass
class TasksConfig:
    """Loaded from tasks.toml — task list + runner settings."""

    general: TasksGeneralConfig
    tasks: list[TaskEntry]
    inspect: TasksInspectConfig = field(default_factory=TasksInspectConfig)


# The model TOML: the [model], [judge], [generation] and [inspect] sections of domyn-eval's config, with the same
# names and defaults. Only the fields domyn-inspect reads are declared; the loader drops the rest.


@dataclass
class ModelConfig:
    """``[model]`` and ``[judge]``: a model and its endpoint.

    Attributes:
        id: The model's id, which keys its results directory.
        path: The weights, or a HF hub id; the model name sent to the endpoint.
        url: The OpenAI-compatible base URL; ``/v1`` is appended if missing.
    """

    id: str | None = None
    path: str = ""
    url: str | None = None

    def __post_init__(self) -> None:
        """Append ``/v1`` to the URL if missing; an empty URL becomes None."""
        if self.url:
            url = self.url.rstrip("/")
            self.url = url if url.endswith("/v1") else f"{url}/v1"
        else:
            self.url = None


@dataclass
class GenerationModeConfig:
    """``[generation.think]`` and ``[generation.nothink]``: per-mode params; None leaves one to the endpoint."""

    temperature: float | None = None
    top_p: float | None = None
    min_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    frequency_penalty: float | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    min_completion_tokens: int | None = None  # left for generation when long-context tasks truncate; None keeps none


DEFAULT_COMPLETION_RESERVE = 4096
DEFAULT_REASONING_COMPLETION_RESERVE = 8192


@dataclass
class GenerationConfig:
    """``[generation]``: the request budget and the sampling params of each mode."""

    reasoning: bool = True  # selects think over nothink as mode
    think: GenerationModeConfig = field(default_factory=GenerationModeConfig)
    nothink: GenerationModeConfig = field(default_factory=GenerationModeConfig)
    max_tokens: int = 0  # cap sent with every request; 0 lets the server budget each one
    max_model_len: int = 0  # required: the server's --max-model-len
    token_safety_margin: int = 256  # tokens kept free when estimating the available token budget
    request_timeout: int = 3600  # per-request read and connect timeout (s)
    max_retries: int = 0  # inspect's retries of a failed request
    num_threads: int = 64  # concurrent requests; [inspect].num_threads defaults to it
    tokenizer_path: str | None = None

    def __post_init__(self) -> None:
        """Reject a missing ``max_model_len``."""
        if not self.max_model_len or self.max_model_len <= 0:
            msg = (
                f"generation.max_model_len={self.max_model_len}: set it to the server's --max-model-len, "
                "which every request and the long-context truncation are budgeted against"
            )
            raise ValueError(msg)

    @property
    def mode(self) -> GenerationModeConfig:
        """The params of the selected mode."""
        return self.think if self.reasoning else self.nothink

    @property
    def min_completion_tokens(self) -> int:
        """Completion tokens long-context truncation leaves: the selected mode's, else the default for the mode."""
        if self.mode.min_completion_tokens is not None:
            return self.mode.min_completion_tokens
        return DEFAULT_REASONING_COMPLETION_RESERVE if self.reasoning else DEFAULT_COMPLETION_RESERVE


@dataclass
class InspectConfig:
    """``[inspect]``: inspect settings."""

    num_threads: int | None = None  # inspect-ai max_connections; None = generation.num_threads
    max_tasks: int | None = None
    tiktoken_model: str | None = None
    retry_attempts: int | None = None
    retry_wait: int | None = None


@dataclass
class DomynInspectConfig:
    """Top-level config loaded from model.toml."""

    generation: GenerationConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    judge: ModelConfig = field(default_factory=ModelConfig)
    inspect: InspectConfig = field(default_factory=InspectConfig)


def _safe_init[T: DataclassInstance](cls: type[T], raw: dict) -> T:
    """Instantiate a dataclass from a raw dict, ignoring unknown keys.

    This makes domyn-inspect's config loader forward-compatible with new fields
    added to the main repo's dataclasses — extra keys are silently dropped rather
    than raising TypeError.
    """
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in known})


def load_tasks_config(path: str) -> TasksConfig:
    """Load a tasks TOML."""
    with Path(path).open("rb") as f:
        raw = tomllib.load(f)

    general = TasksGeneralConfig(**raw.get("general", {}))
    tasks = [
        TaskEntry(
            task=entry["task"],
            task_args=dict(entry.get("task_args", {})),
            display_name=entry.get("display_name", ""),
            generate=dict(entry.get("generate", {})),
            message_limit=entry.get("message_limit"),
            pass_k=entry.get("pass_k"),
            dual_score=bool(entry.get("dual_score", False)),
        )
        for entry in raw.get("tasks", [])
    ]

    if not tasks:
        msg = f"No [[tasks]] entries found in {path}"
        raise ValueError(msg)

    inspect = _safe_init(TasksInspectConfig, raw.get("inspect", {}))

    return TasksConfig(general=general, tasks=tasks, inspect=inspect)


def load_inspect_config(path: str, inspect_key: str = "inspect") -> DomynInspectConfig:
    """Load a model TOML.

    ``inspect_key`` names an ADDITIONAL section whose InspectConfig-compatible
    keys (``num_threads``, ``max_tasks``, …) are layered over ``[inspect]``.
    A suite that configures its own concurrency — e.g. ``[inspect_swebench]
    num_threads`` — would otherwise be ignored here, silently falling back to
    the ``[inspect]`` default. Unknown keys in that section are dropped by
    ``_safe_init``, so passing a full suite section is safe. Defaults to
    ``"inspect"``, i.e. unchanged behaviour.
    """
    with Path(path).open("rb") as f:
        raw = tomllib.load(f)

    gen_raw = raw.get("generation", {})
    config = DomynInspectConfig(
        model=_safe_init(ModelConfig, raw.get("model", {})),
        judge=_safe_init(ModelConfig, raw.get("judge", {})),
        generation=_safe_init(
            GenerationConfig,
            {
                **gen_raw,
                "think": _safe_init(GenerationModeConfig, gen_raw.get("think", {})),
                "nothink": _safe_init(GenerationModeConfig, gen_raw.get("nothink", {})),
            },
        ),
        inspect=_safe_init(
            InspectConfig,
            {
                **raw.get("inspect", {}),
                **(raw.get(inspect_key, {}) if inspect_key != "inspect" else {}),
            },
        ),
    )

    if config.inspect.num_threads is None:
        config.inspect.num_threads = config.generation.num_threads

    return config
