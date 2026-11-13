# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""The tasks TOML and model TOML configs, and their loaders."""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

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

    request_timeout: float | None = None  # request + connect timeout (s); overrides general.request_timeout
    # Per-recipe cap on generation max_tokens
    max_tokens: int | None = None


@dataclass
class TasksConfig:
    """Loaded from tasks.toml — task list + runner settings."""

    general: TasksGeneralConfig
    tasks: list[TaskEntry]
    inspect: TasksInspectConfig = field(default_factory=TasksInspectConfig)


# The model TOML: the [general], [model], [judge], [user] and [inspect] sections of domyn-eval's config, with the same
# names and defaults. Only the fields domyn-inspect reads are declared; the loader drops the rest.


@dataclass
class GeneralConfig:
    """``[general]``: the run's request settings."""

    num_threads: int = 64  # concurrent requests; [inspect].num_threads defaults to it
    request_timeout: int = 3600  # per-request read and connect timeout (s)
    max_retries: int = 0  # inspect's retries of a failed request


@dataclass
class SamplingConfig:
    """A mode's (``think``, ``nothink``) or a role's (``user``, ``grader``) sampling params.

    None leaves one to the server, or for a role, to the selected mode.
    """

    temperature: float | None = None
    top_p: float | None = None
    min_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    frequency_penalty: float | None = None
    chat_template_kwargs: dict[str, Any] | None = None


DEFAULT_COMPLETION_RESERVE = 4096
DEFAULT_REASONING_COMPLETION_RESERVE = 8192


@dataclass
class ContextConfig:
    """``[<model>.generation.context]``: the context window and its budget."""

    max_tokens: int = 0  # cap sent with every request; 0 lets the server budget each one
    max_model_len: int = 0  # the server's --max-model-len; required for [model]
    completion_reserve: int | None = None  # left for generation when long-context tasks truncate; None: default
    token_safety_margin: int = 256  # tokens kept free when estimating the available token budget
    tokenizer_path: str | None = None


# Each role's defaults, filled in under what its table sets: deterministic, as tau2 asks of its user simulator.
ROLE_DEFAULTS: dict[str, dict[str, Any]] = {"user": {"temperature": 0.0}, "grader": {"temperature": 0.0}}


@dataclass
class GenerationConfig:
    """``[<model>.generation]``: how a model's requests are built.

    The model under test answers with ``mode``; the model playing a role samples with its table, over its ``mode``.
    """

    reasoning: bool = True  # selects think over nothink as mode
    think: SamplingConfig = field(default_factory=SamplingConfig)
    nothink: SamplingConfig = field(default_factory=SamplingConfig)
    user: SamplingConfig = field(default_factory=lambda: SamplingConfig(**ROLE_DEFAULTS["user"]))
    grader: SamplingConfig = field(default_factory=lambda: SamplingConfig(**ROLE_DEFAULTS["grader"]))
    context: ContextConfig = field(default_factory=ContextConfig)

    @property
    def mode(self) -> SamplingConfig:
        """The params of the selected mode."""
        return self.think if self.reasoning else self.nothink

    def role(self, name: str) -> SamplingConfig:
        """The params of role ``name`` (``user`` or ``grader``): its own, over the selected mode's."""
        own = {key: value for key, value in dataclasses.asdict(getattr(self, name)).items() if value is not None}
        return dataclasses.replace(self.mode, **own)

    @property
    def completion_reserve(self) -> int:
        """Completion tokens long-context truncation leaves: ``context.completion_reserve``, else the mode's default."""
        if self.context.completion_reserve is not None:
            return self.context.completion_reserve
        return DEFAULT_REASONING_COMPLETION_RESERVE if self.reasoning else DEFAULT_COMPLETION_RESERVE


@dataclass
class ModelConfig:
    """``[model]``, ``[judge]`` and ``[user]``: a model, its endpoint and its requests.

    Attributes:
        id: The model's id, which keys its results directory.
        path: The weights, or a HF hub id; the model name sent to the endpoint.
        url: The OpenAI-compatible base URL; ``/v1`` is appended if missing.
        generation: How its requests are built.
    """

    id: str | None = None
    path: str = ""
    url: str | None = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def __post_init__(self) -> None:
        """Append ``/v1`` to the URL if missing; an empty URL becomes None."""
        if self.url:
            url = self.url.rstrip("/")
            self.url = url if url.endswith("/v1") else f"{url}/v1"
        else:
            self.url = None

    @property
    def is_served(self) -> bool:
        """Whether the section names a model and an endpoint serving it."""
        return bool(self.path and self.url and urlparse(self.url).hostname)


# The defaults of [judge] and [user] that differ from a model's, filled in under what they set: a looping judge would
# otherwise write up to its whole context window into the conversation.
ROLE_MODEL_DEFAULTS: dict[str, Any] = {"generation": {"context": {"max_tokens": 4096}}}


@dataclass
class InspectConfig:
    """``[inspect]``: inspect settings."""

    num_threads: int | None = None  # inspect-ai max_connections; None = general.num_threads
    max_tasks: int | None = None
    tiktoken_model: str | None = None
    retry_attempts: int | None = None
    retry_wait: int | None = None


@dataclass
class DomynInspectConfig:
    """Top-level config loaded from model.toml."""

    model: ModelConfig
    judge: ModelConfig = field(default_factory=lambda: _load_model({}, ROLE_MODEL_DEFAULTS))
    user: ModelConfig = field(default_factory=lambda: _load_model({}, ROLE_MODEL_DEFAULTS))
    general: GeneralConfig = field(default_factory=GeneralConfig)
    inspect: InspectConfig = field(default_factory=InspectConfig)

    def __post_init__(self) -> None:
        """Reject a missing ``model.generation.context.max_model_len``."""
        max_model_len = self.model.generation.context.max_model_len
        if not max_model_len or max_model_len <= 0:
            msg = (
                f"model.generation.context.max_model_len={max_model_len}: set it to the server's --max-model-len, "
                "which every request and the long-context truncation are budgeted against"
            )
            raise ValueError(msg)


def _safe_init[T: DataclassInstance](cls: type[T], raw: dict) -> T:
    """Instantiate a dataclass from a raw dict, ignoring unknown keys.

    This makes domyn-inspect's config loader forward-compatible with new fields
    added to the main repo's dataclasses — extra keys are silently dropped rather
    than raising TypeError.
    """
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in known})


def _with_defaults(raw: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """``raw`` with ``defaults`` filled in where it sets nothing, table by table."""
    merged = {**defaults, **raw}
    for key, default in defaults.items():
        if isinstance(default, dict) and isinstance(raw.get(key), dict):
            merged[key] = _with_defaults(raw[key], default)
    return merged


def _load_model(raw: dict, defaults: dict[str, Any] | None = None) -> ModelConfig:
    """A ``[model]``-like section with its generation tables, over ``defaults``."""
    raw = _with_defaults(raw, defaults or {})
    gen = raw.get("generation", {})
    tables = {"think": {}, "nothink": {}, **ROLE_DEFAULTS}  # each sampling table's defaults
    sampling = {name: _safe_init(SamplingConfig, {**default, **gen.get(name, {})}) for name, default in tables.items()}
    context = _safe_init(ContextConfig, gen.get("context", {}))
    generation = _safe_init(GenerationConfig, {**gen, **sampling, "context": context})
    return _safe_init(ModelConfig, {**raw, "generation": generation})


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

    config = DomynInspectConfig(
        model=_load_model(raw.get("model", {})),
        judge=_load_model(raw.get("judge", {}), ROLE_MODEL_DEFAULTS),
        user=_load_model(raw.get("user", {}), ROLE_MODEL_DEFAULTS),
        general=_safe_init(GeneralConfig, raw.get("general", {})),
        inspect=_safe_init(
            InspectConfig,
            {
                **raw.get("inspect", {}),
                **(raw.get(inspect_key, {}) if inspect_key != "inspect" else {}),
            },
        ),
    )

    if config.inspect.num_threads is None:
        config.inspect.num_threads = config.general.num_threads

    return config
