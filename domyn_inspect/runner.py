# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Eval execution and dry-run display."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import click

import domyn_inspect.provider  # noqa: F401
from domyn_inspect.config import DomynInspectConfig, GenerationConfig, TasksConfig  # type: ignore


def _build_model_roles(
    model_cfg: DomynInspectConfig, user_sim_temperature: float, *, client_timeout: float
) -> dict[str, Any]:
    """Register the inspect model roles served by the ``[judge]`` endpoint.

    - ``role="user"`` — tau2's *user simulator*.
    - ``role="grader"`` — HLE's LLM-as-judge scorer.

    Endpoint selection:
      - a dedicated ``[judge]`` endpoint if one is configured, else
      - **self-play**: the model-under-test's own endpoint (``user`` only).

    """
    from urllib.parse import urlparse

    from inspect_ai.model import GenerateConfig, Model, get_model

    judge = model_cfg.judge
    judge_ok = bool(judge and judge.path and judge.url and urlparse(judge.url).hostname)
    if judge_ok:
        path, url = judge.path, judge.url
    else:  # self-play
        path, url = model_cfg.model.path, model_cfg.model.url

    api_key = os.getenv("API_KEY", "") or os.getenv("_API_KEY", "") or "EMPTY"

    def _role_model(temperature: float) -> Model:
        return get_model(
            f"domyn/{path}",
            base_url=url,
            api_key=api_key,
            config=GenerateConfig(temperature=temperature),
            max_retries=0,
            client_timeout=client_timeout,
        )

    roles: dict[str, Any] = {"user": _role_model(user_sim_temperature)}

    if judge_ok:
        roles["grader"] = _role_model(0.0)

    return roles


def effective_max_tokens(tasks_cfg: TasksConfig, model_cfg: DomynInspectConfig) -> int | None:
    """The max_tokens of every task whose entry sets none, or None to let the server budget each request.

    A tasks TOML's [inspect] max_tokens overrides the model's [generation] max_tokens when it
    is > 0; otherwise the model's applies. A cap that fills the whole context window leaves no
    room for any prompt, so it is dropped and vLLM budgets each request as
    max_model_len - prompt_tokens.
    """
    gen = model_cfg.generation
    tasks_max_tokens = tasks_cfg.inspect.max_tokens
    max_tokens = tasks_max_tokens if tasks_max_tokens and tasks_max_tokens > 0 else gen.max_tokens
    if not max_tokens or max_tokens <= 0:
        return None
    window = gen.max_model_len
    if max_tokens >= window:
        logging.getLogger(__name__).info(
            "max_tokens=%s fills the %s-token context window; omitting it so the server budgets each request",
            max_tokens,
            window,
        )
        return None
    return max_tokens


def sampling_extra_body(gen: GenerationConfig) -> dict[str, Any]:
    """The vLLM sampling params with no OpenAI API field, sent in the request body as-is.

    top_k is one: inspect's OpenAI-compatible provider drops ``GenerateConfig.top_k``, so without it here vLLM
    samples with the model's default (no top-k when it ships no generation_config.json).
    """
    mode = gen.mode
    extra_body: dict[str, Any] = {}
    if mode.top_k is not None:
        extra_body["top_k"] = mode.top_k
    if mode.min_p is not None:
        extra_body["min_p"] = mode.min_p
    if mode.repetition_penalty is not None:
        extra_body["repetition_penalty"] = mode.repetition_penalty

    if mode.chat_template_kwargs:
        extra_body["chat_template_kwargs"] = mode.chat_template_kwargs
    return extra_body


def token_budget(tasks_cfg: TasksConfig, model_cfg: DomynInspectConfig) -> dict[str, Any]:
    """The budget the long-context tasks truncate against and the provider clamps each request to.

    Built from the model config alone, with the cap inspect actually sends (``effective_max_tokens``), so the
    completion room a truncated prompt leaves is the room the request asks for.
    """
    gen = model_cfg.generation
    return {
        "max_model_len": gen.max_model_len,
        "max_tokens": effective_max_tokens(tasks_cfg, model_cfg),
        "token_safety_margin": gen.token_safety_margin,
        "min_completion_tokens": gen.min_completion_tokens,
        "tokenizer_path": (gen.tokenizer_path or model_cfg.model.path).rstrip("/"),
        "chat_template_kwargs": gen.mode.chat_template_kwargs or {},
    }


def run_eval(
    tasks_cfg: TasksConfig,
    model_cfg: DomynInspectConfig,
    *,
    log_dir: str,
    limit: int | None = None,
) -> None:
    """Resolve tasks and run them via ``inspect_ai.eval_set()``."""
    from inspect_ai import eval_set
    from inspect_ai._util.http_defaults_httpx2 import DEFAULT_POOL_CONNECTIONS

    from domyn_inspect.resolver import resolve_tasks
    from domyn_inspect.tasks import _long_context

    # Some tasks call get_model() during instantiation (e.g. niah).
    # Set the env var so the model is available at resolve time.
    os.environ["INSPECT_EVAL_MODEL"] = f"domyn/{model_cfg.model.path}"

    # Before any model is built: inspect sizes each client's pool from this env var
    os.environ.setdefault(
        "INSPECT_HTTP_POOL_CONNECTIONS",
        str(max(model_cfg.inspect.num_threads or DEFAULT_POOL_CONNECTIONS, DEFAULT_POOL_CONNECTIONS)),
    )

    if model_cfg.inspect.tiktoken_model:
        os.environ["INSPECT_TIKTOKEN_MODEL"] = model_cfg.inspect.tiktoken_model

    # Ensure tiktoken uses a persistent shared cache
    os.environ.setdefault(
        "TIKTOKEN_CACHE_DIR",
        str(Path(__file__).resolve().parents[1] / ".cache" / "tiktoken"),
    )

    # Before resolving: the long-context tasks budget their prompts while they are built
    budget = token_budget(tasks_cfg, model_cfg)
    _long_context.set_budget(budget)

    click.echo("Resolving tasks …")
    # Scorers come back final: errored-sample handling, group breakdown and dual scoring are applied by the resolver.
    resolved = resolve_tasks(tasks_cfg)

    # Build generation kwargs (only those explicitly set).
    gen_kwargs: dict[str, Any] = {}
    gen = model_cfg.generation
    gen_kwargs["temperature"] = gen.mode.temperature
    gen_kwargs["top_p"] = gen.mode.top_p
    # Set per task, not passed to eval_set: eval_set's config is merged over each task's own,
    # so it would override an entry's `generate = { max_tokens = N }`.
    max_tokens = effective_max_tokens(tasks_cfg, model_cfg)
    for entry, task in zip(tasks_cfg.tasks, resolved, strict=True):
        if max_tokens is not None and "max_tokens" not in entry.generate:
            task.config = task.config.model_copy(update={"max_tokens": max_tokens})
    gen_kwargs["presence_penalty"] = gen.mode.presence_penalty
    gen_kwargs["frequency_penalty"] = gen.mode.frequency_penalty
    gen_kwargs["extra_body"] = sampling_extra_body(gen)

    http_timeout_s = float(tasks_cfg.inspect.request_timeout or gen.request_timeout)
    gen_kwargs["timeout"] = int(http_timeout_s)
    gen_kwargs["max_retries"] = gen.max_retries

    # Register judge-driven roles: tau2 user-simulator (role="user") and
    # HLE grader (role="grader"); see _build_model_roles.
    user_sim_temperature = tasks_cfg.inspect.user_sim_temperature
    model_roles = _build_model_roles(model_cfg, user_sim_temperature, client_timeout=http_timeout_s)
    _j = model_cfg.judge
    _user_src = f"[judge] {_j.url}" if "grader" in model_roles else f"self-play {model_cfg.model.url}"
    click.echo(f"User-simulator role -> {_user_src}  (temperature={user_sim_temperature})")
    _grader_src = (
        f"[judge] {_j.url}  (temperature=0.0)" if "grader" in model_roles else "model-under-test (inspect default)"
    )
    click.echo(f"Grader role         -> {_grader_src}")

    click.echo(f"Running eval_set  ({len(resolved)} tasks, max_tasks={tasks_cfg.general.max_tasks}) …\n")

    click.echo(
        f"Model name: {model_cfg.model.id}  "
        f"Model path: domyn/{model_cfg.model.path}  "
        f"Endpoint: {model_cfg.model.url}  "
        f"Max connections: {model_cfg.inspect.num_threads}  "
        f"Retry attempts: {tasks_cfg.general.retry_attempts}  "
        f"Retry wait: {tasks_cfg.general.retry_wait}s  "
        f"Log shared: {tasks_cfg.general.log_shared}  "
        f"Log dir: {log_dir}"
    )

    click.echo(
        f"HTTP: client_timeout={http_timeout_s}s  "
        f"model_retries={gen_kwargs['max_retries']} within {gen_kwargs['timeout']}s"
    )

    success, _ = eval_set(
        tasks=resolved,
        log_dir=log_dir,
        model=f"domyn/{model_cfg.model.path}",
        model_base_url=model_cfg.model.url,
        model_roles=model_roles,
        model_args={"max_retries": 0, "client_timeout": http_timeout_s, "budget": budget},
        max_tasks=tasks_cfg.general.max_tasks,
        max_connections=model_cfg.inspect.num_threads,
        retry_attempts=tasks_cfg.general.retry_attempts,
        retry_wait=tasks_cfg.general.retry_wait,
        log_shared=tasks_cfg.general.log_shared,
        fail_on_error=tasks_cfg.general.fail_on_error,
        score_on_error=True,
        log_dir_allow_dirty=True,
        limit=limit,
        display="log",
        log_level=os.environ.get("INSPECT_LOG_LEVEL", "info"),  # the log display's progress lines are INFO
        **gen_kwargs,
    )

    # Scores live in the .eval logs: the domyn-eval report driver reads them with inspect's log API.
    if not success:
        click.echo("Some tasks failed.  Check logs for details.", err=True)
        sys.exit(1)


def print_dry_run(
    tasks_cfg: TasksConfig,
    model_cfg: DomynInspectConfig,
    *,
    log_dir: str,
    limit: int | None = None,
) -> None:
    """Show what *would* be executed without actually running."""
    click.echo("\n[DRY-RUN]  The following tasks will be passed to eval_set():\n")
    for i, entry in enumerate(tasks_cfg.tasks, 1):
        label = entry.display_name or entry.task
        args_str = ", ".join(f"{k}={v}" for k, v in entry.task_args.items())
        gen_str = ", ".join(f"{k}={v}" for k, v in entry.generate.items())
        click.echo(f"  {i:>3}. {label}")
        if args_str:
            click.echo(f"       task_args: {args_str}")
        if gen_str:
            click.echo(f"       generate:  {gen_str}")

    click.echo(f"\n  Model:           {model_cfg.model.id}")
    if model_cfg.judge.path:
        click.echo(f"  Judge (user):    {model_cfg.judge.path}  ({model_cfg.judge.url})")
    click.echo(f"  Log dir:         {log_dir}")
    click.echo(f"  Max tasks:       {tasks_cfg.general.max_tasks}")
    click.echo(f"  Max connections: {model_cfg.inspect.num_threads}")
    click.echo(f"  Retry attempts:  {tasks_cfg.general.retry_attempts}")
    click.echo(f"  Retry wait:      {tasks_cfg.general.retry_wait}s")
    click.echo(f"  Log shared:      {tasks_cfg.general.log_shared}")
    if limit is not None:
        click.echo(f"  Limit:           {limit}")

    # Generation
    gen = model_cfg.generation
    mode = gen.mode
    _window = gen.max_model_len
    _eff_max_tokens = effective_max_tokens(tasks_cfg, model_cfg)
    _mt = _eff_max_tokens if _eff_max_tokens is not None else f"auto (window {_window})"
    if (tasks_cfg.inspect.max_tokens or 0) > 0:
        _mt = f"{_mt} (tasks override; model {gen.max_tokens})"
    click.echo(
        f"\n  temperature={mode.temperature}  top_p={mode.top_p}  "
        f"top_k={mode.top_k}  min_p={mode.min_p}  "
        f"max_tokens={_mt}  "
        f"presence_penalty={mode.presence_penalty}  "
        f"repetition_penalty={mode.repetition_penalty}  "
        f"frequency_penalty={mode.frequency_penalty}"
    )

    click.echo(f"\nTotal: {len(tasks_cfg.tasks)} task(s)")
    click.echo("\nPass --run to execute.\n")
