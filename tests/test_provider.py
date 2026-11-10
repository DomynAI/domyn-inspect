"""The domyn/ provider: context-length 400s.

Regression: a run with generation.max_tokens == --max-model-len 400s on every
request; the old handler fed the error text to the scorer as the model's answer,
so IFEval "scored" 0.07 and the run reported success.
"""

import asyncio

import pytest
from inspect_ai.model import ChatMessageUser, GenerateConfig, ModelOutput, get_model
from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI

import domyn_inspect.provider  # registers domyn/
import domyn_inspect.token_budget
from domyn_inspect.provider import (
    ContextBudgetError,
    DomynOpenAIAPI,
    completion_budget_overflows,
)

# Verbatim from results/qwen3_4b/run_2026-08-25_15-15-36 (max_tokens == window).
BUDGET_MSG = (
    "This model's maximum context length is 32768 tokens. However, you requested "
    "32814 tokens (46 in the messages, 32768 in the completion). Please reduce the "
    "length of the messages or completion. None"
)
# A genuinely over-long prompt with a sane completion budget.
PROMPT_MSG = (
    "This model's maximum context length is 32768 tokens. However, you requested "
    "39000 tokens (38000 in the messages, 1000 in the completion). Please reduce "
    "the length of the messages or completion. None"
)


class _FakeBadRequestError(Exception):
    """Stands in for openai.BadRequestError: same attributes the handlers read.

    ``code``/``type`` are left unset, as vLLM sends them, so upstream's
    OpenAI-native checks don't match and our handling is what decides.
    """

    def __init__(self, msg):
        super().__init__(msg)
        self.body = {"message": msg}
        self.message = msg
        self.status_code = 400
        self.code = None
        self.type = None


@pytest.fixture(scope="module")
def api() -> DomynOpenAIAPI:
    return DomynOpenAIAPI("domyn/domynedge-sft", base_url="http://localhost:1/v1")


def _handle(api, msg):
    return api.handle_bad_request(_FakeBadRequestError(msg))


# ── registration ──────────────────────────────────────────────────────────────


def test_provider_is_registered():
    model = get_model("domyn/Qwen/Qwen3-4B", base_url="http://localhost:1/v1")
    assert isinstance(model.api, DomynOpenAIAPI)


def test_model_id_is_sent_whole(api):
    """The base class strips a leading "<service>/", i.e. "domyn/" here."""
    assert api.service_model_name() == "domyn/domynedge-sft"


# ── classification ────────────────────────────────────────────────────────────


def test_budget_overflow_detected():
    assert completion_budget_overflows(BUDGET_MSG) is True


def test_long_prompt_is_not_a_budget_overflow():
    assert completion_budget_overflows(PROMPT_MSG) is False


def test_long_prompt_with_large_max_tokens_still_degrades():
    """A 30k prompt leaving no room for an 8k budget is the prompt's fault, not
    the config's — aborting the task here would break long-context suites."""
    msg = (
        "This model's maximum context length is 32768 tokens. However, you requested "
        "38192 tokens (30000 in the messages, 8192 in the completion)."
    )
    assert completion_budget_overflows(msg) is False


def test_unparseable_message_is_undecidable():
    assert completion_budget_overflows("prompt is too long") is None


# ── handler behaviour ─────────────────────────────────────────────────────────


def test_budget_overflow_returns_an_error(api):
    result = _handle(api, BUDGET_MSG)
    assert isinstance(result, ContextBudgetError)
    assert "max_tokens" in str(result)


def test_long_prompt_degrades_to_empty_output(api):
    result = _handle(api, PROMPT_MSG)
    assert isinstance(result, ModelOutput)
    assert result.choices[0].stop_reason == "model_length"
    # The regression: error text as completion earns accidental scorer credit.
    assert result.completion == ""
    assert result.error == PROMPT_MSG


def test_unparseable_context_error_degrades_to_empty_output(api):
    result = _handle(api, "This model's maximum context length is exceeded.")
    assert isinstance(result, ModelOutput)
    assert result.completion == ""


def test_non_context_bad_request_is_left_alone(api):
    result = _handle(api, "you must provide a model parameter")
    assert not isinstance(result, ModelOutput)


# ── vLLM 0.23 messages ─────────────────────────────────────────────────────────

# Verbatim from vLLM 0.23: the prompt is counted only up to the first token over the allowance
AT_LEAST_MSG = (
    "This model's maximum context length is 131072 tokens. However, you requested 32768 output tokens and your "
    "prompt contains at least 98305 input tokens, for a total of at least 131073 tokens. Please reduce the length "
    "of the input prompt or the number of requested output tokens. (parameter=input_tokens, value=98305)"
)
# Verbatim from vLLM 0.23's character pre-check, with max_tokens = the window
CHARS_ZERO_MSG = (
    "This model's maximum context length is 131072 tokens. However, you requested 131072 output tokens and your "
    "prompt contains 241 characters (more than 0 characters, which is the upper bound for 0 input tokens). "
    "Please reduce the length of the input prompt or the number of requested output tokens."
)


def test_the_new_formats_tell_a_full_window_budget_from_a_long_prompt():
    assert completion_budget_overflows(CHARS_ZERO_MSG) is True
    assert completion_budget_overflows(AT_LEAST_MSG) is False


# ── max_tokens cut to what the prompt leaves ──────────────────────────────────


WINDOW = 32768
MARGIN = 256
BUDGET = {
    "max_model_len": WINDOW,
    "token_safety_margin": MARGIN,
    "tokenizer_path": "/models/domynedge-sft",
    "chat_template_kwargs": {"enable_thinking": False},
}


class FakeTokenizer:
    """Renders every prompt as ``prompt_tokens`` tokens, recording what it was asked to render."""

    def __init__(self, prompt_tokens):
        self.prompt_tokens = prompt_tokens
        self.calls = []

    def apply_chat_template(self, conversation, **kwargs):
        self.calls.append({"conversation": conversation, **kwargs})
        return [0] * self.prompt_tokens


NO_TOKENIZER = None


def _send(monkeypatch, max_tokens, prompt_tokens=100, budget=BUDGET, content="hello", tokenizer=...):
    """The max_tokens the provider sends for a ``prompt_tokens``-token prompt, and the tokenizer that counted it."""
    tokenizer = FakeTokenizer(prompt_tokens) if tokenizer is ... else tokenizer
    loaded = []

    def load(path):
        loaded.append(path)
        return tokenizer

    # The provider loads it up front; count_prompt (core/token_budget.py) counts with it
    monkeypatch.setattr(domyn_inspect.provider, "load_tokenizer", load)
    monkeypatch.setattr(domyn_inspect.token_budget, "load_tokenizer", lambda path: tokenizer)  # noqa: ARG005
    api = DomynOpenAIAPI("domyn/domynedge-sft", base_url="http://localhost:1/v1", budget=budget)
    if budget:
        assert loaded == [budget["tokenizer_path"]]

    async def generate(self, input, tools, tool_choice, config):  # noqa: A002, ARG001
        return ModelOutput.from_content(model="m", content=str(config.max_tokens)), None

    monkeypatch.setattr(OpenAICompatibleAPI, "generate", generate)
    config = GenerateConfig(max_tokens=max_tokens)
    output, _ = asyncio.run(api.generate([ChatMessageUser(content=content)], [], "none", config))
    return output.completion, tokenizer


def test_a_cap_the_prompt_leaves_room_for_is_sent_as_is(monkeypatch):
    assert _send(monkeypatch, 8192, prompt_tokens=1000)[0] == "8192"


def test_a_cap_is_cut_to_what_the_prompt_leaves(monkeypatch):
    sent, tokenizer = _send(monkeypatch, 8192, prompt_tokens=WINDOW - MARGIN - 2768)
    assert sent == "2768"
    [call] = tokenizer.calls
    assert call["conversation"] == [{"role": "user", "content": "hello"}]  # counted as sent
    assert call["tools"] is None
    assert call["add_generation_prompt"]
    assert call["tokenize"]
    assert call["return_dict"] is False
    assert call["enable_thinking"] is False  # rendered with the run's template flags


def test_a_prompt_that_fills_the_window_asks_for_one_token(monkeypatch):
    assert _send(monkeypatch, 8192, prompt_tokens=WINDOW)[0] == "1"


def test_without_a_cap_the_server_budgets_the_request(monkeypatch):
    sent, tokenizer = _send(monkeypatch, None, prompt_tokens=WINDOW)
    assert sent == "None"
    assert tokenizer.calls == []


def test_a_model_without_a_budget_sends_its_cap_as_is(monkeypatch):
    # The judge roles: built without the run's budget, which is the tested model's window
    sent, tokenizer = _send(monkeypatch, 8192, prompt_tokens=WINDOW, budget=None)
    assert sent == "8192"
    assert tokenizer.calls == []


# 40k characters: a 10k-token estimate, which keeps 5k free on top
LONG = "x" * 40_000


def test_without_a_tokenizer_the_prompt_is_estimated(monkeypatch):
    sent, _ = _send(monkeypatch, 30000, content=LONG, tokenizer=NO_TOKENIZER)
    assert sent == str(WINDOW - 10_000 - 5_000)


def test_a_prompt_the_template_cannot_render_is_estimated(monkeypatch, caplog):
    class BrokenTemplate:
        def apply_chat_template(self, conversation, **_kwargs):  # noqa: ARG002
            msg = "bad template"
            raise ValueError(msg)

    sent, _ = _send(monkeypatch, 30000, content=LONG, tokenizer=BrokenTemplate())
    assert sent == str(WINDOW - 10_000 - 5_000)
    assert "could not render the prompt" in caplog.text


def test_a_short_estimate_keeps_the_configured_margin(monkeypatch):
    # "hello" is a 1-token estimate; half of it is below the 256-token margin
    sent, _ = _send(monkeypatch, 40000, tokenizer=NO_TOKENIZER)
    assert sent == str(WINDOW - 1 - MARGIN)
