"""domyn_inspect/tasks/_long_context.py: truncation of overflowing long-context prompts."""

import pytest

import domyn_inspect.token_budget
from domyn_inspect.tasks import _long_context
from domyn_inspect.tasks._long_context import DocumentTruncator, cut_middle, prompt_token_limit

W = 32768
# What core/runner.py:token_budget builds with a mode's min_completion_tokens set and no explicit cap
BUDGET = {
    "max_model_len": W,
    "max_tokens": None,
    "token_safety_margin": 256,
    "min_completion_tokens": 4096,
    "tokenizer_path": "stub",
    "chat_template_kwargs": {},
}


class CharTokenizer:
    """One token per character; its chat template adds ``template_tokens`` around the messages."""

    template_tokens = 0

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002, FBT002
        return [ord(c) for c in text]

    def apply_chat_template(self, conversation, **kwargs):
        assert kwargs["add_generation_prompt"]
        assert kwargs["return_dict"] is False
        return [0] * self.template_tokens + [t for m in conversation for t in self.encode(m["content"])]

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002, FBT002
        return "".join(chr(i) for i in ids)


class TemplateTokenizer(CharTokenizer):
    template_tokens = 100


class SeamTokenizer(CharTokenizer):
    """Decoding adds a character, as a real tokenizer can at a cut: the first fit comes out too long."""

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002, FBT002
        return super().decode(ids) + "~"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (BUDGET, W - 256 - 4096),  # no explicit cap
        ({**BUDGET, "max_tokens": 0}, W - 256 - 4096),
        ({**BUDGET, "max_tokens": W}, W - 256 - 4096),  # max_tokens = window: implicit
        ({**BUDGET, "max_tokens": 20000}, W - 256 - 20000),  # explicit: all of it
        ({**BUDGET, "max_tokens": 1000}, W - 256 - 1000),
        ({**BUDGET, "min_completion_tokens": None}, W - 256),  # unset: no reserve
        ({**BUDGET, "token_safety_margin": 0, "min_completion_tokens": 8192}, W - 8192),
    ],
)
def test_prompt_token_limit(args, expected):
    assert prompt_token_limit(args) == expected


def test_a_reserve_that_fills_the_window_fails():
    with pytest.raises(ValueError, match="no room for a prompt"):
        prompt_token_limit({**BUDGET, "max_model_len": 4096})


def test_cut_middle_keeps_head_and_tail_halves():
    assert cut_middle(list(range(10)), 5) == [0, 1, 7, 8, 9]
    assert cut_middle(list(range(10)), 4) == [0, 1, 8, 9]
    assert cut_middle(list(range(3)), 5) == [0, 1, 2]


def use_tokenizer(monkeypatch, tokenizer):
    # _long_context cuts with it; count_prompt (core/token_budget.py) counts with it
    for module in (_long_context, domyn_inspect.token_budget):
        monkeypatch.setattr(module, "load_tokenizer", lambda path: tokenizer)  # noqa: ARG005


def truncator(monkeypatch, limit, tokenizer=...):
    """A truncator whose prompt limit is ``limit`` (the reserve is min_completion_tokens, 4096)."""
    use_tokenizer(monkeypatch, CharTokenizer() if tokenizer is ... else tokenizer)
    return DocumentTruncator({**BUDGET, "max_model_len": limit + 256 + 4096})


def render(docs):
    body = "\n".join(f"<doc {i}>{doc}</doc {i}>" for i, doc in enumerate(docs))
    return f"Read these.\n{body}\nQuestion: which?"


def test_a_prompt_that_fits_is_left_alone(monkeypatch):
    docs = ["a" * 100, "b" * 200]
    assert truncator(monkeypatch, 1000)(render, docs) == (
        render(docs),
        {"prompt_tokens": len(render(docs)), "truncated": False},
    )


def test_one_document_loses_its_middle_and_the_question_survives(monkeypatch):
    doc = "a" * 5000 + "b" * 5000
    overhead = len(render([""]))
    prompt, meta = truncator(monkeypatch, 1000 + overhead)(render, [doc])
    assert prompt == render(["a" * 500 + "b" * 500])
    assert meta == {"prompt_tokens": 10000 + overhead, "truncated": True}


def test_documents_are_cut_in_proportion_and_keep_their_markers(monkeypatch):
    docs = ["x" * 1000, "y" * 3000, "z" * 6000]
    overhead = len(render(["", "", ""]))
    prompt, meta = truncator(monkeypatch, 5000 + overhead)(render, docs)
    assert meta["truncated"]
    assert len(prompt) <= 5000 + overhead
    assert [prompt.count(ch) for ch in "xyz"] == [500, 1500, 3000]
    assert all(f"<doc {i}>" in prompt and f"</doc {i}>" in prompt for i in range(3))
    assert prompt.startswith("Read these.\n")
    assert prompt.endswith("\nQuestion: which?")


def test_the_chat_template_is_counted(monkeypatch):
    doc = "a" * 5000 + "b" * 5000
    limit = 1000 + len(render([""])) + TemplateTokenizer.template_tokens
    prompt, meta = truncator(monkeypatch, limit, TemplateTokenizer())(render, [doc])
    assert prompt == render(["a" * 500 + "b" * 500])  # the template's 100 tokens come out of the documents
    assert meta["prompt_tokens"] == 10000 + len(render([""])) + 100


def test_without_a_tokenizer_documents_are_cut_by_characters(monkeypatch):
    # A 25000-token estimate keeps 12500 free instead of 256, so the first cut aims at 20000 - 12244 = 7756
    # tokens, i.e. 31024 characters; that prompt's estimate fits the margin it keeps
    prompt, meta = truncator(monkeypatch, 20000, tokenizer=None)(lambda docs: docs[0], ["a" * 50000 + "b" * 50000])
    assert prompt == "a" * 15512 + "b" * 15512
    assert meta == {"prompt_tokens": 25000, "truncated": True}


def test_the_fit_absorbs_tokens_added_at_the_cuts(monkeypatch):
    docs = ["a" * 1000, "b" * 3000]
    limit = 2000 + len(render(["", ""]))
    prompt, meta = truncator(monkeypatch, limit, SeamTokenizer())(render, docs)
    assert meta["truncated"]
    assert len(prompt) <= limit


def test_truncator_is_a_no_op_without_a_budget(monkeypatch):
    monkeypatch.setattr(_long_context, "_budget", None)
    assert DocumentTruncator()(render, ["x" * 100_000]) == (
        render(["x" * 100_000]),
        {"prompt_tokens": None, "truncated": False},
    )


def test_truncator_uses_the_budget_the_runner_set(monkeypatch):
    budget = {**BUDGET, "max_model_len": 8192 + 256 + 4096}
    monkeypatch.setattr(_long_context, "_budget", None)
    _long_context.set_budget(budget)
    use_tokenizer(monkeypatch, CharTokenizer())
    prompt, meta = DocumentTruncator()(lambda docs: docs[0], ["a" * 5000 + "b" * 5000])
    assert prompt == "a" * 4096 + "b" * 4096
    assert meta == {"prompt_tokens": 10000, "truncated": True}
