"""Sampling params with no OpenAI API field reach vLLM through extra_body.

Regression: generation.top_k went only into GenerateConfig.top_k, which inspect's OpenAI-compatible provider drops,
so every domyn/ request sampled without top-k (Qwen3.5-4B ships no generation_config.json to supply one).
"""

from inspect_ai.model import GenerateConfig

from domyn_inspect.config import GenerationConfig, GenerationModeConfig
from domyn_inspect.provider import DomynOpenAIAPI
from domyn_inspect.runner import sampling_extra_body


def gen(**overrides):
    fields = {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking": True}}
    return GenerationConfig(think=GenerationModeConfig(**{**fields, **overrides}), max_model_len=65536)


def test_extra_body_carries_top_k():
    assert sampling_extra_body(gen()) == {
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_unset_params_are_left_to_the_server():
    assert sampling_extra_body(gen(top_k=None, min_p=None, repetition_penalty=None, chat_template_kwargs=None)) == {}


def test_request_sends_top_k():
    api = DomynOpenAIAPI("domyn/m", base_url="http://localhost:1/v1")
    config = GenerateConfig(temperature=1.0, top_k=20, extra_body=sampling_extra_body(gen()))
    params = api.completion_params(config, tools=False)
    assert "top_k" not in params  # what the provider does with GenerateConfig.top_k alone
    assert params["extra_body"]["top_k"] == 20
