"""``[model]`` and ``[judge]``: the endpoint URL is normalised as domyn-eval does."""

import pytest

from domyn_inspect.config import ModelConfig


@pytest.mark.parametrize("url", ["http://h:9000", "http://h:9000/", "http://h:9000/v1", "http://h:9000/v1/"])
def test_url_ends_in_v1(url):
    assert ModelConfig(url=url).url == "http://h:9000/v1"


@pytest.mark.parametrize("url", [None, ""])
def test_no_url_stays_unset(url):
    assert ModelConfig(url=url).url is None
