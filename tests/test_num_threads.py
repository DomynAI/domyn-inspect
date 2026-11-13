"""``[inspect].num_threads`` (inspect-ai's max_connections) falls back to ``[general].num_threads``."""

from pathlib import Path

import pytest

from domyn_inspect.config import load_inspect_config

DUMMY_TOML = Path(__file__).parent / "fixtures" / "dummy_model.toml"


def _load(tmp_path, general="", inspect="", extra=""):
    text = f"[general]\n{general}\n" + DUMMY_TOML.read_text()
    if inspect:
        text += f"\n[inspect]\n{inspect}"
    path = tmp_path / "model.toml"
    path.write_text(text + extra)
    return load_inspect_config(str(path), inspect_key="inspect_extra" if extra else "inspect")


@pytest.mark.parametrize(
    ("general", "inspect", "expected"),
    [
        ("", "", 64),  # neither set: the general default
        ("num_threads = 96\n", "", 96),
        ("num_threads = 96\n", "num_threads = 12\n", 12),
    ],
)
def test_num_threads_resolution(tmp_path, general, inspect, expected):
    assert _load(tmp_path, general, inspect).inspect.num_threads == expected


def test_inspect_key_section_overrides_num_threads(tmp_path):
    cfg = _load(tmp_path, "num_threads = 96\n", extra="\n[inspect_extra]\nnum_threads = 7\n")
    assert cfg.inspect.num_threads == 7
