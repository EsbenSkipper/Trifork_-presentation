"""Shared fixtures for the project_2 test suite.

The tests run **fully offline** — no API key is needed. Both bots import cleanly
without a key (the LLM client is only constructed when actually used), and their
*deterministic* core (pandas filtering, aggregations, dispute rules, isolation)
is what we assert here. That is the part that must always be correct, whichever
way the LLM routes.
"""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def old():
    """The router bot (Old_LLM_BOT/main.py)."""
    return _load("oldbot_under_test", "Old_LLM_BOT/main.py")


@pytest.fixture(scope="session")
def tools():
    """The tool-calling bot (LLM_BOT/main_tools.py)."""
    return _load("toolsbot_under_test", "LLM_BOT/main_tools.py")