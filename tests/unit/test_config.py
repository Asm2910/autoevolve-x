"""Both dataset configs must carry every key the scripts read, so a missing key fails here, not in a live run."""

from pathlib import Path

import pytest
import yaml

from agentkit.budget import Budget
from agentkit.tracing import Observer
from scripts.run_agent import DEFAULT_LLM_MODEL, llm_model_name, make_proposer

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
NAMES = ["models.yaml", "ieee.yaml"]


def load(name):
    return yaml.safe_load((CONFIGS / name).read_text())


@pytest.mark.parametrize("name", NAMES)
def test_agent_config_has_the_keys_the_scripts_read(name):
    agent = load(name)["agent"]
    assert {"max_iterations", "min_gain", "llm_model"} <= set(agent)
    assert agent["llm_model"].startswith("claude-")


def test_model_name_prefers_the_environment_then_config_then_default(monkeypatch):
    monkeypatch.delenv("AUTOEVOLVE_LLM_MODEL", raising=False)
    assert llm_model_name({"agent": {"llm_model": "claude-x"}}) == "claude-x"
    assert llm_model_name({"agent": {}}) == DEFAULT_LLM_MODEL
    assert llm_model_name({}) == DEFAULT_LLM_MODEL
    monkeypatch.setenv("AUTOEVOLVE_LLM_MODEL", "claude-env")
    assert llm_model_name({"agent": {"llm_model": "claude-x"}}) == "claude-env"


@pytest.mark.parametrize("name", NAMES)
def test_llm_proposer_builds_from_each_config(name, monkeypatch):
    pytest.importorskip("langchain_anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")  # construction only, no network call
    proposer = make_proposer("llm", load(name), Budget.unlimited(), Observer())
    assert proposer is not None
