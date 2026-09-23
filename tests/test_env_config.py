"""env_config() fallback-chain tests.

Scenario being covered: the user runs `mforege` from a folder that has its
own .env (e.g. a Django project). That .env never contains MForege settings,
so `env_config()` must fall through to `~/.mforege/.env` for any key the
local file doesn't define. Previously a bare `os.path.exists(".env")` check
made the local file shadow the global config completely.
"""
import os

import pytest

import main


@pytest.fixture()
def layered_env(tmp_path, monkeypatch):
    """CWD with a project .env + a global ~/.mforege/.env + no repo .env."""
    local = tmp_path / "project"
    local.mkdir()
    (local / ".env").write_text("DEBUG=True\n", encoding="utf-8")

    monkeypatch.chdir(local)
    monkeypatch.setattr(main, "_root_config", None)          # repo .env hidden
    (tmp_path / "global.env").write_text(
        "LLM_BACKEND=custom\nAPI_KEY=gsk_test\nBASE_URL=https://api.groq.com/openai/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        main, "_home_config", main.Config(main.RepositoryEnv(str(tmp_path / "global.env")))
    )
    return local


def test_local_key_wins_over_global(layered_env):
    (layered_env / ".env").write_text("DEBUG=True\nLLM_BACKEND=openai\n", encoding="utf-8")
    assert main.env_config("LLM_BACKEND") == "openai"


def test_missing_local_key_falls_through_to_global(layered_env):
    # The exact failure from the bug report: local .env has no LLM_* keys.
    assert main.env_config("LLM_BACKEND") == "custom"
    assert main.env_config("API_KEY") == "gsk_test"
    assert main.env_config("BASE_URL") == "https://api.groq.com/openai/v1"


def test_missing_everywhere_returns_default(layered_env):
    assert main.env_config("OPENAI_API_KEY", default="") == ""


def test_no_local_env_uses_global(layered_env, tmp_path):
    os.remove(layered_env / ".env")
    assert main.env_config("LLM_BACKEND") == "custom"
