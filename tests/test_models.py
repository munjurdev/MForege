"""Tests for the model catalog (declarative registry) and /model resolution."""
import pytest

import app.models as models
from app.agent.agent import Agent, AgentConfig


class TestRegistryStructure:
    def test_every_model_has_a_known_provider(self):
        for m in models.MODELS:
            assert m.provider in models.PROVIDERS, f"{m.id} → unknown provider"

    def test_short_names_are_unique(self):
        names = [m.name for m in models.MODELS]
        assert len(names) == len(set(names)), "duplicate short names break /model <name>"

    def test_model_ids_are_unique(self):
        ids = [m.id for m in models.MODELS]
        assert len(ids) == len(set(ids))

    def test_groq_free_defaults_present(self):
        ids = {m.id for m in models.models_for("groq")}
        assert "openai/gpt-oss-20b" in ids
        assert "llama-3.3-70b-versatile" in ids

    def test_openrouter_models_are_free_tier(self):
        for m in models.models_for("openrouter"):
            assert m.id.endswith(":free")

    def test_every_provider_has_at_least_one_model(self):
        for key in models.PROVIDERS:
            assert models.models_for(key), f"provider {key} has no models"

    def test_every_provider_has_a_default(self):
        for key in models.PROVIDERS:
            assert models.default_for(key)


class TestResolve:
    def test_by_number(self):
        model, amb = models.resolve("2", "groq")
        assert model is not None and amb == []
        assert model == models.models_for("groq")[1]

    def test_by_short_name(self):
        model, _ = models.resolve("gpt20b", "groq")
        assert model is not None and model.id == "openai/gpt-oss-20b"

    def test_by_exact_id(self):
        model, _ = models.resolve("llama-3.3-70b-versatile", "groq")
        assert model is not None and model.name == "llama70b"

    def test_by_unique_substring(self):
        model, _ = models.resolve("kimi", "groq")
        assert model is not None and model.provider == "groq"

    def test_ambiguous_substring_lists_candidates(self):
        model, amb = models.resolve("llama", "groq")   # llama70b + llama8b
        assert model is None
        assert len(amb) == 2

    def test_custom_id_passes_through(self):
        model, _ = models.resolve("some-totally-new/model-x", "groq")
        assert model is not None
        assert model.id == "some-totally-new/model-x"
        assert model.note == "custom"

    def test_out_of_range_number(self):
        model, _ = models.resolve("99", "groq")
        assert model is None

    def test_empty_token(self):
        model, _ = models.resolve("", "groq")
        assert model is None


class TestProviderForCurrent:
    def test_known_model_maps_to_provider(self):
        assert models.provider_for_current("openai/gpt-oss-20b") == "groq"
        assert models.provider_for_current("deepseek/deepseek-r1:free") == "openrouter"

    def test_custom_model_keeps_fallback(self):
        assert models.provider_for_current("my-own/model", fallback="groq") == "groq"


class TestMenu:
    def test_renders_names_and_current_marker(self):
        lines = models.render_menu("groq", current="openai/gpt-oss-20b")
        assert any("← current" in ln for ln in lines)
        assert any("gpt20b" in ln for ln in lines)


class TestSwitchModel:
    def _agent(self) -> Agent:
        return Agent(
            config=AgentConfig(streaming=False, model="openai/gpt-oss-20b"),
            api_key="x", backend="custom", base_url="http://x/v1",
        )

    def test_switch_updates_config_and_client(self):
        agent = self._agent()
        agent.switch_model("llama-3.3-70b-versatile")
        assert agent.config.model == "llama-3.3-70b-versatile"
        assert agent.llm.model == "llama-3.3-70b-versatile"

    def test_switch_resets_413_cap(self):
        agent = self._agent()
        agent._max_tokens_cap = 512
        agent.switch_model("openai/gpt-oss-120b")
        assert agent._max_tokens_cap is None

    def test_switch_keeps_history(self):
        from app.agent.memory import Message
        agent = self._agent()
        agent.memory.add(Message(role="user", content="building task_api"))
        agent.switch_model("llama-3.3-70b-versatile")
        assert agent.memory.messages[-1].content == "building task_api"

    def test_switch_rejects_empty(self):
        agent = self._agent()
        with pytest.raises(ValueError):
            agent.switch_model("   ")


class TestRunModelCommand:
    """The shared plain/UI handler — pure, no I/O."""

    def _agent(self) -> Agent:
        return Agent(
            config=AgentConfig(streaming=False, model="openai/gpt-oss-20b"),
            api_key="x", backend="custom", base_url="http://x/v1",
        )

    def test_list_shows_menu(self):
        import main as main_mod
        out = main_mod._run_model_command("", self._agent())
        kinds = {k for _, k in out}
        assert "title" in kinds
        assert any("gpt20b" in text for text, _ in out)

    def test_switch_by_number(self, monkeypatch, tmp_path):
        import main as main_mod
        cfg = tmp_path / "env"
        cfg.write_text("LLM_MODEL=openai/gpt-oss-20b\n")
        monkeypatch.setattr(main_mod, "_HOME_CONFIG_PATH", str(cfg))
        agent = self._agent()
        out = main_mod._run_model_command("2", agent)
        assert agent.config.model == "openai/gpt-oss-120b"
        assert out[-1][1] == "ok"
        assert "LLM_MODEL=openai/gpt-oss-120b" in cfg.read_text()

    def test_switch_by_name(self, monkeypatch, tmp_path):
        import main as main_mod
        cfg = tmp_path / "env"
        cfg.write_text("LLM_MODEL=openai/gpt-oss-20b\n")
        monkeypatch.setattr(main_mod, "_HOME_CONFIG_PATH", str(cfg))
        agent = self._agent()
        main_mod._run_model_command("kimi", agent)
        assert "kimi-k2" in agent.config.model

    def test_ambiguous_gives_hint(self):
        import main as main_mod
        out = main_mod._run_model_command("llama", self._agent())
        assert out[-1][1] == "error"
        assert "be specific" in out[-1][0]

    def test_already_on_model(self):
        import main as main_mod
        out = main_mod._run_model_command("gpt20b", self._agent())
        assert any("Already on" in text for text, _ in out)
