"""Tests for the Freebuff-style auto-update flow.

Covers: get_available_update() data API, the loop-guard / opt-out logic,
the helper command shape, perform_update()'s spawn+exit contract, and the
main() integration points (--no-update flag, env opt-out, cache-only
decisions, fallback when the helper fails).
"""
import json

import pytest

import app.self_update as su
import app.update_check as uc
import main


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """Isolated cache + no real network + no real detached processes."""
    cache = tmp_path / "update_check.json"
    monkeypatch.setattr(uc, "CACHE_PATH", str(cache))
    monkeypatch.setattr("urllib.request.urlopen", _fail_urlopen)
    spawned = []
    monkeypatch.setattr(
        su, "_spawn_detached",
        lambda cmd, flags, env: spawned.append((cmd, flags, env)),
    )
    return {"cache": cache, "spawned": spawned}


class _ExitCalled(Exception):
    """Raised by SystemExit interception in tests that need to inspect it."""


def _expect_system_exit(fn, *a, **k) -> int:
    """Run fn(), catch its SystemExit, return the exit code."""
    try:
        fn(*a, **k)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0
    raise AssertionError("expected SystemExit")


def _fail_urlopen(*a, **k):
    raise OSError("no network in tests")


def _helper_json_arg(cmd: list, index: int) -> list:
    """Extract the JSON-encoded sub-command the helper receives via argv."""
    return json.loads(cmd[index])


def _fake_pypi(version: str):
    import io

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    payload = json.dumps({"info": {"version": version}}).encode()

    def _open(*a, **k):
        return FakeResp(payload)
    return _open


# ── get_available_update: data API ─────────────────────────────────────

class TestGetAvailableUpdate:
    def test_newer_version_returned(self, isolated, monkeypatch):
        monkeypatch.setattr("urllib.request.urlopen", _fake_pypi("99.0.0"))
        assert uc.get_available_update("0.1.1", force=True) == "99.0.0"

    def test_same_version_returns_empty(self, isolated, monkeypatch):
        monkeypatch.setattr("urllib.request.urlopen", _fake_pypi("0.1.1"))
        assert uc.get_available_update("0.1.1", force=True) == ""

    def test_network_failure_returns_empty(self, isolated):
        assert uc.get_available_update("0.1.1", force=True) == ""

    def test_cache_hit_answers_without_network(self, isolated):
        isolated["cache"].write_text(
            json.dumps({"latest": "9.9.9", "checked_at": 1}), encoding="utf-8")
        assert uc.get_available_update("0.1.0") == "9.9.9"  # urlopen would fail

    def test_check_for_update_still_works(self, isolated, monkeypatch):
        monkeypatch.setattr("urllib.request.urlopen", _fake_pypi("99.0.0"))
        banner = uc.check_for_update("0.1.1", force=True)
        assert banner and "99.0.0" in banner


# ── guards ─────────────────────────────────────────────────────────────

class TestGuards:
    def test_disabled_by_env_opt_out(self, monkeypatch):
        monkeypatch.setenv(su.ENV_NO_UPDATE, "1")
        assert su.updates_disabled() is True

    def test_disabled_by_inner_relaunch(self, monkeypatch):
        monkeypatch.setenv(su.ENV_UPDATING, "1")
        assert su.updates_disabled() is True

    def test_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv(su.ENV_NO_UPDATE, raising=False)
        monkeypatch.delenv(su.ENV_UPDATING, raising=False)
        assert su.updates_disabled() is False

    def test_explicit_dict_not_affected_by_os_environ(self, monkeypatch):
        monkeypatch.setenv(su.ENV_NO_UPDATE, "1")
        assert su.updates_disabled({}) is False


# ── helper command ─────────────────────────────────────────────────────

class TestHelperCommand:
    def test_shape(self):
        cmd = su._helper_command()
        assert cmd[0] == su.sys.executable
        assert cmd[1] == "-c"
        upgrade = _helper_json_arg(cmd, 3)
        relaunch = _helper_json_arg(cmd, 4)
        assert "pip" in upgrade and "mforege" in upgrade
        assert relaunch == ["mforege"]

    def test_helper_source_compiles(self):
        compile(su._helper_source(), "<helper>", "exec")

    def test_helper_upgrades_and_relaunches(self, isolated):
        code = _expect_system_exit(su.perform_update)
        assert code == 0
        cmd = isolated["spawned"][0][0]
        helper_code = cmd[2]
        upgrade = _helper_json_arg(cmd, 3)
        relaunch = _helper_json_arg(cmd, 4)
        assert "pip" in upgrade and "mforege" in upgrade
        # the embedded helper references relaunch + attach-console steps
        assert "AttachConsole" in helper_code
        assert "Popen" in helper_code

    def test_user_fallback_is_a_real_list_concat(self):
        # Regression: '--user' fallback must append to a LIST (a string
        # command + list raised TypeError, silently killing the fallback).
        helper_code = su._helper_source()
        assert 'upgrade_cmd + ["--user"]' in helper_code
        assert '"--user" not in upgrade_cmd' in helper_code


# ── perform_update contract ────────────────────────────────────────────

class TestPerformUpdate:
    def test_spawns_detached_with_loop_guard_and_exits(self, isolated, monkeypatch):
        monkeypatch.delenv(su.ENV_UPDATING, raising=False)
        code = _expect_system_exit(su.perform_update)
        assert code == 0
        cmd, flags, env = isolated["spawned"][0]
        assert env[su.ENV_UPDATING] == "1"      # guard travels to the relaunch
        assert flags == su.DETACHED_FLAGS       # fully detached on Windows

    def test_falls_back_when_helper_cannot_start(self, isolated, monkeypatch):
        def boom(*a, **k):
            raise OSError("spawn failed")
        monkeypatch.setattr(su, "_spawn_detached", boom)
        # must NOT raise: old version launches as usual
        su.perform_update()


class TestRelaunchFlags:
    def test_relaunch_is_attached_not_detached(self):
        # Regression: the relaunched app must carry NO creation flags — the
        # helper source must not hardcode DETACHED_PROCESS into Popen.
        helper_code = su._helper_source()
        popen_calls = [ln for ln in helper_code.splitlines()
                       if "Popen(relaunch_cmd" in ln]
        assert popen_calls and "creationflags" not in popen_calls[0]

    def test_helper_source_compiles_and_runs_relaunch_posix_safe(self):
        # The embedded source must be syntactically valid (it runs via -c)
        code = su._helper_source()
        compile(code, "<helper>", "exec")
        assert 'os.name == "nt"' in code  # AttachConsole guarded per-OS


# ── main() integration ─────────────────────────────────────────────────

class _StubUI:
    """Sentinel: reaching ChatUI(...) means main() got past the update step."""

    def __init__(self, *a, **k):
        raise _ReachedUI()


class _ReachedUI(Exception):
    pass


class TestMainIntegration:
    def _prep(self, monkeypatch, tmp_path):
        """Common stubs: PyPI says 99.0.0; ChatUI raises _ReachedUI; wizard
        raises _ReachedUI too (so both code paths funnel into one signal).

        Hermetic config: CI runners have no ~/.mforege/.env and no ./.env,
        so main() would fire the setup wizard or exit(1) on a missing key
        before ever reaching the update step. A fake global config makes
        the environment identical to a configured machine — and chdir away
        from the repo so a developer's local .env can't leak in either.
        """
        monkeypatch.setattr("urllib.request.urlopen", _fake_pypi("99.0.0"))
        monkeypatch.setattr(main, "ChatUI", _StubUI)
        monkeypatch.setattr(main, "run_setup_wizard", _ReachedUI)
        home = tmp_path / "home"
        cfg_dir = home / ".mforege"
        cfg_dir.mkdir(parents=True)
        cfg_file = cfg_dir / ".env"
        cfg_file.write_text(
            "LLM_BACKEND=custom\nAPI_KEY=test-key\nBASE_URL=http://localhost:1\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)                       # no ./.env
        monkeypatch.setattr(main, "_HOME_CONFIG_PATH", str(cfg_file))
        monkeypatch.setattr(main, "_root_config", None)   # no repo .env
        monkeypatch.setattr(main, "_MFOREGE_ROOT", str(tmp_path))
        monkeypatch.setattr(
            main, "_home_config", main.Config(main.RepositoryEnv(str(cfg_file)))
        )

    def _run_main(self, argv):
        try:
            main.asyncio.run(main.main(argv))
        except _ReachedUI:
            return "no-update-path"
        except SystemExit as e:
            if isinstance(e.code, int) and e.code == 0:
                return "updated"
            raise
        return "no-update-path"

    def test_main_triggers_update_by_default(self, isolated, monkeypatch, tmp_path):
        self._prep(monkeypatch, tmp_path)
        assert self._run_main([]) == "updated"

    def test_main_skips_update_with_flag(self, isolated, monkeypatch, tmp_path):
        self._prep(monkeypatch, tmp_path)
        assert self._run_main(["--no-update"]) == "no-update-path"

    def test_main_skips_update_with_env(self, isolated, monkeypatch, tmp_path):
        self._prep(monkeypatch, tmp_path)
        monkeypatch.setenv(su.ENV_NO_UPDATE, "1")
        assert self._run_main([]) == "no-update-path"

    def test_main_help_wins_over_update(self, isolated, monkeypatch, capsys, tmp_path):
        # --help must print help and exit 0 without any update side effects
        self._prep(monkeypatch, tmp_path)
        with pytest.raises(SystemExit) as ei:
            main.asyncio.run(main.main(["--help"]))
        assert ei.value.code == 0
        assert "--no-update" in capsys.readouterr().out


