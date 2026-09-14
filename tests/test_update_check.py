"""Tests for the PyPI update checker (all network paths are mocked)"""
import json
import os

import pytest

import app.update_check as uc


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Point the cache at a temp dir so tests never touch the real one."""
    cache = tmp_path / "update_check.json"
    monkeypatch.setattr(uc, "CACHE_PATH", str(cache))
    return cache


class TestVersionCompare:
    def test_higher_patch(self):
        assert uc.is_newer("0.1.2", "0.1.1") is True

    def test_higher_minor(self):
        assert uc.is_newer("0.2.0", "0.1.9") is True

    def test_numeric_not_alphabetic(self):
        # alphabetic compare would say "10" < "9"
        assert uc.is_newer("1.10.0", "1.9.0") is True
        assert uc.is_newer("1.2.10", "1.2.9") is True

    def test_same_version(self):
        assert uc.is_newer("0.1.1", "0.1.1") is False

    def test_older_remote(self):
        assert uc.is_newer("0.1.0", "0.1.1") is False

    def test_garbage_versions(self):
        assert uc.is_newer("", "0.1.1") is False
        assert uc.is_newer("abc", "0.1.1") is False


class TestNetworkPaths:
    def test_newer_version_returns_banner(self, monkeypatch):
        monkeypatch.setattr(uc, "CHECK_URL", "http://localhost:1")  # will fail
        # mock urlopen instead of relying on connection failure
        import io

        class FakeResp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        payload = json.dumps({"info": {"version": "99.0.0"}}).encode()
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda *a, **k: FakeResp(payload)
        )
        banner = uc.check_for_update("0.1.1", force=True)
        assert banner is not None
        assert "99.0.0" in banner
        assert "pip install --upgrade mforege" in banner

    def test_same_version_returns_none(self, monkeypatch):
        import io

        class FakeResp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        payload = json.dumps({"info": {"version": "0.1.1"}}).encode()
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda *a, **k: FakeResp(payload)
        )
        assert uc.check_for_update("0.1.1", force=True) is None

    def test_network_failure_is_silent(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("no network")

        monkeypatch.setattr("urllib.request.urlopen", boom)
        assert uc.check_for_update("0.1.1", force=True) is None

    def test_malformed_response_is_silent(self, monkeypatch):
        import io

        class FakeResp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **k: FakeResp(b"not json at all"),
        )
        assert uc.check_for_update("0.1.1", force=True) is None


class TestCache:
    def test_second_call_uses_cache_no_network(self, monkeypatch):
        import io

        calls = []

        class FakeResp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(*a, **k):
            calls.append(1)
            return FakeResp(json.dumps({"info": {"version": "99.0.0"}}).encode())

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        first = uc.check_for_update("0.1.1", force=True)
        assert first is not None
        assert len(calls) == 1

        # second call, NOT forced: served from cache, no network hit
        second = uc.check_for_update("0.1.1")
        assert second is not None
        assert "99.0.0" in second
        assert len(calls) == 1  # still one network call

    def test_cached_fresh_version_returns_none(self, monkeypatch):
        import io

        class FakeResp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **k: FakeResp(json.dumps({"info": {"version": "0.1.1"}}).encode()),
        )
        assert uc.check_for_update("0.1.1", force=True) is None
        # cached "same version" -> None without network
        assert uc.check_for_update("0.1.1") is None
