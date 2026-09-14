"""Tests for session persistence (save / list / resume across restarts)"""
import pytest

from app.agent.sessions import SessionStore
from app.agent.agent import Agent, AgentConfig


@pytest.fixture
def store(tmp_path):
    return SessionStore(directory=str(tmp_path / "sessions"))


@pytest.fixture
def agent():
    return Agent(
        config=AgentConfig(streaming=False),
        api_key="x",
        backend="custom",
        base_url="http://localhost:9999",  # never called in these tests
    )


class TestSessionStore:
    def test_start_gives_unique_ids(self, store):
        s1 = store.start()
        s2 = store.start()
        s3 = store.start()
        assert len({s1.id, s2.id, s3.id}) == 3

    def test_save_and_list(self, store):
        s = store.start()
        s.append("hello", "Hi there!")
        assert store.save(s) is True
        rows = store.list()
        assert len(rows) == 1
        assert rows[0]["count"] == 2

    def test_auto_title_from_first_user_message(self, store):
        s = store.start()
        s.append("build me a calculator please", "Done!")
        store.save(s)
        rows = store.list()
        assert "build me a calculator" in rows[0]["title"]

    def test_load_returns_messages(self, store):
        s = store.start()
        s.append("q1", "a1")
        s.append("q2", "a2")
        store.save(s)
        loaded = store.load(s.id)
        assert loaded is not None
        assert loaded.messages == [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]

    def test_load_missing_returns_none(self, store):
        assert store.load("no-such-id") is None

    def test_list_skips_corrupt_files(self, store):
        s = store.start()
        s.append("q", "a")
        store.save(s)
        # drop a corrupt file next to it
        bad = store.directory + "/garbage.json"
        with open(bad, "w", encoding="utf-8") as f:
            f.write("{not json")
        rows = store.list()
        assert len(rows) == 1  # corrupt skipped silently

    def test_newest_first_ordering(self, store):
        import time
        s1 = store.start()
        s1.append("old", "chat")
        store.save(s1)
        time.sleep(0.02)
        s2 = store.start()
        s2.append("new", "chat")
        store.save(s2)
        rows = store.list()
        assert rows[0]["id"] == s2.id

    def test_prune_keeps_only_newest(self, store):
        store.MAX_SESSIONS = 3
        for i in range(5):
            s = store.start()
            s.append(f"msg {i}", "ok")
            store.save(s)
            import time
            time.sleep(0.01)
        rows = store.list(limit=100)
        assert len(rows) <= 3


class TestRestoreIntoAgent:
    def test_restore_round_trip(self, store, agent):
        s = store.start()
        s.append("what is my project?", "It's a Django e-commerce site.")
        store.save(s)
        loaded = store.load(s.id)
        agent.restore_session_messages(loaded.messages)
        roles = [m.role for m in agent.memory.messages]
        assert roles == ["user", "assistant"]
        # the restored context flows into API messages
        api = agent._build_api_messages()
        joined = " ".join(m["content"] for m in api)
        assert "Django" in joined

    def test_restore_skips_malformed_entries(self, agent):
        agent.restore_session_messages([
            {"role": "tool", "content": "x"},      # tool role not restored
            {"role": "user"},                       # missing content
            {"bad": True},                          # garbage
            {"role": "user", "content": "ok"},      # valid
        ])
        assert [m.content for m in agent.memory.messages] == ["ok"]

    def test_restore_clears_condensed_summary(self, agent):
        agent.memory.summary = "old folded stuff"
        agent.restore_session_messages([{"role": "user", "content": "hi"}])
        assert agent.memory.summary == ""
        api = agent._build_api_messages()
        assert not any("folded" in m["content"] for m in api)

    def test_restore_empty_is_safe(self, agent):
        agent.restore_session_messages([])
        assert agent.memory.messages == []
        agent.restore_session_messages(None)
        assert agent.memory.messages == []
