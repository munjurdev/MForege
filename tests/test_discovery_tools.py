"""Tests for discovery tools: search_code, glob_files, and read_file windows"""
import os

import pytest

from app.tools.system_tools import (
    SearchCodeTool,
    GlobFilesTool,
    ReadFileTool,
    SKIP_DIRS,
)


@pytest.fixture
def ws(tmp_path):
    """Workspace with a small multi-file python project"""
    app = tmp_path / "app"
    app.mkdir()
    (app / "agent.py").write_text(
        "def create_agent():\n    pass\n\n\ndef other():\n    pass\n",
        encoding="utf-8",
    )
    (app / "tools.py").write_text(
        "def create_tool():\n    pass\n# TODO: improve\n",
        encoding="utf-8",
    )
    (app / "readme.md").write_text("# Project\n", encoding="utf-8")
    sub = app / "sub"
    sub.mkdir()
    (sub / "deep.py").write_text("def create_agent():\n    pass\n", encoding="utf-8")
    junk = tmp_path / "__pycache__"
    junk.mkdir()
    (junk / "junk.py").write_text("def create_agent():\n    pass\n", encoding="utf-8")
    return str(tmp_path)


class TestSearchCode:
    @pytest.mark.asyncio
    async def test_finds_matches_with_file_line(self, ws):
        tool = SearchCodeTool(workspace=ws)
        result = await tool.execute(pattern="def create_")
        assert "app/agent.py:1" in result
        assert "app/tools.py:1" in result

    @pytest.mark.asyncio
    async def test_skips_noise_dirs(self, ws):
        tool = SearchCodeTool(workspace=ws)
        result = await tool.execute(pattern="def create_")
        assert "__pycache__" not in result

    @pytest.mark.asyncio
    async def test_respects_max_results(self, ws):
        tool = SearchCodeTool(workspace=ws)
        result = await tool.execute(pattern="def ", max_results=2)
        assert "stopped at 2 matches" in result

    @pytest.mark.asyncio
    async def test_no_matches_message(self, ws):
        tool = SearchCodeTool(workspace=ws)
        result = await tool.execute(pattern="zzz_not_here_zzz")
        assert "No matches" in result
        assert "scanned" in result

    @pytest.mark.asyncio
    async def test_invalid_regex_reported(self, ws):
        tool = SearchCodeTool(workspace=ws)
        result = await tool.execute(pattern="([unclosed")
        assert "invalid regex" in result

    @pytest.mark.asyncio
    async def test_empty_pattern(self, ws):
        tool = SearchCodeTool(workspace=ws)
        result = await tool.execute(pattern="  ")
        assert "empty pattern" in result

    @pytest.mark.asyncio
    async def test_case_insensitive_by_default(self, ws):
        tool = SearchCodeTool(workspace=ws)
        result = await tool.execute(pattern="TODO")
        assert "app/tools.py" in result

    def test_skip_dirs_sane(self):
        assert "__pycache__" in SKIP_DIRS
        assert "node_modules" in SKIP_DIRS
        assert ".git" in SKIP_DIRS


class TestGlobFiles:
    @pytest.mark.asyncio
    async def test_all_python_files_recursive(self, ws):
        tool = GlobFilesTool(workspace=ws)
        result = await tool.execute(pattern="**/*.py")
        assert "app/agent.py" in result
        assert "app/sub/deep.py" in result
        assert "__pycache__" not in result

    @pytest.mark.asyncio
    async def test_bare_star_pattern_any_depth(self, ws):
        tool = GlobFilesTool(workspace=ws)
        result = await tool.execute(pattern="*.md")
        assert "app/readme.md" in result

    @pytest.mark.asyncio
    async def test_no_matches(self, ws):
        tool = GlobFilesTool(workspace=ws)
        result = await tool.execute(pattern="**/*.rs")
        assert "No files matching" in result

    @pytest.mark.asyncio
    async def test_prefix_pattern(self, ws):
        tool = GlobFilesTool(workspace=ws)
        result = await tool.execute(pattern="**/agent*.py")
        assert "app/agent.py" in result
        assert "tools.py" not in result.split("matching")[0].split(":")[-1]


class TestReadFileWindows:
    @pytest.mark.asyncio
    async def test_offset_starts_at_line(self, tmp_path):
        content = "\n".join(f"line{i}" for i in range(1, 51)) + "\n"
        (tmp_path / "big.txt").write_text(content, encoding="utf-8")
        tool = ReadFileTool(workspace=str(tmp_path))
        result = await tool.execute(path="big.txt", offset=10, limit=5)
        assert "10| line10" in result
        assert "line9" not in result
        assert "line14" in result  # offset 10 + limit 5 = lines 10..14
        assert "line15" not in result

    @pytest.mark.asyncio
    async def test_continue_hint(self, tmp_path):
        content = "\n".join(f"line{i}" for i in range(1, 51)) + "\n"
        (tmp_path / "big.txt").write_text(content, encoding="utf-8")
        tool = ReadFileTool(workspace=str(tmp_path))
        result = await tool.execute(path="big.txt", offset=1, limit=10)
        assert "offset=11" in result

    @pytest.mark.asyncio
    async def test_offset_beyond_end(self, tmp_path):
        (tmp_path / "short.txt").write_text("one\n", encoding="utf-8")
        tool = ReadFileTool(workspace=str(tmp_path))
        result = await tool.execute(path="short.txt", offset=100)
        assert "Contents of" in result  # empty window, but no crash

    @pytest.mark.asyncio
    async def test_limit_capped_at_500(self, tmp_path):
        content = "\n".join(f"line{i}" for i in range(1, 1001)) + "\n"
        (tmp_path / "huge.txt").write_text(content, encoding="utf-8")
        tool = ReadFileTool(workspace=str(tmp_path))
        result = await tool.execute(path="huge.txt", limit=5000)
        # Cap applies; output gets truncated before all 500 lines show
        assert "lines 1-500" in result

    @pytest.mark.asyncio
    async def test_default_reads_200(self, tmp_path):
        content = "\n".join(f"line{i}" for i in range(1, 301)) + "\n"
        (tmp_path / "med.txt").write_text(content, encoding="utf-8")
        tool = ReadFileTool(workspace=str(tmp_path))
        result = await tool.execute(path="med.txt")
        assert "line200" in result
        assert "line201" not in result
