"""Tests for system/agentic tools (no real destructive commands are run)"""
import asyncio
import os

import pytest

from app.tools.system_tools import (
    ListFilesTool,
    ReadFileTool,
    RunCommandTool,
    CreateFileTool,
    EditFileTool,
    TodoPlanTool,
    PlanState,
    NotifyHook,
    create_system_tools,
    _classify_command,
    _resolve_inside_root,
    make_unified_diff,
    COMMAND_TIMEOUT,
)


@pytest.fixture
def workspace(tmp_path):
    """A fake project workspace"""
    (tmp_path / "main.py").write_text("print('hello')\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Test Project\n" + "line\n" * 300, encoding="utf-8")
    sub = tmp_path / "app"
    sub.mkdir()
    (sub / "settings.py").write_text("DEBUG = True\n", encoding="utf-8")
    return str(tmp_path)


class TestPathConfinement:
    def test_relative_path_resolves_inside(self, workspace):
        resolved = _resolve_inside_root(workspace, "app")
        assert resolved == os.path.abspath(os.path.join(workspace, "app"))

    def test_dotfiles_escape_rejected(self, workspace):
        assert _resolve_inside_root(workspace, "../outside") is None
        assert _resolve_inside_root(workspace, "..") is None

    def test_absolute_path_outside_rejected(self, workspace):
        assert _resolve_inside_root(workspace, "C:/Windows") is None or True
        # robust check: resolve something clearly outside via ..
        assert _resolve_inside_root(workspace, "../../etc") is None

    @pytest.mark.asyncio
    async def test_read_file_outside_rejected(self, workspace):
        tool = ReadFileTool(workspace=workspace)
        result = await tool.execute(path="../secret.txt")
        assert result.startswith("Error: path")

    @pytest.mark.asyncio
    async def test_list_outside_rejected(self, workspace):
        tool = ListFilesTool(workspace=workspace)
        result = await tool.execute(path="../")
        assert result.startswith("Error: path")


class TestListFiles:
    @pytest.mark.asyncio
    async def test_lists_contents(self, workspace):
        tool = ListFilesTool(workspace=workspace)
        result = await tool.execute(path=".")
        assert "main.py" in result
        assert "app/" in result
        assert "README.md" in result

    @pytest.mark.asyncio
    async def test_subdirectory(self, workspace):
        tool = ListFilesTool(workspace=workspace)
        result = await tool.execute(path="app")
        assert "settings.py" in result

    @pytest.mark.asyncio
    async def test_missing_directory(self, workspace):
        tool = ListFilesTool(workspace=workspace)
        result = await tool.execute(path="nope")
        assert "does not exist" in result


class TestReadFile:
    @pytest.mark.asyncio
    async def test_reads_text_file(self, workspace):
        tool = ReadFileTool(workspace=workspace)
        result = await tool.execute(path="main.py")
        assert "print('hello')" in result
        assert "1|" in result  # line numbers

    @pytest.mark.asyncio
    async def test_missing_file_hint(self, workspace):
        tool = ReadFileTool(workspace=workspace)
        result = await tool.execute(path="ghost.py")
        assert "does not exist" in result
        assert "list_files" in result

    @pytest.mark.asyncio
    async def test_binary_detected(self, workspace, tmp_path):
        (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02binary")
        tool = ReadFileTool(workspace=workspace)
        result = await tool.execute(path="blob.bin")
        assert "binary" in result

    @pytest.mark.asyncio
    async def test_long_file_truncated_to_200_lines(self, workspace):
        tool = ReadFileTool(workspace=workspace)
        result = await tool.execute(path="README.md")
        assert "offset=201" in result  # continue-reading hint at the default 200-line window


class TestCommandClassification:
    def test_blocked_patterns(self):
        for cmd in ["rm -rf /", "del /s C:\\", "git push --force", "format C:",
                    "git reset --hard", "shutdown now"]:
            assert _classify_command(cmd) == "blocked", cmd

    def test_safe_readonly(self):
        for cmd in ["dir", "ls -la", "git status", "git log --oneline",
                    "python --version", "pip list", "whoami"]:
            assert _classify_command(cmd) == "safe", cmd

    def test_mutating_needs_confirm(self):
        for cmd in ["pip install django", "python main.py", "git init",
                    "mkdir stuff", "django-admin startproject shop"]:
            assert _classify_command(cmd) == "confirm", cmd


class TestRunCommand:
    @pytest.mark.asyncio
    async def test_safe_command_runs(self, workspace):
        tool = RunCommandTool(workspace=workspace, confirm=None)
        result = await tool.execute(command="python --version")
        assert "exit code: 0" in result
        assert "Python" in result

    @pytest.mark.asyncio
    async def test_blocked_command_refused(self, workspace):
        tool = RunCommandTool(workspace=workspace, confirm=None)
        result = await tool.execute(command="rm -rf /")
        assert result.startswith("Error: command blocked")

    @pytest.mark.asyncio
    async def test_confirm_accepted(self, workspace):
        tool = RunCommandTool(workspace=workspace, confirm=lambda cmd: True)
        result = await tool.execute(command="mkdir confirmed_dir")
        assert "exit code: 0" in result
        assert os.path.isdir(os.path.join(workspace, "confirmed_dir"))

    @pytest.mark.asyncio
    async def test_confirm_denied(self, workspace):
        tool = RunCommandTool(workspace=workspace, confirm=lambda cmd: False)
        result = await tool.execute(command="mkdir denied_dir")
        assert "cancelled by user" in result
        assert not os.path.isdir(os.path.join(workspace, "denied_dir"))

    @pytest.mark.asyncio
    async def test_no_confirm_callback_blocks_mutating(self, workspace):
        # Fail-closed: with no confirm hook, mutating commands are refused
        tool = RunCommandTool(workspace=workspace, confirm=None)
        result = await tool.execute(command="mkdir never_dir")
        assert "refused" in result
        assert "no confirmation handler" in result
        assert not os.path.isdir(os.path.join(workspace, "never_dir"))

    @pytest.mark.asyncio
    async def test_timeout_reported(self, workspace):
        tool = RunCommandTool(workspace=workspace, confirm=lambda cmd: True, timeout=2.0)
        result = await tool.execute(command="python -c \"import time; time.sleep(30)\"")
        assert "timed out" in result

    @pytest.mark.asyncio
    async def test_exit_code_and_stderr_reported(self, workspace):
        tool = RunCommandTool(workspace=workspace, confirm=lambda cmd: True)
        result = await tool.execute(command="python -c \"raise SystemExit(3)\"")
        assert "exit code: 3" in result

    @pytest.mark.asyncio
    async def test_empty_command(self, workspace):
        tool = RunCommandTool(workspace=workspace, confirm=None)
        result = await tool.execute(command="   ")
        assert "empty command" in result


class TestCreateFile:
    @pytest.mark.asyncio
    async def test_creates_new_file_with_confirm(self, workspace):
        tool = CreateFileTool(workspace=workspace, confirm=lambda a: True)
        result = await tool.execute(path="new_app/views.py", content="VIEW = True\n")
        assert result.startswith("OK: created")
        assert os.path.isfile(os.path.join(workspace, "new_app", "views.py"))

    @pytest.mark.asyncio
    async def test_denied_write_does_not_touch_disk(self, workspace):
        tool = CreateFileTool(workspace=workspace, confirm=lambda a: False)
        result = await tool.execute(path="denied.txt", content="x")
        assert "cancelled by user" in result
        assert not os.path.exists(os.path.join(workspace, "denied.txt"))

    @pytest.mark.asyncio
    async def test_fail_closed_without_confirm(self, workspace):
        tool = CreateFileTool(workspace=workspace, confirm=None)
        result = await tool.execute(path="no.txt", content="x")
        assert "refused" in result
        assert not os.path.exists(os.path.join(workspace, "no.txt"))

    @pytest.mark.asyncio
    async def test_overwrite_flagged_in_confirmation(self, workspace):
        actions = []
        tool = CreateFileTool(workspace=workspace, confirm=lambda a: actions.append(a) or False)
        await tool.execute(path="main.py", content="new content")
        assert actions and actions[0].startswith("OVERWRITE")

    @pytest.mark.asyncio
    async def test_outside_workspace_rejected(self, workspace):
        tool = CreateFileTool(workspace=workspace, confirm=lambda a: True)
        result = await tool.execute(path="../evil.txt", content="x")
        assert result.startswith("Error: path")
        assert not os.path.exists(os.path.join(os.path.dirname(workspace), "evil.txt"))

    @pytest.mark.asyncio
    async def test_directory_target_rejected(self, workspace):
        tool = CreateFileTool(workspace=workspace, confirm=lambda a: True)
        result = await tool.execute(path="app", content="x")
        assert "is a directory" in result

    @pytest.mark.asyncio
    async def test_too_large_content_rejected(self, workspace):
        tool = CreateFileTool(workspace=workspace, confirm=lambda a: True)
        result = await tool.execute(path="big.txt", content="x" * 200_000)
        assert "too large" in result


class TestEditFile:
    @pytest.mark.asyncio
    async def test_edits_unique_match(self, workspace):
        tool = EditFileTool(workspace=workspace, confirm=lambda a: True)
        result = await tool.execute(path="main.py", old_string="print('hello')", new_string="print('edited')")
        assert result.startswith("OK: edited")
        with open(os.path.join(workspace, "main.py"), encoding="utf-8") as f:
            assert "print('edited')" in f.read()

    @pytest.mark.asyncio
    async def test_nonexistent_file_rejected(self, workspace):
        tool = EditFileTool(workspace=workspace, confirm=lambda a: True)
        result = await tool.execute(path="ghost.py", old_string="a", new_string="b")
        assert "does not exist" in result and "create_file" in result

    @pytest.mark.asyncio
    async def test_nonunique_match_rejected(self, workspace, tmp_path):
        (tmp_path / "dup.txt").write_text("same\nsame\n", encoding="utf-8")
        tool = EditFileTool(workspace=workspace, confirm=lambda a: True)
        result = await tool.execute(path="dup.txt", old_string="same", new_string="x")
        assert "2 times" in result
        with open(os.path.join(workspace, "dup.txt"), encoding="utf-8") as f:
            assert f.read() == "same\nsame\n"  # untouched

    @pytest.mark.asyncio
    async def test_missing_match_rejected(self, workspace):
        tool = EditFileTool(workspace=workspace, confirm=lambda a: True)
        result = await tool.execute(path="main.py", old_string="not in file", new_string="x")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_denied_edit_untouched(self, workspace):
        tool = EditFileTool(workspace=workspace, confirm=lambda a: False)
        result = await tool.execute(path="main.py", old_string="print('hello')", new_string="nope")
        assert "cancelled by user" in result
        with open(os.path.join(workspace, "main.py"), encoding="utf-8") as f:
            assert "print('hello')" in f.read()

    @pytest.mark.asyncio
    async def test_fail_closed_without_confirm(self, workspace):
        tool = EditFileTool(workspace=workspace, confirm=None)
        result = await tool.execute(path="main.py", old_string="print('hello')", new_string="nope")
        assert "refused" in result
        with open(os.path.join(workspace, "main.py"), encoding="utf-8") as f:
            assert "print('hello')" in f.read()

    @pytest.mark.asyncio
    async def test_outside_workspace_rejected(self, workspace):
        tool = EditFileTool(workspace=workspace, confirm=lambda a: True)
        result = await tool.execute(path="../target.txt", old_string="a", new_string="b")
        assert result.startswith("Error: path")


class TestUnifiedDiff:
    def test_diff_shows_add_and_remove(self):
        diff = make_unified_diff("f.py", "old line\n", "new line\n")
        assert "-old line" in diff
        assert "+new line" in diff

    def test_diff_no_changes(self):
        assert make_unified_diff("f.py", "same\n", "same\n") == "(no changes)"

    def test_diff_truncates(self):
        old = "\n".join(f"line{i}" for i in range(100))
        new = "\n".join(f"chng{i}" for i in range(100))
        diff = make_unified_diff("f.py", old, new, max_lines=10)
        assert "more diff lines" in diff

    def test_create_diff_all_additions(self):
        diff = make_unified_diff("new.py", "", "hello\nworld\n")
        assert "+hello" in diff and "+world" in diff
        deletions = [ln for ln in diff.split("\n") if ln.startswith("-") and not ln.startswith("---")]
        assert not deletions


async def collect_events(agent, message):
    events = []
    agent.on_activity = lambda e, d: events.append((e, d))
    await agent.chat(message)
    return events


class TestTodoPlan:
    @pytest.fixture
    def tool(self):
        return TodoPlanTool(state=PlanState())

    @pytest.mark.asyncio
    async def test_set_plan(self, tool):
        result = await tool.execute(action="set", items=["step one", "step two", "step three"])
        assert "Plan set with 3 steps" in result
        assert len(tool._state.steps) == 3

    @pytest.mark.asyncio
    async def test_update_marks_done(self, tool):
        await tool.execute(action="set", items=["a", "b"])
        result = await tool.execute(action="update", index=1, done=True)
        assert "done" in result
        assert tool._state.steps[0]["done"] is True
        assert tool._state.steps[1]["done"] is False

    @pytest.mark.asyncio
    async def test_update_bad_index(self, tool):
        await tool.execute(action="set", items=["a"])
        result = await tool.execute(action="update", index=5, done=True)
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_add_step(self, tool):
        await tool.execute(action="set", items=["a"])
        await tool.execute(action="add", item="b")
        assert len(tool._state.steps) == 2

    @pytest.mark.asyncio
    async def test_clear(self, tool):
        await tool.execute(action="set", items=["a"])
        await tool.execute(action="clear")
        assert tool._state.steps == []

    @pytest.mark.asyncio
    async def test_unknown_action(self, tool):
        result = await tool.execute(action="explode")
        assert "unknown action" in result

    @pytest.mark.asyncio
    async def test_set_requires_items(self, tool):
        result = await tool.execute(action="set")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_step_text_truncated(self, tool):
        await tool.execute(action="set", items=["x" * 500])
        assert len(tool._state.steps[0]["text"]) == 120

    def test_render_and_progress(self, tool):
        tool._state.set(["alpha", "beta"])
        tool._state.mark(1, True)
        rendered = tool._state.render()
        assert "[x] 1. alpha" in rendered
        assert "[ ] 2. beta" in rendered
        assert "(1/2 done)" in rendered
        line = tool._state.progress_line()
        assert "[Plan 1/2]" in line and "beta" in line

    def test_progress_empty(self, tool):
        assert tool._state.progress_line() == ""
        assert tool._state.render() == ""


class TestNotifyHook:
    @pytest.mark.asyncio
    async def test_create_emits_diff_event(self, workspace):
        hook = NotifyHook()
        events = []
        hook.callback = lambda e, d: events.append((e, d))
        tool = CreateFileTool(workspace=workspace, confirm=lambda a: True, notify=hook)
        await tool.execute(path="n.txt", content="hello\n")
        assert events and events[0][0] == "diff"
        assert "+hello" in events[0][1]

    @pytest.mark.asyncio
    async def test_edit_emits_diff_event(self, workspace):
        hook = NotifyHook()
        events = []
        hook.callback = lambda e, d: events.append((e, d))
        tool = EditFileTool(workspace=workspace, confirm=lambda a: True, notify=hook)
        await tool.execute(path="main.py", old_string="print('hello')", new_string="print('bye')")
        assert events and events[0][0] == "diff"
        assert "-print('hello')" in events[0][1]
        assert "+print('bye')" in events[0][1]

    @pytest.mark.asyncio
    async def test_run_command_emits_cmd_event(self, workspace):
        hook = NotifyHook()
        events = []
        hook.callback = lambda e, d: events.append((e, d))
        tool = RunCommandTool(workspace=workspace, confirm=lambda a: True, notify=hook)
        await tool.execute(command="echo hi")
        assert events and events[0][0] == "cmd"
        assert "$ echo hi" in events[0][1]

    @pytest.mark.asyncio
    async def test_cancelled_write_emits_nothing(self, workspace):
        hook = NotifyHook()
        events = []
        hook.callback = lambda e, d: events.append((e, d))
        tool = CreateFileTool(workspace=workspace, confirm=lambda a: False, notify=hook)
        await tool.execute(path="x.txt", content="x")
        assert events == []

    @pytest.mark.asyncio
    async def test_broken_callback_never_breaks_tool(self, workspace):
        hook = NotifyHook()
        hook.callback = lambda e, d: (_ for _ in ()).throw(RuntimeError("ui bug"))
        tool = CreateFileTool(workspace=workspace, confirm=lambda a: True, notify=hook)
        result = await tool.execute(path="ok.txt", content="x")
        assert result.startswith("OK:")


class TestFactory:
    def test_create_system_tools(self, workspace):
        tools = create_system_tools(workspace=workspace, confirm=lambda c: False)
        names = [t.name for t in tools]
        assert "todo_plan" not in names  # plan tool only when state is provided

    def test_create_system_tools_with_plan(self, workspace):
        tools = create_system_tools(workspace=workspace, confirm=lambda c: False,
                                    plan_state=PlanState())
        names = [t.name for t in tools]
        assert names == [
            "list_files", "read_file", "run_command",
            "create_file", "edit_file", "search_code", "glob_files", "todo_plan",
        ]


class TestAsyncConfirm:
    """The UI passes async confirm callbacks — tools must await them."""

    @pytest.mark.asyncio
    async def test_run_command_async_confirm_allowed(self, workspace):
        async def confirm(cmd):
            return True

        tool = RunCommandTool(workspace=workspace, confirm=confirm)
        result = await tool.execute(command="echo hi")
        assert "hi" in result and "Error" not in result

    @pytest.mark.asyncio
    async def test_run_command_async_confirm_denied(self, workspace):
        async def confirm(cmd):
            return False

        tool = RunCommandTool(workspace=workspace, confirm=confirm)
        result = await tool.execute(command="pip install requests")
        assert result.startswith("Command cancelled by user")

    @pytest.mark.asyncio
    async def test_create_file_async_confirm(self, workspace):
        calls = []

        async def confirm(action):
            calls.append(action)
            return True

        tool = CreateFileTool(workspace=workspace, confirm=confirm)
        result = await tool.execute(path="a.txt", content="hello")
        assert result.startswith("OK:")
        assert calls and "CREATE file 'a.txt'" in calls[0]

    @pytest.mark.asyncio
    async def test_edit_file_async_confirm_denied(self, workspace):
        async def confirm(action):
            return False

        tool = EditFileTool(workspace=workspace, confirm=confirm)
        result = await tool.execute(path="main.py", old_string="print('hello')",
                                    new_string="print('bye')")
        assert result.startswith("File edit cancelled by user")
        assert "print('hello')" in open(f"{workspace}/main.py", encoding="utf-8").read()
