"""
System / Agentic Tools
======================
Give the agent real execution capability on the local machine:

- list_files  — list directory contents
- read_file   — read a text file
- run_command — run a shell command (guarded, optional confirmation)

Safety model (important — the LLM decides when to call these, so the tools
themselves must be defensive):

1. Path confinement: list_files/read_file only operate inside a workspace
   root (default: the directory the CLI was launched from). Symlink and
   ".." escapes are resolved and rejected.
2. Command safety: run_command checks a blocklist of destructive patterns
   (rm -rf, format, del /s, git push --force, etc.), enforces a timeout,
   caps output size, and requires explicit confirmation for commands that
   modify the filesystem (except inside a small allowlist of safe ones).
3. Confirmation hook: each tool accepts an optional `confirm` callback
   (e.g., asks the user y/n in the CLI). If it returns False, the tool
   refuses. Mutating actions always ask; read-only ones don't.
"""

import asyncio
import fnmatch
import inspect
import os
import re
import shlex
import subprocess
from typing import Any, Callable, ClassVar, List, Optional

from pydantic import PrivateAttr

from app.agent.tools import Tool

# ── limits ──────────────────────────────────────────────────────────────────
MAX_OUTPUT_CHARS = 4000          # cap tool output fed back to the model
COMMAND_TIMEOUT = 30.0           # seconds before a command is killed
MAX_LIST_ENTRIES = 200


async def _maybe_await(value: Any) -> Any:
    """Await `value` if it is awaitable (async confirm callbacks), else return it."""
    if inspect.isawaitable(value):
        return await value
    return value


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (output truncated, {len(text) - limit} more chars)"


def make_unified_diff(path: str, old_text: str, new_text: str, max_lines: int = 30) -> str:
    """
    Compact unified diff for display in confirmations (like an IDE).
    Additions prefixed with '+', deletions with '-'. Capped at max_lines.
    """
    import difflib

    old_lines = (old_text or "").splitlines()
    new_lines = (new_text or "").splitlines()
    diff = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"{path} (old)", tofile=f"{path} (new)",
        lineterm="",
    ))
    if not diff:
        return "(no changes)"
    # Skip the ---/+++ header lines for compactness
    body = [d for d in diff if not d.startswith("---") and not d.startswith("+++")]
    shown = body[:max_lines]
    out = "\n".join(shown)
    if len(body) > max_lines:
        out += f"\n... ({len(body) - max_lines} more diff lines)"
    return out


def _resolve_inside_root(root: str, relative: str) -> Optional[str]:
    """
    Resolve `relative` against `root` and ensure it stays inside root.
    Returns the absolute resolved path, or None if it escapes.
    """
    root_abs = os.path.abspath(root)
    target = os.path.abspath(os.path.join(root_abs, relative or "."))
    # os.path.commonpath is the robust check (handles "..", symlinks, case)
    try:
        if os.path.commonpath([root_abs, target]) != root_abs:
            return None
    except ValueError:  # different drives on Windows
        return None
    return target


# ── command safety ──────────────────────────────────────────────────────────
# Patterns that are always rejected, no questions asked.
BLOCKED_PATTERNS = [
    "rm -rf", "rm -fr", "rm -r ", "rmdir /s", "del /s", "del /f",
    "format ", "mkfs", "diskpart",
    "git push --force", "git push -f", "git reset --hard",
    "shutdown", "reboot",
    "curl ... | sh", "curl | sh", "wget | sh",
    "sudo rm", "sudo mkfs",
    ":(){:|:&};:",  # fork bomb
]

# Commands considered safe to run without confirmation (read-only).
SAFE_PREFIXES = (
    "dir", "ls", "echo", "type ", "cat ", "head ", "tail ", "wc ",
    "git status", "git log", "git diff", "git branch",
    "python --version", "python3 --version", "pip list", "pip --version",
    "node --version", "npm --version", "where ", "which ", "whoami",
    "pwd", "cd", "tree",
)


def _classify_command(command: str) -> str:
    """
    Returns one of:
      "blocked"    — always refuse
      "safe"       — read-only, no confirmation needed
      "confirm"    — mutating, needs user confirmation
    """
    cmd = " ".join(command.lower().split())

    for pattern in BLOCKED_PATTERNS:
        if pattern in cmd:
            return "blocked"

    first_word = cmd.split(" ", 1)[0] if cmd else ""
    if any(cmd.startswith(p.rstrip()) or cmd.startswith(p) for p in SAFE_PREFIXES):
        return "safe"

    # python -m pytest / python script.py etc. — confirm (can do anything)
    if first_word in ("python", "python3", "pip", "git", "npm"):
        return "confirm"

    return "confirm"


# ── tools ───────────────────────────────────────────────────────────────────
class NotifyHook:
    """Shared notify callback holder: tools call it after successful mutations."""

    def __init__(self):
        self.callback: Optional[Callable[[str, str], None]] = None

    def notify(self, event: str, detail: str) -> None:
        if self.callback is None:
            return
        try:
            self.callback(event, detail)
        except Exception:
            pass  # display must never break the tool


class ListFilesTool(Tool):
    """List files and folders in a directory (workspace-confined)"""

    _workspace: str = PrivateAttr(default=".")
    _confirm: Optional[Callable] = PrivateAttr(default=None)

    def __init__(self, workspace: str = ".", confirm: Optional[Callable] = None, **data):
        super().__init__(
            name="list_files",
            description=(
                "List files and folders in a directory of the user's project. "
                "Use before reading files to discover what exists. "
                "Paths are relative to the project root."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to list, relative to project root (default '.')"
                    }
                },
                "required": [],
            },
            execute=self._execute,
            **data,
        )
        self._workspace = os.path.abspath(workspace)
        self._confirm = confirm

    async def _execute(self, path: str = ".") -> str:
        resolved = _resolve_inside_root(self._workspace, path)
        if resolved is None:
            return f"Error: path '{path}' is outside the project workspace"

        if not os.path.exists(resolved):
            return f"Error: directory '{path}' does not exist"
        if not os.path.isdir(resolved):
            return f"Error: '{path}' is a file, not a directory"

        try:
            entries = sorted(os.listdir(resolved))
        except OSError as e:
            return f"Error: could not list '{path}': {e}"

        if not entries:
            return f"(empty directory: {path})"

        lines: List[str] = []
        hidden = 0
        for name in entries:
            if len(lines) >= MAX_LIST_ENTRIES:
                hidden = len(entries) - len(lines)
                break
            full = os.path.join(resolved, name)
            marker = "/" if os.path.isdir(full) else ""
            size = "" if marker else f"  ({os.path.getsize(full)} bytes)"
            lines.append(f"{name}{marker}{size}")

        header = f"Contents of {path}:" if path != "." else "Contents of project root:"
        result = header + "\n" + "\n".join(lines)
        if hidden:
            result += f"\n... and {hidden} more entries"
        return _truncate(result)


class ReadFileTool(Tool):
    """Read a text file (workspace-confined, size-capped)"""

    _workspace: str = PrivateAttr(default=".")
    _confirm: Optional[Callable] = PrivateAttr(default=None)
    _max_lines: int = PrivateAttr(default=200)

    def __init__(self, workspace: str = ".", confirm: Optional[Callable] = None, **data):
        super().__init__(
            name="read_file",
            description=(
                "Read the contents of a text file in the user's project. "
                "Optionally read a window: offset (1-based starting line) and "
                "limit (max lines). Use list_files/search_code to find files "
                "first. Paths are relative to the project root."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File to read, relative to project root"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "1-based line number to start reading from (default 1)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read (default 200, max 500)"
                    }
                },
                "required": ["path"],
            },
            execute=self._execute,
            **data,
        )
        self._workspace = os.path.abspath(workspace)
        self._confirm = confirm

    async def _execute(self, path: str, offset: int = 1, limit: int = 200) -> str:
        resolved = _resolve_inside_root(self._workspace, path)
        if resolved is None:
            return f"Error: path '{path}' is outside the project workspace"

        if not os.path.exists(resolved):
            return f"Error: file '{path}' does not exist (use list_files to check)"
        if os.path.isdir(resolved):
            return f"Error: '{path}' is a directory (use list_files on it instead)"

        # Skip obvious binaries
        try:
            with open(resolved, "rb") as f:
                chunk = f.read(1024)
            if b"\x00" in chunk:
                return f"Error: '{path}' looks like a binary file; only text files can be read"
        except OSError as e:
            return f"Error: cannot open '{path}': {e}"

        try:
            offset = max(1, int(offset or 1))
            limit = min(max(1, int(limit or self._max_lines)), 500)
        except (TypeError, ValueError):
            return "Error: offset and limit must be integers"

        try:
            with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                # Skip lines before the window
                for _ in range(offset - 1):
                    if not f.readline():
                        break
                lines = [f.readline() for _ in range(limit + 1)]
        except OSError as e:
            return f"Error: cannot read '{path}': {e}"

        more = len(lines) > limit
        shown = lines[:limit]
        text = "".join(shown).rstrip("\n")
        numbered = "\n".join(
            f"{offset + i:4d}| {line}" for i, line in enumerate(text.split("\n"))
        )
        result = f"Contents of {path} (lines {offset}-{offset + len(text.splitlines()) - 1 if text else offset - 1}):\n{numbered}"
        if more:
            result += f"\n... (showing {limit} lines from offset {offset}; call again with offset={offset + limit} to continue)"
        return _truncate(result)


class RunCommandTool(Tool):
    """
    Run a shell command in the workspace, with a safety net:

    - blocklist of destructive patterns -> refused outright
    - read-only commands (ls, git status, ...) run without asking
    - anything else requires the `confirm` callback to approve it
    - timeout (default 30s), output capped
    """

    _workspace: str = PrivateAttr(default=".")
    _confirm: Optional[Callable] = PrivateAttr(default=None)
    _timeout: float = PrivateAttr(default=COMMAND_TIMEOUT)
    _notify: Optional[NotifyHook] = PrivateAttr(default=None)

    def __init__(self, workspace: str = ".", confirm: Optional[Callable] = None,
                 timeout: float = COMMAND_TIMEOUT, notify: Optional[NotifyHook] = None, **data):
        super().__init__(
            name="run_command",
            description=(
                "Run a shell command in the user's project directory "
                "(e.g., 'python --version', 'pip install django', 'git status'). "
                "Destructive commands are blocked; others may ask the user for "
                "confirmation. Output is returned to you, truncated."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run"
                    }
                },
                "required": ["command"],
            },
            execute=self._execute,
            **data,
        )
        self._workspace = os.path.abspath(workspace)
        self._confirm = confirm
        self._timeout = timeout
        self._notify = notify

    async def _execute(self, command: str) -> str:
        command = (command or "").strip()
        if not command:
            return "Error: empty command"

        verdict = _classify_command(command)
        if verdict == "blocked":
            return f"Error: command blocked for safety: '{command}'"

        if verdict == "confirm":
            if self._confirm is None:
                # Fail closed: mutating commands must never run unconfirmed
                return (
                    f"Error: command requires user confirmation but no confirmation "
                    f"handler is configured; refused: '{command}'"
                )
            allowed = await _maybe_await(self._confirm(command))
            if not allowed:
                return f"Command cancelled by user: '{command}'"

        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                command,
                shell=True,
                cwd=self._workspace,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {self._timeout:.0f}s: '{command}'"
        except OSError as e:
            return f"Error: failed to run command: {e}"

        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        status = f"[exit code: {proc.returncode}]"
        if self._notify:
            self._notify.notify("cmd", f"$ {command} → {status}")
        parts = [status]
        if out:
            parts.append(out)
        if err:
            parts.append(f"stderr:\n{err}")
        if not out and not err:
            parts.append("(no output)")
        return _truncate("\n".join(parts))


class CreateFileTool(Tool):
    """
    Create a new text file (or overwrite an existing one) in the workspace.

    Mutating operation: requires the `confirm` callback (fail-closed).
    Creating new files is approved by describing the action; overwriting an
    existing file says so explicitly in the confirmation message.
    """

    _workspace: str = PrivateAttr(default=".")
    _confirm: Optional[Callable] = PrivateAttr(default=None)
    _max_content_chars: int = PrivateAttr(default=100_000)
    _notify: Optional[NotifyHook] = PrivateAttr(default=None)

    def __init__(self, workspace: str = ".", confirm: Optional[Callable] = None,
                 notify: Optional[NotifyHook] = None, **data):
        super().__init__(
            name="create_file",
            description=(
                "Create a new text file with the given content in the user's "
                "project (parent folders are created automatically). The user "
                "is asked for confirmation before anything is written. "
                "Paths are relative to the project root."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File to create, relative to project root"
                    },
                    "content": {
                        "type": "string",
                        "description": "Full text content to write to the file"
                    }
                },
                "required": ["path", "content"],
            },
            execute=self._execute,
            **data,
        )
        self._workspace = os.path.abspath(workspace)
        self._confirm = confirm
        self._notify = notify

    async def _execute(self, path: str, content: str) -> str:
        resolved = _resolve_inside_root(self._workspace, path)
        if resolved is None:
            return f"Error: path '{path}' is outside the project workspace"
        if not path.strip():
            return "Error: empty path"
        if os.path.isdir(resolved):
            return f"Error: '{path}' is a directory, not a file"

        content = content or ""
        if len(content) > self._max_content_chars:
            return f"Error: content too large ({len(content)} chars, max {self._max_content_chars})"

        overwrite = os.path.exists(resolved)
        old_text = ""
        if overwrite:
            try:
                with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                    old_text = f.read()
            except OSError:
                old_text = ""
        diff_text = make_unified_diff(path, old_text if overwrite else "", content)
        verb = "OVERWRITE" if overwrite else "CREATE"
        action = f"{verb} file '{path}' ({len(content)} chars)\n{diff_text}"

        if self._confirm is None:
            return (
                f"Error: writing files requires user confirmation but no "
                f"confirmation handler is configured; refused: {action}"
            )
        if not await _maybe_await(self._confirm(action)):
            return f"File write cancelled by user: '{path}'"

        try:
            os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)
            with open(resolved, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)
        except OSError as e:
            return f"Error: could not write '{path}': {e}"

        verb = "overwritten" if overwrite else "created"
        if self._notify:
            self._notify.notify("diff", make_unified_diff(path, old_text if overwrite else "", content))
        return f"OK: {verb} '{path}' ({len(content)} chars, {content.count(chr(10)) + 1} lines)"


class EditFileTool(Tool):
    """
    Edit an existing text file by replacing an exact snippet.

    Safety properties:
    - requires the `confirm` callback (fail-closed)
    - the file must already exist (use create_file for new files)
    - `old_string` must appear EXACTLY ONCE in the file, otherwise nothing
      is written — this prevents accidental edits at the wrong place
    """

    _workspace: str = PrivateAttr(default=".")
    _confirm: Optional[Callable] = PrivateAttr(default=None)
    _notify: Optional[NotifyHook] = PrivateAttr(default=None)

    def __init__(self, workspace: str = ".", confirm: Optional[Callable] = None,
                 notify: Optional[NotifyHook] = None, **data):
        super().__init__(
            name="edit_file",
            description=(
                "Edit an existing text file: replace an exact snippet "
                "(old_string) with new text (new_string). old_string must "
                "match exactly and appear only once in the file. The user is "
                "asked for confirmation before anything is written."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File to edit, relative to project root"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact text to replace (must be unique in the file)"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text"
                    }
                },
                "required": ["path", "old_string", "new_string"],
            },
            execute=self._execute,
            **data,
        )
        self._workspace = os.path.abspath(workspace)
        self._confirm = confirm
        self._notify = notify

    async def _execute(self, path: str, old_string: str, new_string: str) -> str:
        resolved = _resolve_inside_root(self._workspace, path)
        if resolved is None:
            return f"Error: path '{path}' is outside the project workspace"

        if not os.path.exists(resolved):
            return f"Error: file '{path}' does not exist (create it first with create_file)"
        if os.path.isdir(resolved):
            return f"Error: '{path}' is a directory"

        old_string = old_string or ""
        new_string = new_string if new_string is not None else ""
        if not old_string:
            return "Error: old_string must not be empty"

        try:
            with open(resolved, "r", encoding="utf-8", errors="strict") as f:
                original = f.read()
        except (OSError, UnicodeDecodeError) as e:
            return f"Error: cannot read '{path}': {e}"

        count = original.count(old_string)
        if count == 0:
            return (
                f"Error: old_string not found in '{path}'. "
                f"Use read_file to check the exact content first."
            )
        if count > 1:
            return (
                f"Error: old_string appears {count} times in '{path}'; "
                f"it must be unique. Include more surrounding text to make it unique."
            )

        updated = original.replace(old_string, new_string, 1)
        summary_old = old_string[:60] + ("..." if len(old_string) > 60 else "")
        summary_new = new_string[:60] + ("..." if len(new_string) > 60 else "")
        diff_text = make_unified_diff(path, original, updated)
        action = (
            f"EDIT file '{path}': replace {summary_old!r} with {summary_new!r}\n"
            f"{diff_text}"
        )

        if self._confirm is None:
            return (
                f"Error: editing files requires user confirmation but no "
                f"confirmation handler is configured; refused: {action}"
            )
        if not await _maybe_await(self._confirm(action)):
            return f"File edit cancelled by user: '{path}'"

        try:
            with open(resolved, "w", encoding="utf-8", newline="") as f:
                f.write(updated)
        except OSError as e:
            return f"Error: could not write '{path}': {e}"

        if self._notify:
            self._notify.notify("diff", make_unified_diff(path, original, updated))
        return f"OK: edited '{path}' (replaced 1 occurrence)"


# Directories never worth searching inside (noise / huge trees)
SKIP_DIRS = {
    "__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules", "venv", "env", ".venv", "dist", "build", "site-packages",
    ".idea", ".vscode", "htmlcov", ".tox", "data",
}

# Extensions treated as binary (skip content search)
BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".tar", ".gz",
    ".7z", ".rar", ".exe", ".dll", ".so", ".dylib", ".pyc", ".pyo",
    ".class", ".jar", ".woff", ".woff2", ".ttf", ".eot", ".mp3", ".mp4",
    ".sqlite", ".db", ".bin", ".woff", ".lock",
}


def _iter_project_files(root: str):
    """Yield relative file paths under root, skipping noise dirs and binaries."""
    root_abs = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root_abs):
        # Prune noise directories in-place
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in BINARY_EXTS:
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root_abs)
            yield full, rel.replace(os.sep, "/")


class SearchCodeTool(Tool):
    """
    Regex search across the workspace (like a mini-ripgrep).

    Read-only: no confirmation needed. Returns file:line: match lines,
    capped so the model isn't flooded. Invalid regex -> helpful error.
    """

    _workspace: str = PrivateAttr(default=".")
    _confirm: Optional[Callable] = PrivateAttr(default=None)

    def __init__(self, workspace: str = ".", confirm: Optional[Callable] = None, **data):
        super().__init__(
            name="search_code",
            description=(
                "Search file contents across the project with a regex pattern "
                "(like grep/ripgrep). Returns matching lines as "
                "'file:line: text'. Best first step to find functions, "
                "classes, TODOs, or where an error string comes from."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression to search for (e.g., 'def create_', 'ALLOWED_HOSTS', 'TODO')"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum matches to return (default 40, max 200)"
                    }
                },
                "required": ["pattern"],
            },
            execute=self._execute,
            **data,
        )
        self._workspace = os.path.abspath(workspace)
        self._confirm = confirm

    async def _execute(self, pattern: str, max_results: int = 40) -> str:
        pattern = (pattern or "").strip()
        if not pattern:
            return "Error: empty pattern"
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"Error: invalid regex pattern: {e}"

        max_results = max(1, min(int(max_results or 40), 200))
        matches: List[str] = []
        files_with_hits = set()
        files_scanned = 0

        for full, rel in _iter_project_files(self._workspace):
            files_scanned += 1
            try:
                with open(full, "r", encoding="utf-8", errors="ignore") as f:
                    for lineno, line in enumerate(f, 1):
                        if rx.search(line):
                            matches.append(f"{rel}:{lineno}: {line.rstrip()[:200]}")
                            files_with_hits.add(rel)
                            if len(matches) >= max_results:
                                break
            except OSError:
                continue
            if len(matches) >= max_results:
                break

        if not matches:
            return f"No matches for pattern {pattern!r} (scanned {files_scanned} files)"

        header = (
            f"{len(matches)} match(es) for {pattern!r} "
            f"in {len(files_with_hits)} file(s):"
        )
        result = header + "\n" + "\n".join(matches)
        if len(matches) >= max_results:
            result += f"\n... (stopped at {max_results} matches — narrow the pattern to see more)"
        return _truncate(result)


class GlobFilesTool(Tool):
    """
    Find files by name pattern, recursively (e.g., **/*.py, test_*.py).
    Read-only: no confirmation needed. Supports *, **, ?, [abc].
    """

    _workspace: str = PrivateAttr(default=".")
    _confirm: Optional[Callable] = PrivateAttr(default=None)

    def __init__(self, workspace: str = ".", confirm: Optional[Callable] = None, **data):
        super().__init__(
            name="glob_files",
            description=(
                "Find files by filename pattern anywhere in the project, "
                "recursively (e.g., '**/*.py', 'test_*.py', 'settings.*'). "
                "Returns relative paths sorted by modification time "
                "(newest first)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern, e.g. '**/*.py' (all Python files), '*.md', 'app/**/*.html'"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum paths to return (default 50, max 300)"
                    }
                },
                "required": ["pattern"],
            },
            execute=self._execute,
            **data,
        )
        self._workspace = os.path.abspath(workspace)
        self._confirm = confirm

    async def _execute(self, pattern: str, max_results: int = 50) -> str:
        pattern = (pattern or "").strip().replace("\\", "/")
        if not pattern:
            return "Error: empty pattern"
        max_results = max(1, min(int(max_results or 50), 300))

        # Build a matcher supporting ** (crosses directories) and * (doesn't)
        rx_parts = []
        for seg in pattern.split("/"):
            if seg == "**":
                rx_parts.append("(?:.*/)?")
            else:
                rx_parts.append(fnmatch.translate(seg).replace("\\Z", "") + "/")
        rx_text = "^" + "".join(rx_parts)
        try:
            rx = re.compile(rx_text)
        except re.error as e:
            return f"Error: invalid pattern: {e}"

        hits = []
        for full, rel in _iter_project_files(self._workspace):
            candidate = rel + "/"
            if rx.match(candidate) or rx.match("/" + candidate):
                try:
                    mtime = os.path.getmtime(full)
                except OSError:
                    mtime = 0
                hits.append((mtime, rel))

        # Also handle simple patterns like "*.py" meaning any depth
        if not hits and "/" not in pattern and "**" not in pattern:
            base_rx = re.compile(fnmatch.translate(pattern) + "$", re.IGNORECASE)
            for full, rel in _iter_project_files(self._workspace):
                if base_rx.match(os.path.basename(rel)):
                    try:
                        mtime = os.path.getmtime(full)
                    except OSError:
                        mtime = 0
                    hits.append((mtime, rel))

        hits.sort(reverse=True)  # newest first
        if not hits:
            return f"No files matching pattern {pattern!r}"

        shown = hits[:max_results]
        lines = [rel for _, rel in shown]
        result = f"{len(hits)} file(s) matching {pattern!r}:\n" + "\n".join(lines)
        if len(hits) > max_results:
            result += f"\n... and {len(hits) - max_results} more"
        return _truncate(result)


class TodoPlanTool(Tool):
    """
    Visible mission plan: lets the model maintain a step-by-step todo list
    during multi-step tasks, and shows progress to the user in the CLI.

    The plan lives in a shared PlanState object (also injected into the
    system prompt each turn, so the model never loses the plot).

    Actions:
      set    — replace the whole plan: items=["step 1", "step 2", ...]
      update — set completion: index (1-based) + done true/false
      add    — append a new step
      clear  — wipe the plan (mission complete)
    """

    MAX_STEPS: ClassVar[int] = 20
    MAX_STEP_CHARS: ClassVar[int] = 120
    _state: "PlanState" = PrivateAttr(default=None)

    def __init__(self, state, **data):
        # state: PlanState instance (shared with the CLI/statusline + prompt)
        super().__init__(
            name="todo_plan",
            description=(
                "Maintain a visible step-by-step plan for multi-step tasks. "
                "Actions: 'set' (items=[...]) to create/replace the plan, "
                "'update' (index, done) to mark a step done/not done, 'add' "
                "(item) to append a step, 'clear' when finished. The user "
                "sees the plan live — keep it accurate."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["set", "update", "add", "clear"],
                        "description": "What to do with the plan"
                    },
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "For action=set: the ordered list of steps"
                    },
                    "index": {
                        "type": "integer",
                        "description": "For action=update: 1-based step number"
                    },
                    "done": {
                        "type": "boolean",
                        "description": "For action=update: true when the step is complete"
                    },
                    "item": {
                        "type": "string",
                        "description": "For action=add: the new step text"
                    }
                },
                "required": ["action"],
            },
            execute=self._execute,
            **data,
        )
        self._state = state

    async def _execute(self, action: str, items=None, index=None, done=None, item=None) -> str:
        action = (action or "").strip().lower()

        if action == "set":
            if not isinstance(items, list) or not items:
                return "Error: action 'set' needs a non-empty 'items' list"
            steps = []
            for it in items[: self.MAX_STEPS]:
                text = str(it).strip()[: self.MAX_STEP_CHARS]
                if text:
                    steps.append(text)
            if not steps:
                return "Error: no valid steps in 'items'"
            self._state.set(steps)
            return f"Plan set with {len(steps)} steps."

        if action == "update":
            try:
                idx = int(index)
            except (TypeError, ValueError):
                return "Error: action 'update' needs an integer 'index'"
            changed = self._state.mark(idx, bool(done))
            if not changed:
                return f"Error: no step #{idx} (plan has {len(self._state.steps)} steps)"
            return f"Step #{idx} marked {'done' if done else 'not done'}."

        if action == "add":
            text = str(item or "").strip()[: self.MAX_STEP_CHARS]
            if not text:
                return "Error: action 'add' needs non-empty 'item' text"
            if len(self._state.steps) >= self.MAX_STEPS:
                return f"Error: plan is full ({self.MAX_STEPS} steps max)"
            self._state.add(text)
            return f"Step added: {text!r}"

        if action == "clear":
            self._state.clear()
            return "Plan cleared. Mission complete!"

        return f"Error: unknown action '{action}' (use set/update/add/clear)"


class PlanState:
    """Shared plan storage: used by the tool, the CLI statusline, and prompt injection."""

    def __init__(self):
        self.steps: List[dict] = []  # [{"text": str, "done": bool}]

    def set(self, steps: List[str]) -> None:
        self.steps = [{"text": s, "done": False} for s in steps]

    def add(self, text: str) -> None:
        self.steps.append({"text": text, "done": False})

    def mark(self, index: int, done: bool) -> bool:
        if 1 <= index <= len(self.steps):
            self.steps[index - 1]["done"] = done
            return True
        return False

    def clear(self) -> None:
        self.steps = []

    def render(self) -> str:
        """Human-readable plan, e.g. for the CLI statusline."""
        if not self.steps:
            return ""
        lines = []
        for i, s in enumerate(self.steps, 1):
            box = "x" if s["done"] else " "
            lines.append(f"[{box}] {i}. {s['text']}")
        done_count = sum(1 for s in self.steps if s["done"])
        return f"\n".join(lines) + f"\n({done_count}/{len(self.steps)} done)"

    def progress_line(self) -> str:
        """One-line summary for the CLI statusline."""
        if not self.steps:
            return ""
        done_count = sum(1 for s in self.steps if s["done"])
        current = next((s["text"] for s in self.steps if not s["done"]), "all done")
        return f"[Plan {done_count}/{len(self.steps)}] next: {current[:60]}"


def create_system_tools(workspace: str = ".", confirm: Optional[Callable] = None,
                        plan_state: Optional["PlanState"] = None,
                        notify: Optional[NotifyHook] = None) -> list:
    """Convenience: build the standard agentic toolset for a workspace."""
    tools = [
        ListFilesTool(workspace=workspace, confirm=confirm),
        ReadFileTool(workspace=workspace, confirm=confirm),
        RunCommandTool(workspace=workspace, confirm=confirm, notify=notify),
        CreateFileTool(workspace=workspace, confirm=confirm, notify=notify),
        EditFileTool(workspace=workspace, confirm=confirm, notify=notify),
        SearchCodeTool(workspace=workspace, confirm=confirm),
        GlobFilesTool(workspace=workspace, confirm=confirm),
    ]
    if plan_state is not None:
        tools.append(TodoPlanTool(state=plan_state))
    return tools
