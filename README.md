# MForege — AI Agent CLI

[![PyPI version](https://img.shields.io/pypi/v/mforege.svg?v=2)](https://pypi.org/project/mforege/)
[![Python](https://img.shields.io/pypi/pyversions/mforege.svg?v=2)](https://pypi.org/project/mforege/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A command-line AI agent with a transparent, IDE-style interface: live tool
activity, colored diffs on every file change, mission plans, and session
persistence. Supports OpenAI, Ollama (free & local), OpenRouter (one free key
→ many free models), and any OpenAI-compatible API (Groq, Together, etc.).

---

## Table of contents

- [Install](#install)
- [First run — guided setup](#first-run--guided-setup)
- [Always up to date (auto-update)](#always-up-to-date-auto-update)
- [Features](#features)
  - [Agent core](#agent-core)
  - [Tools](#tools)
  - [Safety](#safety)
  - [Chat-style interface](#chat-style-interface-prompt_toolkit)
  - [Memory](#memory)
  - [Resilience](#resilience)
- [Setup (from source)](#setup-from-source)
- [Backends](#backends)
- [Switching models live](#switching-models-live)
- [Web search (optional)](#web-search-optional)
- [Workspace](#workspace)
- [Run](#run)
- [Configuration resolution](#configuration-resolution)
- [CLI commands](#cli-commands)
- [Example session](#example-session)
- [Adding your own tool](#adding-your-own-tool)
- [Tests](#tests)
- [License](#license)

---

## Install

```bash
pip install mforege
```

Or with [pipx](https://pipx.pypa.io) (recommended for CLIs — isolated env,
command on PATH everywhere):

```bash
pipx install mforege
```

Or straight from source:

```bash
pip install git+https://github.com/munjurdev/MForege.git
```

## First run — guided setup

No config files needed. The first `mforege` launches a 30-second wizard:

```
Where should MForege get its brain?
  [1] Groq        free cloud API — fastest (recommended)
  [2] OpenRouter  free — ONE key unlocks 20+ free models
  [3] Ollama      free, 100% local (needs https://ollama.com)
  [4] OpenAI      paid API (uses OPENAI_API_KEY)
  [5] Custom      any OpenAI-compatible endpoint
```

Pick one, paste your key (hidden input), done — the key is verified with a
tiny test call before anything is saved, and the config lands in
`~/.mforege/.env` where it works from **every** folder, forever. A `./.env`
in the current folder can override it for one project. Change your mind
later with `mforege --setup`.

For Groq: grab a free key at https://console.groq.com (no credit card).

## Always up to date (auto-update)

Launching `mforege` checks PyPI (24-hour cached, silent when offline — never
delays startup more than ~3s once a day). If a newer release exists:

```
[i] New version available: mforege 0.1.16 (you have 0.1.15)
[i] Updating mforege to the latest version — restarting…
```

The app exits to release its file lock, a background helper runs
`pip install --upgrade mforege` (falling back to `--user` when the system
site-packages is not writable), and mforege reopens itself in the same
terminal — you end up inside the newest version without doing anything.
If the upgrade fails for any reason, the installed version launches as usual.

Skip the check entirely with `mforege --no-update` or by setting
`MFOREGE_NO_UPDATE=1` in the environment (handy for CI and pinning).

---

## Features

### Agent core

- **Full tool loop** — when the model calls a tool, the result is sent back so
  the model can use it in its answer (streaming and non-streaming); up to 10
  rounds per message, configurable via `AgentConfig.max_tool_rounds`
- **Streaming output** — responses appear token by token, with a live
  `⏳ thinking Ns (Esc=stop)` timer in the status bar while the model works
- **Markdown rendering** — assistant replies render formatted, live:
  `**bold**` becomes bold, `` `code` `` gets its own color, fenced code
  blocks, headers, `•` bullets and `▌` quotes all appear styled while the
  model streams — raw markdown syntax is never shown (agent-panel style)
- **Esc to stop** — press Esc while the agent works and it stops cleanly:
  the in-flight request unwinds, any partial answer is kept, and the next
  message works normally (Esc with text in the input clears the line instead)
- **Personality + response control** — friendly, witty, emoji-moderate; short
  answers for casual chat, clarifying questions for ambiguous or underspecified
  requests (never dumps giant tutorials or acts on invented details)

### Tools

| Tool | Purpose |
|------|---------|
| `list_files` | List directories (workspace-confined) |
| `read_file` | Read text files with line numbers; `offset`/`limit` windows for big files |
| `search_code` | Regex search across the project (`file:line: match`), noise dirs skipped |
| `glob_files` | Find files by pattern, recursively (`**/*.py`), newest first |
| `run_command` | Shell commands with a 3-tier safety model |
| `create_file` | New files (parent folders auto-created) |
| `edit_file` | Exact, unique-snippet replacement |
| `todo_plan` | Visible step-by-step mission plan |
| `calculator` | Safe AST-based math (no `eval`) |
| `get_current_time` | Current date/time |
| `web_search` | Live web search via Exa (needs `EXA_API_KEY`) |

### Safety

- **Destructive commands blocked** outright (`rm -rf`, `git push --force`,
  `format`, ...)
- **Chained commands are never "safe"** — a read-only first word followed by
  `&&`, `;`, `|`, `>` etc. always requires confirmation (`echo hi && rm -rf /`
  is blocked outright, not waved through)
- **Mutating actions require approval** — file writes/edits show a colored
  diff before you confirm; commands show the exact line to be run
- **Fail-closed** — without a confirmation handler, mutating operations refuse
  rather than run unconfirmed
- **Workspace-confined** — path tools only operate inside the folder you
  choose at launch; `../` and symlink escapes are rejected
- **Timeouts & caps** — 30s command timeout, 30s web-search timeout, tool
  output capped so the context window never balloons

### Chat-style interface (prompt_toolkit)

- **Slash-command menu** — type `/` and a live two-column popup appears
  (command + description); keep typing to filter, ↑/↓ to select, Enter to
  accept, Esc to close — exactly like a modern IDE command palette
- **Structured welcome** — a clean boxed panel shows the agent name, model,
  and workspace on startup
- **Bottom input box** — the input stays pinned at the bottom in a bordered
  frame like a chat app; the transcript scrolls above it. It grows with your
  text (2–5 rows) and every status element sits *above* it, never below
- **Scrolling** — PgUp/PgDn (10 lines), Ctrl+↑/↓ or Shift+↑/↓ (1 line),
  and the **mouse wheel** all scroll the transcript; `End` returns to the
  live tail. While scrolled up the status bar shows `↑ N lines · End=bottom`
  and new output never slides the viewport away from what you're reading
- **Scrollbar** — a thin scrollbar on the transcript's right edge shows
  the viewport position (auto-hidden when everything fits on screen)
- **Live activity stream** — every tool call prints as `· tool(args) ✓ (0.3s)`
- **Thinking block** — reasoning tokens stream into an italic `✻ Thinking…`
  block that collapses when the answer starts; **Ctrl+T** re-expands it
- **Colored diffs** — `+` green / `-` red, shown before approval *and* in the
  transcript after the edit lands
- **Plan block** — a live `TODOS` panel with ✓ done / ✎ current / · pending,
  plus a `[Plan 2/5] next: write the test` line after each reply
- **Status bar** — model, session clock, context meter, and the phase timer
- **Keys** — Enter sends, Shift+Enter adds a newline (Windows Terminal,
  cmd, Unix terminals). In VS Code's terminal use **Alt+Enter** for a
  newline — VS Code never transmits Shift+Enter to any program.
  Esc stops the agent (or clears the input), Ctrl+T expands the AI's
  thinking, Ctrl+C quits
- **Themes** — `/theme:toggle` switches between dark and light palettes
- **Message queueing** — messages sent while the agent works are queued with
  a visible `✓ Queued` note and run in order after the current turn

### Memory

- **Conversation memory** — history per session, `clear` resets it
- **Auto-condense** — when history nears the context window, old turns
  fold into a compact summary injected before the recent ones, so long
  missions never lose their start; on a "request too large" error the
  agent condenses further and shrinks its output budget automatically
- **Context meter** — live `ctx ~12.3K (2%)` in the status bar (counts the
  system prompt and tool schemas, not just history)
- **Session resume** — every conversation auto-saves (atomically, crash-safe)
  to `~/.mforege/sessions/` — outside the package, so `pip` upgrades never
  wipe your chat history. Launches start FRESH (no auto-continue);
  `/sessions` lists past chats and `/resume [id]` restores one explicitly

### Resilience

- Request timeouts + automatic retries with backoff for transient failures
- Friendly errors for missing keys, rejected keys, rate limits, or Ollama
  not running — plus a one-line hint pointing at `/byok` / `/model`
- Adaptive recovery from "request too large" (413): condense context, shrink
  `max_tokens`, wait out the rate-limit window, then retry
- UTF-8 output on Windows (emoji-safe), clean async shutdown

---

## Setup (from source)

```bash
python -m venv env
env/Scripts/activate          # Windows (bash: source env/Scripts/activate)
pip install -r requirements.txt
pip install -e .              # installs the `mforege` command
```

After `pip install -e .`, MForege works from **any** terminal — including VS
Code's — with the venv activated:

```bash
mforege                                    # in any folder (uses MForege's own .env)
mforege --workspace "C:\path\to\project"   # work inside another project
```

## Backends

**OpenAI** — put your key in `.env`:

```
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

**Ollama (free, local)** — install from https://ollama.com, then:

```bash
ollama pull llama3
mforege --backend ollama --model llama3
```

**Custom (e.g., Groq)**:

```
API_KEY=gsk_...
BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.1-8b-instant
```

**OpenRouter (one free key → many `:free` models)**:

```
API_KEY=sk-or-...
BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=deepseek/deepseek-r1:free
```

## Switching models live

MForege ships a curated free-model catalog (Groq, OpenRouter `:free`, Ollama,
OpenAI) defined in one place — `app/models.py`. Add or remove a model by
editing one `Model(...)` line there; every menu picks it up automatically.

Switch mid-conversation — history is kept, and the choice persists:

```
/model            # list models for your provider
/model 2          # pick by menu number
/model kimi       # pick by short name
/model openai/gpt-oss-120b         # or paste any model id
```

The switch applies to your next message. `mforege --setup` also offers the
catalog (and OpenRouter) instead of a bare model prompt.

## Web search (optional)

1. Get a free API key at https://dashboard.exa.ai
2. Add `EXA_API_KEY=your_key` to `./.env` (project) **or**
   `~/.mforege/.env` (global) — both work
3. Restart the CLI

Without a key everything else works; the agent just says search is off.

## Workspace

MForege can only read/write inside one folder — the one you choose:

```bash
mforege --workspace "C:\path\to\your\project"
```

The header shows the active workspace. To work in a different folder, restart
with a different `--workspace` (or set `WORKSPACE` in `.env`).

## Run

```bash
python main.py                  # from the project folder
mforege                         # from ANY folder (after pip install -e .)
mforege --backend ollama --model llama3
mforege --workspace "C:\path\to\your\project"
```

## Configuration resolution

Settings resolve **per key**, stopping at the first source that actually
defines it:

1. real environment variables (shell/session)
2. `./.env` — current folder (per-project override)
3. `~/.mforege/.env` — global config written by the setup wizard
4. `<MForege repo>/.env` — developer installs
5. built-in default

Because the check is per key, a project's own `.env` (a Django app with
`DEBUG`/`SECRET_KEY`, for example) never hides your global MForege config.
Running `mforege` from inside such a project simply finds your `LLM_BACKEND`,
`API_KEY`, etc. in `~/.mforege/.env`. To give one project its own brain, add
the keys to that folder's `.env` — local keys always win for that project.
Files are re-read on lookup, so edits to `.env` (and `mforege --setup`)
apply without restarting.

## CLI commands

| Command | Action |
|---------|--------|
| `/help` | Show the command menu |
| `/plan` | Show the current mission plan |
| `/tools` | List registered tools |
| `/new` | Clear the conversation and start a new chat |
| `/history` | Browse past conversations (same as `/sessions`) |
| `/sessions` | List past conversations |
| `/resume` | Resume the most recent chat (or `/resume <id>`) |
| `/bash <cmd>` | Run a shell command with the agent's safety guards |
| `/byok` | Show where to configure your API key / model |
| `/model` | Switch model live (`/model 2`, `/model kimi`, or an id) |
| `/reasoning` | Thinking effort: low / high / max |
| `/interview` | Guided Q&A to spec a task before building |
| `/diagnostics` | Version, model, context usage, sessions |
| `/review` | Review changes made this conversation |
| `/copy` | Copy the conversation to the clipboard |
| `/export` | Write the conversation to a .json file |
| `/theme:toggle` | Toggle light/dark mode |
| `/queue` | Message queueing info |
| `/feedback` | Share feedback / open an issue |
| `/vscode-hint` | Fix Shift+Enter for VS Code's terminal |
| `/clear` | Reset conversation (current session) |
| `/exit` | Quit (plain `exit` also works) |

## Example session

*(plain mode — the full-screen UI shows the same flow with colored diffs and
live blocks)*

```
Nice to meet you! 👋 I'm MForege.

You: create utils.py with a greet function, then test it
MForege:
  · todo_plan(action=set) ✓ (0.0s)
  · create_file(path=utils.py) ✓ (0.0s)
[?] MForege wants to:
CREATE file 'utils.py' (95 chars)
--- utils.py (old)
+++ utils.py (new)
+def greet(name):
+    return f"Hello, {name}!"
    Allow? [y/N]: y
  ✓ Applied:
+def greet(name):
...
[Plan 2/4] next: run the tests
```

## Adding your own tool

```python
from app.agent.tools import create_tool

async def get_weather(city: str):
    return f"Sunny in {city}, 25C"

agent.register_tools(create_tool("get_weather", "Get weather for a city", get_weather))
```

## Tests

```bash
python -m pytest                # or: env/Scripts/python -m pytest
```

336 tests covering the self-update flow (version compare, cache, loop guard,
detached helper), the update banner, conversation memory (including
auto-condensation), the tool registry, the safe calculator, the web search
tool, session persistence (save, list, resume), the terminal UI (scrolling,
thinking blocks, Esc-stop survival, live plan blocks, Windows Shift+Enter
compatibility), system tools (path confinement, command classification
including chained-command safety, confirmation flow — sync and async, diffs,
notifications), discovery tools (search/glob/read windows), the todo plan,
the agent's tool-call loop, and 413 adaptive recovery — all with a mocked
LLM, no network needed.

## License

[MIT](LICENSE) — free to use, modify, and ship.
