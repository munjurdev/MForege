# MForege — AI Agent CLI

[![PyPI version](https://img.shields.io/pypi/v/mforege.svg?v=2)](https://pypi.org/project/mforege/)
[![Python](https://img.shields.io/pypi/pyversions/mforege.svg?v=2)](https://pypi.org/project/mforege/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A command-line AI agent with a transparent, IDE-style interface: live tool
activity, colored diffs on every file change, mission plans, and session
persistence. Supports OpenAI, Ollama (free & local), and any OpenAI-compatible
API (Groq, Together, etc.).

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

### First run — guided setup

No config files needed. The first `mforege` launches a 30-second wizard:

```
Where should MForege get its brain?
  [1] Groq    free cloud API — fastest option (recommended)
  [2] Ollama  free, 100% local (needs https://ollama.com installed)
  [3] OpenAI  paid API
  [4] Custom  any OpenAI-compatible endpoint
```

Pick one, paste your key (hidden input), done — config is saved to
`~/.mforege/.env` and works from **every** folder, forever. A `./.env` in
the current folder can override it for one project. Change your mind later
with `mforege --setup`.

For Groq: grab a free key at https://console.groq.com (no credit card).

## Features

### Agent core

- **Full tool loop** — when the model calls a tool, the result is sent back so
  the model can use it in its answer (streaming and non-streaming); up to 10
  rounds per message, configurable via `AgentConfig.max_tool_rounds`
- **Streaming output** — responses appear token by token, with a
  `thinking Ns...` timer while the model works
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
- **Mutating actions require approval** — file writes/edits show a colored
  diff before you confirm; commands show the exact line to be run
- **Fail-closed** — without a confirmation handler, mutating operations refuse
  rather than run unconfirmed
- **Workspace-confined** — path tools only operate inside the folder you
  choose at launch; `../` and symlink escapes are rejected
- **Timeouts & caps** — 30s command timeout, tool output capped so the context
  window never balloons

### Chat-style interface (prompt_toolkit)

- **Slash-command menu** — type `/` and a live two-column popup appears
  (command + description); keep typing to filter, ↑/↓ to select, Enter to
  accept, Esc to close — exactly like a modern IDE command palette
- **Structured welcome** — a clean rounded box shows the agent name, model,
  and workspace on startup
- **Bottom input box** — the input stays pinned at the bottom in a bordered
  frame like a chat app; the transcript scrolls above it
- **Scrolling** — PgUp/PgDn (10 lines), Ctrl+↑/↓ or Shift+↑/↓ (1 line),
  and the **mouse wheel** all scroll the transcript; `End` returns to the
  live tail. While scrolled up the status bar shows `↑ N lines · End=bottom`
- **Scrollbar** — a thin scrollbar on the transcript's right edge shows
  the viewport position (auto-hidden when everything fits on screen)
- **Live activity stream** — every tool call prints as `· tool(args) ✓ (0.3s)`
- **Thinking timer** — `thinking Ns...` runs until the first token arrives
- **Colored diffs** — `+` green / `-` red, shown before approval *and* in the
  transcript after the edit lands
- **Plan statusline** — `[Plan 2/5] next: write the test` after each reply
- **Status bar** — model, workspace, and shortcut hints always visible
- **Keys** — Enter sends, Shift+Enter adds a newline (Windows Terminal,
  cmd, Unix terminals). In VS Code's terminal use **Alt+Enter** for a
  newline — VS Code never transmits Shift+Enter to any program.
  Ctrl+T expands the AI's thinking, Ctrl+C quits
- **Slash commands** — `/help`, `/plan`, `/tools`, `/new`, `/history`,
  `/sessions`, `/resume`, `/bash`, `/byok`, `/interview`, `/diagnostics`,
  `/review`, `/copy`, `/export`, `/theme:toggle`, `/reasoning`,
  `/vscode-hint`, `/clear`, `/exit`

### Memory

- **Conversation memory** — history per session, `clear` resets it
- **Auto-condense** — when history nears the context window, old turns
  fold into a compact summary injected before the recent ones, so long
  missions never lose their start
- **Context meter** — live `ctx ~12.3K (2%)` in the status bar
- **Session resume** — every conversation auto-saves; on the next launch
  MForege automatically continues your last chat (like an agent session
  that survives a cutoff). `/sessions` lists past chats, `/resume [id]`
  restores a specific one

### Resilience

- Request timeouts + automatic retries with backoff for transient failures
- Friendly errors for missing keys or Ollama not running
- UTF-8 output on Windows (emoji-safe), clean async shutdown

## Setup

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

### Backends

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

### Web search (optional)

1. Get a free API key at https://dashboard.exa.ai
2. Add to `.env`: `EXA_API_KEY=your_key`
3. Restart the CLI

### Workspace

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

`.env` resolution: if the current folder has a `.env`, it wins; otherwise
MForege falls back to its own `.env` — so API keys are found no matter where
you launch it. Workspace defaults to the current directory.

## CLI commands

| Command  | Action                                     |
|----------|--------------------------------------------|
| `/help`  | Show the command menu                      |
| `/plan`  | Show the current mission plan              |
| `/tools` | List registered tools                      |
| `/new`   | Clear the conversation and start a new chat |
| `/history` | Browse past conversations (same as `/sessions`) |
| `/sessions` | List past conversations                     |
| `/resume` | Resume the most recent chat (or `/resume <id>`) |
| `/bash <cmd>` | Run a shell command with the agent's safety guards |
| `/byok`  | Show where to configure your API key / model |
| `/interview` | Guided Q&A to spec a task before building |
| `/diagnostics` | Version, model, context usage, sessions |
| `/review` | Review changes made this conversation      |
| `/copy`  | Copy the conversation to the clipboard     |
| `/export` | Write the conversation to a .json file     |
| `/theme:toggle` | Toggle light/dark mode               |
| `/reasoning` | Thinking effort: low / high / max      |
| `/vscode-hint` | Fix Shift+Enter for VS Code's terminal |
| `/clear` | Reset conversation (current session)       |
| `/exit`  | Quit (plain `exit` also works)             |

## Example session

```
Nice to meet you! 👋 I'm MForege.

You: create utils.py with a greet function, then test it
Assistant: thinking 2s...
  · todo_plan(action=set) ✓ (0.0s)
  · create_file(path=utils.py) ✓ (0.0s)
[?] MForege wants to: CREATE file 'utils.py' (95 chars)
  @@ -0,0 +1,3 @@
  +def greet(name):
  +    return f"Hello, {name}!"
    Allow? [y/N]: y
  ✓ Edit applied:
  @@ -0,0 +1,3 @@
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
env/Scripts/python -m pytest
```

202 tests covering conversation memory (including auto-condensation), the
tool registry, the safe calculator, the web search tool, session
persistence (save, list, resume), the terminal UI (scrolling, thinking
blocks, live plan blocks, Windows Shift+Enter compatibility), system tools
(path confinement, command classification, confirmation flow — sync and
async, diffs, notifications), discovery tools (search/glob/read windows),
the todo plan, and the agent's tool-call loop — all with a mocked LLM, no
network needed.
