# Claude Code — Complete Reference

A practical guide to everything you can do with Claude Code: the **tools** Claude uses
under the hood, the **slash commands** you type in chat, the **CLI flags** you pass at
launch, **keyboard shortcuts**, and the configuration/automation options available.

> Claude Code is Anthropic's official CLI agent for software engineering. It runs in your
> terminal (also desktop app, web at claude.ai/code, and IDE extensions for VS Code &
> JetBrains).

---

## 1. Slash Commands (typed in the chat prompt)

These are the `/commands` you type directly into the input box during a session.

### Session & conversation control
| Command | What it does |
|---|---|
| `/clear` | Wipe the current conversation history and start fresh (frees up context). |
| `/compact` | Summarize the conversation so far to reclaim context window space, keeping the gist. |
| `/resume` | Resume a previous conversation/session. |
| `/rewind` | Roll back the conversation (and optionally file changes) to an earlier checkpoint. |
| `/export` | Export the current conversation to a file or clipboard. |
| `/help` | Show built-in help and the list of available commands. |
| `/exit` or `/quit` | End the session. |

### Configuration & account
| Command | What it does |
|---|---|
| `/config` | Open the settings UI (theme, model, editor behavior, notifications, etc.). |
| `/model` | Switch the active model (e.g. Opus, Sonnet, Haiku). |
| `/fast` | Toggle **Fast mode** — Opus with faster output (does *not* downgrade to a smaller model). Available on Opus 4.8/4.7/4.6. |
| `/login` / `/logout` | Authenticate or sign out of your Anthropic account. |
| `/status` | Show account, model, version, and connection status. |
| `/cost` | Show token usage and cost for the current session. |
| `/permissions` | View and edit tool-permission rules (allow/deny lists). |
| `/vim` | Toggle vim-style editing in the input box. |
| `/terminal-setup` | Configure terminal key bindings (e.g. Shift+Enter for newline). |

### Project & context
| Command | What it does |
|---|---|
| `/init` | Generate a `CLAUDE.md` file documenting the codebase for future sessions. |
| `/memory` | Edit Claude's persistent memory files. |
| `/add-dir` | Add another directory to the workspace so Claude can read/edit it. |
| `/context` | Inspect what's currently loaded into the context window. |

### MCP & integrations
| Command | What it does |
|---|---|
| `/mcp` | Manage MCP (Model Context Protocol) servers — list, authenticate, enable/disable. |
| `/agents` | Manage custom subagents (create, edit, list agent definitions). |
| `/hooks` | View and configure hooks (automated commands on tool events). |

### Built-in workflows
| Command | What it does |
|---|---|
| `/review` | Review a pull request. |
| `/security-review` | Run a security review of pending changes on the current branch. |
| `/code-review` | Review the current diff for bugs & cleanups at a chosen effort level (`low`→`max`, or `ultra` for cloud multi-agent). Supports `--comment` (post inline PR comments) and `--fix` (apply findings). |
| `/pr-comments` | Fetch and show comments on a PR. |
| `/bug` | Report a bug about Claude Code to Anthropic. |

> Type `/` in the prompt to see the live, up-to-date list — available commands depend on
> your version, plugins, and project skills.

---

## 2. Skills (invocable specialized capabilities)

Skills are richer than commands — they load domain knowledge and multi-step procedures.
Invoke with `/<skill-name>`. The ones available in this environment:

| Skill | Purpose |
|---|---|
| `/deep-research` | Fan-out web research with adversarial fact-checking → a cited report. |
| `/code-review` | Diff review for correctness bugs and cleanups (see above). |
| `/simplify` | Apply reuse/simplification/efficiency cleanups (quality only, no bug-hunting). |
| `/verify` | Run the app and observe behavior to confirm a change actually works. |
| `/run` | Launch and drive the project's app (CLI/server/TUI/Electron/browser). |
| `/loop` | Run a prompt or command on a recurring interval (e.g. `/loop 5m /foo`). |
| `/schedule` | Create/manage scheduled cloud agents (cron-style routines). |
| `/init` | Initialize a `CLAUDE.md` for the codebase. |
| `/update-config` | Configure the harness via `settings.json` (hooks, permissions, env vars). |
| `/keybindings-help` | Customize keyboard shortcuts in `~/.claude/keybindings.json`. |
| `/fewer-permission-prompts` | Scan transcripts and build an allowlist to cut permission prompts. |
| `/claude-api` | Reference for the Claude API / Anthropic SDK (models, pricing, tool use). |
| `/security-review` | Full security review of pending branch changes. |
| `/review` | Review a pull request. |

---

## 3. CLI Flags & Commands (passed when launching `claude`)

Run these in your shell to start or control Claude Code.

### Starting a session
| Command / flag | What it does |
|---|---|
| `claude` | Start an interactive session in the current directory. |
| `claude "prompt"` | Start with an initial prompt. |
| `claude -p "prompt"` / `--print` | **Headless / non-interactive**: print the response and exit (great for scripts/pipes). |
| `claude -c` / `--continue` | Continue the most recent conversation. |
| `claude -r "<id>"` / `--resume` | Resume a specific session by ID. |
| `cat file \| claude -p "..."` | Pipe content in as context. |

### Model & behavior flags
| Flag | What it does |
|---|---|
| `--model <name>` | Choose the model (e.g. `claude-opus-4-8`, `claude-sonnet-4-6`). |
| `--permission-mode <mode>` | Set permission mode (e.g. `plan`, `acceptEdits`, `bypassPermissions`). |
| `--dangerously-skip-permissions` | Skip all permission prompts (use with caution; sandbox/CI only). |
| `--allowedTools` / `--disallowedTools` | Whitelist/blacklist specific tools for the run. |
| `--add-dir <path>` | Add extra working directories. |
| `--output-format <fmt>` | For `-p`: `text`, `json`, or `stream-json`. |
| `--verbose` | Verbose logging/output. |
| `--mcp-config <file>` | Load MCP servers from a config file. |

### Management subcommands
| Command | What it does |
|---|---|
| `claude config` | Get/set configuration values from the CLI. |
| `claude mcp` | Add, remove, and list MCP servers. |
| `claude update` | Update Claude Code to the latest version. |
| `claude doctor` | Diagnose installation/health issues. |
| `claude --version` | Print the installed version. |
| `claude --help` | Show all CLI flags and subcommands. |

---

## 4. Keyboard Shortcuts (in an interactive session)

| Key | Action |
|---|---|
| `Enter` | Submit the message. |
| `Shift+Enter` or `\` + `Enter` | Insert a newline (multi-line input). May need `/terminal-setup`. |
| `Esc` | Interrupt Claude / cancel the current action. |
| `Esc` `Esc` (double) | Edit your previous message / rewind. |
| `Ctrl+C` | Cancel current input or generation. |
| `Ctrl+D` | Exit the session. |
| `Ctrl+L` | Clear the terminal screen (keeps conversation). |
| `Ctrl+R` | Reverse-search command/prompt history. |
| `Up` / `Down` | Navigate input history. |
| `Tab` | Autocomplete files (`@`) and commands (`/`). |
| `#` (at line start) | Quick-add a note to memory / `CLAUDE.md`. |
| `!` (at line start) | **Bash mode** — run a shell command directly; output enters the conversation. |
| `@` | Mention/attach a file or directory as context. |
| `/` | Open the slash-command menu. |

> The `!` prefix is especially handy for interactive commands you must run yourself
> (e.g. `! gcloud auth login`) — the output lands directly in the session.

---

## 5. The Tools Claude Uses (under the hood)

These aren't typed by you — Claude calls them to do work. Knowing them helps you
understand and scope what Claude can do, and write effective permission rules.

### File & code tools
| Tool | What it does |
|---|---|
| **Read** | Read a file (text, images, PDFs, Jupyter notebooks). |
| **Write** | Create or overwrite a file. |
| **Edit** | Exact string-replacement edits in an existing file. |
| **NotebookEdit** | Edit cells in a Jupyter `.ipynb` notebook. |
| **Glob** | Fast filename pattern matching (`**/*.ts`). |
| **Grep** | Content search built on ripgrep (regex, file-type filters). |

### Execution & shell
| Tool | What it does |
|---|---|
| **Bash** | Run shell commands (POSIX/Git Bash). Supports background processes. |
| **PowerShell** | Run PowerShell commands on Windows. |
| **Monitor** | Watch/await a long-running or background condition. |

### Web & research
| Tool | What it does |
|---|---|
| **WebSearch** | Search the web. |
| **WebFetch** | Fetch and read a specific URL. |

### Agents & orchestration
| Tool | What it does |
|---|---|
| **Agent** | Launch a subagent (Explore, Plan, general-purpose, etc.) for complex/parallel work. |
| **Workflow** | Run a deterministic multi-agent orchestration script (fan-out, pipelines, verification). |
| **Task** tools | Create/list/get/stop/update background tasks (`TaskCreate`, `TaskList`, …). |
| **Cron** tools | Schedule recurring agents (`CronCreate`, `CronList`, `CronDelete`). |

### Planning & flow control
| Tool | What it does |
|---|---|
| **EnterPlanMode / ExitPlanMode** | Enter a read-only planning phase, then present a plan for approval. |
| **AskUserQuestion** | Ask you a structured multiple-choice question when a decision is genuinely yours. |
| **ScheduleWakeup** | Self-pace recurring loop work. |
| **EnterWorktree / ExitWorktree** | Work in an isolated git worktree. |
| **Skill** | Invoke one of the skills listed above. |
| **ToolSearch** | Load schemas for deferred/MCP tools on demand. |

### MCP (Model Context Protocol) tools
When you connect MCP servers, their tools become available too. Examples present here:
- **Atlassian** (Jira, Confluence, Compass)
- **Figma** (design ↔ code, FigJam, Code Connect)
- **Supabase** (Postgres, migrations, edge functions, branches)
- **Notion**, **Gmail**, **Google Calendar/Drive**, **Vercel**, **Hugging Face**,
  **Indeed**, **Dice**, **Similarweb**, **Three.js Viewer**.

Manage these with `/mcp`. Tool naming convention: `mcp__<server>__<tool>`.

---

## 6. Configuration Files & Automation

| File / mechanism | Purpose |
|---|---|
| `CLAUDE.md` | Project/user instructions auto-loaded into context. Created via `/init`. Overrides default behavior. |
| `~/.claude/settings.json` | Global settings: permissions, env vars, hooks, model defaults. |
| `.claude/settings.json` (per-project) | Project-scoped settings (committed to the repo). |
| `.claude/settings.local.json` | Local, un-committed per-project overrides. |
| `~/.claude/keybindings.json` | Custom keyboard shortcuts (`/keybindings-help`). |
| **Hooks** | Run shell commands automatically on tool events (PreToolUse, PostToolUse, Stop, etc.). For "always do X when Y" rules. |
| **Memory** (`~/.claude/.../memory/`) | Persistent facts across sessions, indexed by `MEMORY.md`. |
| **Permissions** | Allow/deny rules per tool (e.g. allow all `npm` commands, deny `rm -rf`). |

### Permission modes
| Mode | Behavior |
|---|---|
| `default` | Prompt for approval on sensitive actions. |
| `acceptEdits` | Auto-accept file edits, still prompt for other actions. |
| `plan` | Read-only — Claude plans but makes no changes until you approve. |
| `bypassPermissions` | No prompts at all (sandbox/CI use). |

---

## 7. Quick "How do I…" Cheat Sheet

| I want to… | Do this |
|---|---|
| Start clean | `/clear` |
| Free up context but keep the gist | `/compact` |
| Undo recent changes | `/rewind` |
| Switch models | `/model` or `--model` |
| Run faster without dropping to a smaller model | `/fast` |
| Plan before editing | Launch in `plan` mode (`--permission-mode plan`) |
| Document my repo for Claude | `/init` |
| Run a one-off shell command myself | type `! <command>` |
| Add another folder to the workspace | `/add-dir` or `--add-dir` |
| Connect an external tool | `/mcp` |
| Automate "whenever X happens, do Y" | Configure a **hook** (`/hooks` or `/update-config`) |
| Schedule recurring agent runs | `/schedule` |
| Review my changes for bugs | `/code-review` |
| Cut down permission prompts | `/fewer-permission-prompts` |
| Use Claude in a script | `claude -p "prompt" --output-format json` |

---

*Generated as a reference. The exact set of commands/flags can vary by Claude Code
version — run `/help` or `claude --help` for the authoritative, current list, and see
the official docs at https://docs.claude.com/en/docs/claude-code.*
