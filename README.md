# NVIDIA Agentic AI Stack (NCP-AAI)

An interactive study hub for the **NVIDIA Certified Professional — Agentic AI (NCP-AAI)** exam.
It covers the complete exam blueprint with deep-dive notes, interactive 3D explainers, runnable
code companions on the NVIDIA stack, a practice exam, a glossary, and a hands-on setup track.

Content is verified against the live NVIDIA documentation (NeMo Agent Toolkit **v1.7**, current
NIM / NeMo / Nemotron 3 model lineup) and the broader agentic-AI landscape as of mid-2026.

> **Live site:** **[nvidia-agentic-ai-stack-ncp-aai.vercel.app](https://nvidia-agentic-ai-stack-ncp-aai.vercel.app/)**
> — or run it locally (double-click `visuals/index.html`).

---

## Exam coverage

All ten exam domains are covered as a paired **interactive page** (`visuals/`) + **markdown deep-dive** (`notes/`).
Domain weights follow NVIDIA's public NCP-AAI exam guide.

| # | Domain | Weight | Highlights |
|---|--------|:---:|---|
| 1 | Agent Architecture & Design | 15% | ReAct/Reflexion/Plan-Execute, NAT agent types, multi-agent topologies, A2A/MCP, NVIDIA Blueprints |
| 2 | Agent Development | 15% | Prompting, function calling, NAT functions/groups/middleware/auth, LangGraph, reliability patterns |
| 3 | Evaluation & Tuning | 13% | LLM-as-judge, RAGAS triad, retrieval metrics, the tuning ladder, NAT eval/profiler |
| 4 | Deployment & Scaling | 13% | NIM, Triton, TensorRT-LLM, TP vs PP, autoscaling, NAT deployment servers, Dynamo |
| 5 | Cognition, Planning & Memory | 10% | CoT/ToT/reflexion, planning, memory types, NAT reasoning/ReWOO/test-time-compute, object stores |
| 6 | Knowledge Integration & Data | 10% | RAG pipeline, chunking, NeMo Retriever, Milvus+cuVS, hybrid search, reranking, GraphRAG |
| 7 | NVIDIA Platform Implementation | 7% | CUDA → TensorRT-LLM → Triton → NIM → NeMo → Blueprints; Nemotron tiers; AI Enterprise |
| 8 | Run, Monitor & Maintain | 5% | Tracing/observability, drift, SLOs & burn-rate, health probes, rollback, Data Flywheel |
| 9 | Safety, Ethics & Compliance | 5% | NeMo Guardrails, NemoGuard NIMs, OWASP LLM/Agentic Top 10, garak, EU AI Act, NIST AI RMF |
| 10 | Human-AI Interaction & Oversight | 5% | HITL/HOTL, approval gates, interrupt/resume, confidence deferral, hard controls, OpenShell |

Each domain page ends with a **Code Companion** — minimal, runnable, verified snippets on the
NVIDIA stack (NIM endpoints, NeMo Guardrails, NeMo Retriever, LangGraph integration, Langfuse/OTel
tracing) — plus a **practitioner voices** section with sourced 2025–26 takes from the field.

## Beyond the domains

| Page | What it is |
|------|------------|
| **Builder's Pulse** | The 10 conversations the agent-building world is having now (context engineering, the evals debate, 12-factor agents, MCP/A2A, multi-agent skepticism, SLM economics), mapped to exam domains, with an annotated reading list. |
| **Mock Exam** | 50 weighted practice questions across all 10 domains, each with the answer, distractor analysis, and a domain tag — plus exam strategy and the two capstone scenarios. |
| **Glossary** | Every NCP-AAI term defined in plain language, tagged by domain, filterable. |
| **Hands-On Labs & Platform Setup** | A runnable onboarding track — API key → first NIM call → first NAT workflow → first Guardrails run → self-hosted NIM → Blueprint — with an install matrix and a troubleshooting matrix. |

## Suggested study path

1. **Start at the hub** (`visuals/index.html`) and skim **Builder's Pulse** for the lay of the land.
2. **Work the domains in weight order** (D1 → D10). On each page: explore the 3D scene to build the
   mental model, read the infographic sections, then attempt the scenario quiz cards.
3. **Type out the Code Companions** against [build.nvidia.com](https://build.nvidia.com) (free hosted
   NIM endpoints) — reading is not enough; the **Hands-On Labs** page wires these together.
4. **Take the Mock Exam** under timed conditions; revisit weak domains and re-read their notes.
5. Use the **Glossary** to close any term-level gaps.

## How it's organized

```
.
├── index.html                  # root redirect → visuals/index.html (for static hosts)
├── visuals/                    # the website: index hub + 16 self-contained pages
│   ├── index.html              # study hub
│   ├── domain-01..10-*.html    # the ten exam domains (3D scenes + infographics + quizzes)
│   ├── domain-00-builders-pulse.html
│   ├── glossary.html  ·  mock-exam.html  ·  hands-on-labs.html
│   ├── study-guide.html  ·  resources.html
│   └── theme.css / theme.js    # shared theme + injected nav menu (single source of truth)
├── notes/                      # one markdown deep-dive per page (same content, plain text)
└── NCP-AAI-Study-Guide.md      # top-level study guide & exam-domain map
```

## Features

- **Interactive 3D explainers** — each domain has a Three.js scene whose *motion teaches the concept*
  (e.g. in-flight batching, guardrail rails blocking attacks, an HITL approval gate). Loads from CDN.
- **Black & white theme** with light/dark mode — toggle in the **bottom-right** corner; defaults to
  your system preference and is remembered in `localStorage` (`visuals/theme.css`).
- **Site menu** — the ☰ button (bottom-right) opens a navigation drawer to every page.
- **Self-contained pages** — every page is a single HTML file with inline CSS/JS; no build step.
- **Mobile responsive** — fluid grids, scrollable code blocks, adaptive nav.

## Run locally

It's a fully static site — no build step, no dependencies:

```bash
# any static server works, e.g.
npx serve .
# …then open http://localhost:3000  (root redirects into the hub)
# or simply double-click visuals/index.html
```

## Deploy

Any static host works (Vercel, Netlify, GitHub Pages, Cloudflare Pages). Import the repo with the
default settings — **Framework: Other**, no build command, root directory `./`. The root
`index.html` redirects visitors into `visuals/`, so the hub loads at `/`.

## Disclaimer

Personal study material — not affiliated with, authorized, or endorsed by NVIDIA. Exam-domain names
and weights follow NVIDIA's public NCP-AAI exam guide; technical details are verified against public
NVIDIA documentation but always confirm specifics against the official source. Practitioner quotes
are attributed and linked to their original sources.
