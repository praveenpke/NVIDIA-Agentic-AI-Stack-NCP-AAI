# NCP-AAI Study Hub

Interactive study site for the **NVIDIA Certified Professional — Agentic AI (NCP-AAI)** exam:
deep-dive notes, 3D visual explainers, runnable code companions, and a briefing on what the
agent-building field is talking about in 2025–26.

**Live site:** deployed on Vercel (see the repository's About link).

## What's inside

| Path | Contents |
|---|---|
| `visuals/` | The website — an index hub plus seven self-contained interactive pages (Three.js 3D scenes, infographics, quizzes, code companions). Light/dark mode with warm theming. |
| `notes/` | The companion markdown deep-dives, one per domain — same content, readable as plain text. |
| `NCP-AAI-Study-Guide.md` | Top-level study guide and exam-domain map. |

### The pages

- **Builder's Pulse** — the 10 conversations the agent-building world is having right now
  (context engineering, the evals debate, 12-factor agents, MCP/A2A, multi-agent skepticism,
  SLM economics…), mapped to exam domains, with a full annotated reading list.
- **Domain 3 — Evaluation and Tuning** (13%) — LLM-as-a-judge, RAGAS, trajectory metrics, the tuning ladder.
- **Domain 4 — Deployment and Scaling** (13%) — NIM microservices, Triton, TensorRT-LLM, Kubernetes autoscaling.
- **Domain 7 — NVIDIA Platform Implementation** (7%) — the full stack: CUDA → TensorRT-LLM → Triton → NIM → NeMo → Blueprints.
- **Domain 8 — Run, Monitor, and Maintain** (5%) — observability, drift, SLOs, rollback patterns.
- **Domain 9 — Safety, Ethics, and Compliance** (5%) — NeMo Guardrails, prompt injection, OWASP LLM Top 10, garak.
- **Domain 10 — Human-AI Interaction and Oversight** (5%) — HITL gates, interrupt/resume, confidence-based deferral.

Each domain page ends with a **Code Companion** (minimal, runnable, verified snippets using the
NVIDIA stack — NIM endpoints, NeMo Guardrails, LangGraph integration, Langfuse/OTel tracing) and
a **practitioner voices** section with sourced 2025–26 takes from the field.

## Run locally

It's a fully static site — no build step, no dependencies:

```bash
# any static server works, e.g.
npx serve visuals
# or just open visuals/index.html in a browser
```

## Features

- **Light/dark mode** — toggle in the top-right corner; defaults to your system preference,
  remembered in localStorage. Warm palette in both modes (`visuals/theme.css`).
- **Self-contained pages** — every page is a single HTML file with inline CSS/JS;
  Three.js scenes load from CDN.
- **Mobile responsive** — fluid grids, scrollable code blocks, adaptive nav.

## Disclaimer

Personal study material, not affiliated with or endorsed by NVIDIA. Exam-domain names and
weights follow NVIDIA's public NCP-AAI exam guide. Practitioner quotes are attributed and
linked to their original sources.
