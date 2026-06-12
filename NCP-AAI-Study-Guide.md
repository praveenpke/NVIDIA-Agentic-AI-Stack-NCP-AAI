# NVIDIA Certified Professional: Agentic AI (NCP-AAI) — Study Guide

> Compiled June 2026 from the official NVIDIA certification page and community exam guides.
> Official page: https://www.nvidia.com/en-us/learn/certification/agentic-ai-professional/

> **🎯 Deep-dive materials for weak domains (3, 4, 7, 8, 9, 10):**
> - **Visual hub (start here):** [visuals/index.html](visuals/index.html) — interactive 3D Three.js scenes + 2D infographics per domain
> - **Full notes:** [notes/](notes/) — one detailed file per domain with mental models, decision tables, exam traps, and scenario drills

---

## 1. Exam Overview

| Item | Detail |
|---|---|
| **Certification** | NVIDIA-Certified Professional: Agentic AI (NCP-AAI) |
| **Level** | Professional (intermediate) |
| **Format** | Online, remotely proctored |
| **Questions** | 60–70 (multiple choice + scenario-based) |
| **Duration** | 120 minutes (~2 min/question) |
| **Price** | $200 USD |
| **Language** | English |
| **Validity** | 2 years |
| **Recommended experience** | 1–2 years in AI/ML roles with hands-on, production-level agentic AI projects |

**What it validates:** the ability to architect, develop, deploy, and govern advanced agentic AI solutions — multi-agent interaction, distributed reasoning, scalability, and ethical safeguards.

---

## 2. Exam Domains & Weights (Official Blueprint)

| # | Domain | Weight |
|---|---|---|
| 1 | Agent Architecture and Design | **15%** |
| 2 | Agent Development | **15%** |
| 3 | Evaluation and Tuning | **13%** |
| 4 | Deployment and Scaling | **13%** |
| 5 | Cognition, Planning, and Memory | **10%** |
| 6 | Knowledge Integration and Data Handling | **10%** |
| 7 | NVIDIA Platform Implementation | **7%** |
| 8 | Run, Monitor, and Maintain | **5%** |
| 9 | Safety, Ethics, and Compliance | **5%** |
| 10 | Human-AI Interaction and Oversight | **5%** |

> 56% of the exam is the top four domains — prioritize architecture, development, evaluation, and deployment.

---

## 3. Domain Deep Dive

### Domain 1 — Agent Architecture and Design (15%)
- **Agent patterns:** ReAct (reason + act), Reflexion (self-critique loops), Plan-and-Execute, AutoGPT-style autonomous loops, router/supervisor patterns.
- **Single-agent vs multi-agent design:** when to decompose into specialist agents vs one generalist agent.
- **Multi-agent topologies:** supervisor/worker (hierarchical), peer-to-peer, blackboard/shared-state, pipeline (sequential hand-off).
- **Orchestration frameworks:** LangGraph (stateful graphs, checkpointing), LangChain, AutoGen / AG2 (conversational multi-agent), CrewAI (role-based crews), NVIDIA NeMo Agent Toolkit.
- **Inter-agent communication standards:** Agent2Agent (A2A) protocol, Model Context Protocol (MCP) for tool/context integration.
- **Design trade-offs:** latency vs accuracy, cost vs capability, autonomy vs controllability, statefulness vs statelessness.
- **Tool design:** tool schemas, function calling, idempotency, error contracts.

### Domain 2 — Agent Development (15%)
- **Prompt engineering:** system prompts, few-shot examples, chain-of-thought, structured output (JSON schema), prompt templates and versioning.
- **Tool/function calling:** defining tools, parallel tool calls, handling tool errors and retries, MCP servers as first-class tool layer.
- **Structured workflows:** state machines and graphs (LangGraph nodes/edges/conditional edges), human-in-the-loop interrupts, checkpoints.
- **Multimodal agents:** vision-language models, document understanding, audio I/O.
- **Reliability engineering:** retries with backoff, timeouts, fallback models, output validation (Pydantic/JSON schema), guarded generation.
- **Streaming** responses and partial results; async/concurrent agent execution.
- **Rapid prototyping:** NVIDIA AI Workbench, build.nvidia.com hosted endpoints, NIM API catalog.

### Domain 3 — Evaluation and Tuning (13%)
- **Evaluation types:** offline eval (golden datasets), online eval (A/B tests), LLM-as-a-judge, human evaluation, pairwise comparison.
- **Agent-specific metrics:** task completion rate, tool-call accuracy, trajectory/step efficiency, hallucination rate, latency, cost per task.
- **RAG evaluation:** retrieval metrics (recall@k, precision@k, MRR, NDCG), generation metrics (faithfulness/groundedness, answer relevance, context relevance) — RAGAS-style triad.
- **Tuning approaches:** prompt optimization, RAG tuning (chunking/embedding/reranker choices), fine-tuning (SFT, LoRA/PEFT), preference alignment (DPO/RLHF) — and when each is appropriate.
- **NVIDIA NeMo Evaluator** microservice; benchmark suites (MMLU, HELM-style, domain benchmarks).
- **Regression testing** for prompts and agents; eval-driven development.

### Domain 4 — Deployment and Scaling (13%)
- **NVIDIA NIM microservices:** containerized, optimized inference endpoints (OpenAI-compatible APIs); deploying on Kubernetes/cloud/on-prem.
- **Triton Inference Server:** multi-framework serving, dynamic batching, concurrent model execution, model ensembles.
- **TensorRT / TensorRT-LLM:** engine optimization, quantization (FP8/INT8/INT4, AWQ), KV-cache management, in-flight (continuous) batching.
- **Scaling patterns:** horizontal autoscaling, GPU utilization optimization, multi-GPU/multi-node inference (tensor/pipeline parallelism), load balancing.
- **Serving trade-offs:** throughput vs latency, batch size tuning, speculative decoding.
- **Infrastructure:** Kubernetes + GPU operator, Helm charts for NIM, cloud GPU instances (DGX Cloud), edge deployment (Jetson / Nemotron Nano models).
- **CI/CD for agents:** containerization, model versioning, blue-green/canary rollout.

### Domain 5 — Cognition, Planning, and Memory (10%)
- **Reasoning frameworks:** chain-of-thought, tree-of-thoughts, self-consistency, reasoning models (test-time compute).
- **Planning:** task decomposition, hierarchical planning, re-planning on failure, plan validation.
- **Memory types:** short-term (conversation buffer/window), long-term (vector store memory), episodic vs semantic vs procedural memory, entity memory.
- **Context management:** context window budgeting, summarization/compaction of history, scratchpads, retrieval-augmented memory.
- **State management** across sessions; checkpointing and resumability.

### Domain 6 — Knowledge Integration and Data Handling (10%)
- **RAG pipelines end-to-end:** ingestion → chunking → embedding → indexing → retrieval → reranking → generation.
- **Chunking strategies:** fixed-size, recursive, semantic, document-structure-aware; chunk overlap trade-offs.
- **Embeddings & vector databases:** NVIDIA NeMo Retriever (NIM embedding/reranking models), Milvus, FAISS, pgvector; ANN indexes (HNSW, IVF).
- **Hybrid search:** dense + sparse (BM25) with fusion (RRF); metadata filtering.
- **Reranking:** cross-encoder rerankers to boost precision.
- **Data handling:** PII redaction, document parsing (PDF/tables/images — NV-Ingest / NeMo Retriever extraction), data freshness and re-indexing.
- **GraphRAG and structured knowledge** (knowledge graphs) for multi-hop questions.

### Domain 7 — NVIDIA Platform Implementation (7%)
- **NIM (NVIDIA Inference Microservices):** model catalog at build.nvidia.com, self-hosted vs hosted endpoints, OpenAI-compatible API.
- **NeMo framework family:** NeMo Customizer (fine-tuning), NeMo Evaluator, NeMo Guardrails, NeMo Retriever, NeMo Agent Toolkit (formerly AgentIQ).
- **Nemotron model family:** open NVIDIA models for agentic use (e.g., Nemotron Super/Ultra; Nano variants for edge).
- **NVIDIA AI Enterprise:** production support, security, and lifecycle for the stack.
- **NVIDIA Blueprints:** reference agentic workflows (e.g., AI virtual assistant, multimodal PDF RAG).
- **AI Workbench / DGX Cloud** for development and training environments.

### Domain 8 — Run, Monitor, and Maintain (5%)
- **Observability:** tracing agent trajectories (OpenTelemetry, LangSmith/LangFuse-style tooling), logging tool calls and intermediate steps.
- **Monitoring:** latency, token usage/cost, error rates, GPU utilization (DCGM metrics), throughput.
- **Drift detection:** data drift, model behavior drift, eval-in-production.
- **Alerting and incident response;** dashboards; SLOs for agent services.
- **Maintenance:** model updates, prompt version rollouts, index refreshes, dependency upgrades.

### Domain 9 — Safety, Ethics, and Compliance (5%)
- **NeMo Guardrails:** input/output rails, topic control, jailbreak detection, content safety models (e.g., Llama Guard-class, Aegis), Colang policies.
- **Threats:** prompt injection (direct/indirect), data exfiltration via tools, excessive agency, insecure output handling (OWASP LLM Top 10).
- **Red-teaming** agents; adversarial testing (e.g., Garak-style scanners).
- **Governance & compliance:** auditability, data residency, EU AI Act / NIST AI RMF awareness, model cards, usage policies.
- **Responsible AI:** bias evaluation, transparency, content provenance.

### Domain 10 — Human-AI Interaction and Oversight (5%)
- **Human-in-the-loop (HITL):** approval gates for high-risk actions, interrupt/resume workflows, escalation paths.
- **Confidence thresholds** and selective deferral to humans.
- **Transparency:** explaining agent decisions, citing sources, showing tool-use traces to users.
- **UX for agents:** streaming feedback, progress indication, undo/confirmation for irreversible actions.
- **Oversight design:** least-privilege tool access, action allowlists, audit logs.

---

## 4. Key NVIDIA Technologies — Quick Reference

| Technology | What it is | Exam relevance |
|---|---|---|
| **NIM** | Containerized, optimized inference microservices with OpenAI-compatible APIs | Deployment questions; self-hosted vs API |
| **NeMo Agent Toolkit** | Framework-agnostic library to build/profile/optimize agent workflows | Agent orchestration, profiling |
| **NeMo Guardrails** | Programmable rails for safety, topic control, jailbreak detection | Safety domain |
| **NeMo Retriever** | Embedding, reranking, and extraction NIMs for RAG | Knowledge integration |
| **NeMo Customizer / Evaluator** | Fine-tuning (LoRA/SFT) and evaluation microservices | Tuning & evaluation |
| **Triton Inference Server** | High-performance multi-framework model serving | Scaling, batching |
| **TensorRT-LLM** | LLM inference optimization: quantization, KV cache, in-flight batching | Performance questions |
| **Nemotron models** | NVIDIA's open model family (incl. Nano for edge) | Model selection scenarios |
| **NVIDIA Blueprints** | Reference architectures for agentic workflows | Architecture scenarios |
| **AI Workbench** | Local/remote dev environment manager | Prototyping |
| **MCP / A2A** | Tool-integration and agent-to-agent protocols | Interoperability questions |

---

## 5. Recommended Official Training (NVIDIA DLI)

1. **Building RAG Agents With LLMs** — 8 h, $90 (self-paced/instructor-led)
2. **Evaluating RAG and Semantic Search Systems** — 3 h, $30 (self-paced)
3. **Building Agentic AI Applications With LLMs** — 8 h, $90 (self-paced/instructor-led)
4. **Adding New Knowledge to LLMs** — 8 h, $500 (instructor-led)
5. **Introduction to Deploying RAG Pipelines for Production at Scale** — 8 h, $90

---

## 6. 8-Week Study Plan

| Week | Focus | Activities |
|---|---|---|
| 1 | Foundations + Architecture (D1) | Agent patterns (ReAct, Reflexion, plan-execute), multi-agent topologies; read LangGraph & NeMo Agent Toolkit docs |
| 2 | Agent Development (D2) | Build a tool-using agent with function calling + MCP; structured outputs; error handling; DLI "Building Agentic AI Applications" |
| 3 | Cognition & Memory (D5) | Implement short/long-term memory, summarization, planning loops in a project |
| 4 | RAG & Knowledge (D6) | Build a RAG pipeline with NeMo Retriever or open embeddings + Milvus; hybrid search + reranking; DLI "Building RAG Agents" |
| 5 | Evaluation & Tuning (D3) | RAGAS-style evals, LLM-as-judge, golden datasets; when to prompt-tune vs LoRA vs RAG; DLI "Evaluating RAG" |
| 6 | Deployment & Scaling (D4 + D7) | Deploy a NIM locally (Docker) or use build.nvidia.com; learn Triton concepts, TensorRT-LLM optimizations, K8s autoscaling |
| 7 | Ops, Safety, HITL (D8–D10) | Add tracing/observability; implement NeMo Guardrails; add HITL approval gate; study OWASP LLM Top 10 |
| 8 | Practice & review | Full-length timed practice exams; review weak domains; redo missed topics hands-on |

---

## 7. Exam-Day Tips

- **Pace:** ~2 minutes/question; flag and return to long scenario questions.
- **Read carefully:** watch for **NOT** / **EXCEPT** / "BEST" / "FIRST" qualifiers.
- **Think like a system designer:** scenario questions usually ask for the *best* trade-off (cost, latency, safety, scalability), not the only correct tool.
- **NVIDIA-first answers:** when a question mentions the NVIDIA stack, prefer the NVIDIA-native component (NIM, NeMo Guardrails, Triton) over generic alternatives.
- **Hands-on beats theory:** the exam tests practical decision-making; lab time matters more than memorization.
- Test your webcam/proctoring setup 24 h before; have a clean desk and stable internet.

---

## 8. Practice Resources

- NVIDIA official cert page & study guide: https://www.nvidia.com/en-us/learn/certification/agentic-ai-professional/
- build.nvidia.com — free hosted NIM endpoints for hands-on practice
- NVIDIA Blueprints & GitHub (GenerativeAIExamples repo)
- Community guides & practice tests: FlashGenius, Whizlabs, Preporato, Udemy mock exams
- Frameworks to practice with: LangGraph, AutoGen/AG2, CrewAI, NeMo Agent Toolkit

---

## 9. Quick Self-Check (can you answer these?)

1. When would you choose a supervisor multi-agent pattern over a single ReAct agent?
2. What does in-flight batching do in TensorRT-LLM, and why does it improve throughput?
3. How do you measure RAG faithfulness vs answer relevance?
4. Where do input rails vs output rails sit in NeMo Guardrails, and what threats does each mitigate?
5. RAG vs fine-tuning vs prompt engineering — which do you pick for fast-changing knowledge, and why?
6. What is the role of MCP vs A2A in an agentic system?
7. How would you add a human approval step before an agent executes a destructive tool call?
8. Which metrics indicate an agent regression after a prompt update in production?
