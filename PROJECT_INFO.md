# REX AI: Project Information

REX AI is a multi-agent system for a training academy (Andalusia Academy, with synthetic demonstration data). Learners chat with an assistant that recommends courses and answers course questions. Administrators use a dashboard where four specialist agents find tenders, design curricula, write educational documents and create marketing campaigns. It is a graduation project, built in Python with a local language model.

Status as of 2026-11-11. For full details see [README.md](README.md) and [docs/architecture.md](docs/architecture.md).

---

## 1. The five agents

| # | Agent | Who uses it | What it does |
|---|---|---|---|
| 1 | **REX Assistant** | Learners | Answers questions about a specific course, gives personalised recommendations, and lists the course catalogue. English, Arabic, Egyptian Arabic. Answers only from the academy's documents, with sources. |
| 2 | **RFP Research** | Admins | Finds training tenders in Egypt (World Bank API) and Saudi Arabia (web research), filters them, and queues them for review. An approved tender gets an application email draft. |
| 3 | **Curriculum** | Admins | Drafts a curriculum and its Arabic/English companion. It validates both, repairs what can be repaired safely, keeps every version, and creates the PDFs after approval. |
| 4 | **Document Generation** | Admins | Writes 10 document types (course description, proposal, guides, question bank, rubrics…) from an *approved* curriculum, with PDFs. |
| 5 | **Marketing** | Admins | Writes a campaign strategy and 1–3 variations per channel (social, ads, LinkedIn, email, WhatsApp, headlines, calls to action) for an *approved* course. Nothing is published or sent. |

Nothing an agent produces is final until an administrator approves it.

---

## 2. Architecture

```text
Learners:  User -> Streamlit app -> REX Assistant -> RAG MCP server -> knowledge base -> REX writes the answer
Admins:    Admin -> Streamlit app -> AdminService -> Central Orchestrator
                 -> RFP Research | Curriculum | Documents | Marketing  (each through its own MCP server)
                 -> one combined summary -> Admin
```

- **Two separate paths.** REX never goes through the admin orchestrator, and the orchestrator never runs REX. Tests check both.
- **RAG lives in one place.** REX contains no retrieval code. It calls the RAG MCP server, which:
  - translates Arabic queries to English before embedding;
  - searches ChromaDB;
  - reranks the results and returns evidence with file, page, chunk, relevance and distance.
- **Three REX modes:** a specific course, recommendations (only courses the search actually retrieved), and the full catalogue. The catalogue mode returns the complete list, not just the top search results.
- **Orchestrator:**
  - **Planning:** it plans which agents a request needs, including chained requests ("create a curriculum, then a proposal") and mixed ones ("find tenders and create a curriculum"). In a mixed request, each agent receives only its own part.
  - **Running:** it runs every planned agent. If one fails, the others still run.
  - **Reporting:** it writes one summary with each step's result, ids, warnings, whether the steps depended on each other, and a tally.
- **Waiting for approval:** a chained step that needs an approved curriculum pauses. It resumes automatically after the admin approves.

---

## 3. Technology

| Area | Tools |
|---|---|
| Language model | Ollama, `qwen2.5:3b` (configurable per agent) |
| Agent workflows | LangGraph, LangChain |
| Agent tools | MCP: five servers (knowledge base, tenders, curriculum, documents, marketing) |
| Retrieval | ChromaDB, `all-MiniLM-L6-v2` embeddings (384 dimensions), cross-encoder reranking, MarianMT Arabic→English translation |
| PDFs | ReportLab, `arabic-reshaper`, `python-bidi`, Noto fonts |
| External data | World Bank Procurement Notices API, Tavily web search, Resend (email) |
| Interface | Streamlit, English and Arabic (right-to-left) |
| Storage | SQLite (one database per agent), ChromaDB |
| Testing | pytest, Streamlit `AppTest`, opt-in real-model tests |

---

## 4. Knowledge base

- **Source:** the 10 Andalusia Academy files in `knowledge_base/documents/` (14 courses, fees, schedules, instructors, FAQs, admission). These are the only source of academy facts. The data is synthetic.
- **Current index:** 98 chunks, collection `rex_knowledge`, in `knowledge_base/vector_db/`.
- **The index is generated, not committed.** Build it after cloning or after the documents change (stop the app first):

  ```powershell
  python -m knowledge_base.pipeline
  ```

- If the index holds documents that are no longer in the documents folder, the RAG server refuses to use it (`stale_index`), so an old catalogue can't be searched by mistake.

---

## 5. Setup and running

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
ollama pull qwen2.5:3b
python -m knowledge_base.pipeline        # build the knowledge base
python -m streamlit run rex_ai_app.py    # open http://localhost:8501
```

Settings go in `.env` in the project root. It is gitignored and must never be committed. `.env.example` lists every setting:

| Setting | Purpose |
|---|---|
| `REX_ADMIN_PASSCODE` | Turns on the admin dashboard |
| `TAVILY_API_KEY` | Web research (needed for Saudi tenders) |
| `RESEND_API_KEY`, `RESEND_FROM_EMAIL` | Sending application emails |
| `REX_EMAIL_TEST_RECIPIENT` | **The only address any email is sent to** |
| `REX_EMAIL_MODE` | `test` (default) or `production` |
| `REX_LLM_MODEL`, `CURRICULUM_LLM_MODEL`, `MARKETING_LLM_MODEL` | Model choice per agent |

**Admin dashboard sections:** Tender review, New tender search, Ask agents, Tender dashboard, Curricula, Documents, Marketing, Application profile, Audit trail.

---

## 6. Safety rules built into the system

- **Approval first:**
  - Curricula, documents and campaigns need an administrator's approval.
  - Each language is approved separately.
  - Editing an approved curriculum creates a new version that needs its own approval.
- **Email:**
  - Every application email goes **only** to `REX_EMAIL_TEST_RECIPIENT`, in test and production mode alike. The tender's own contact is never used, not even as a fallback.
  - If the setting is missing or invalid, sending is blocked. This is enforced in the backend, not only the UI.
  - Sending also needs the admin's explicit confirmation.
  - Production mode also refuses drafts written from demonstration profile data.
- **No invented facts:**
  - REX recommends only verified catalogue courses.
  - Curricula, documents and campaigns are checked against the approved data. A campaign that invents prices, dates, certificates or prerequisites is refused and not saved.
- **Tests never touch real data or send email.** Every test uses temporary databases and indexes, and the email key is emptied.

---

## 7. Testing

```powershell
python -m pytest -q
```

- **Automated suite:** 3121 passed, 0 failed, 11 skipped. The 11 skipped tests are the opt-in real-model tests. The suite needs no network, no Ollama and no real database.
- **Real-model tests** (need Ollama; take 10+ minutes):

  ```powershell
  $env:REX_REAL_MODEL_TESTS = "1"
  python -m pytest tests/test_rex_real_mcp_e2e.py tests/test_rex_real_model_grounding.py tests/test_curriculum_real_model_e2e.py tests/test_admin_real_workflows_e2e.py -s
  ```

  Last run: 9 passed, 1 skipped (the live tender sources returned no Egypt tenders that day). The marketing workflow failed, as described in section 8.
- **Real browser testing** (headless Chrome against an isolated copy of the app):
  - **Learner side:** all three REX modes, in English and Arabic (right-to-left).
  - **Admin side, sign-in and navigation:** sign-in, all nine sections, language switching.
  - **Tenders and email:** tender approval, the email draft, the configured recipient, and the blocked send.
  - **Curricula:** generation in chat, approval, editing into new versions, PDF and ZIP downloads.
  - **Documents and campaigns:** document generation and approval, and campaign review and download.
- **Manual test plan:** [docs/MANUAL_TEST_PLAN.md](docs/MANUAL_TEST_PLAN.md).

---

## 8. Current status

| Part | Status |
|---|---|
| REX Assistant (English and Arabic, three modes) | Verified with the real model, real MCP server and real browser |
| RFP Research and application emails | Verified. The live search depends on what external sources return. |
| Curriculum Agent | Verified, including the bilingual review and editing |
| Document Generation | Verified from approved curriculum versions |
| Orchestrator and multi-agent workflows | Verified (chained and mixed requests, pause and resume) |
| Marketing Agent | **Not reliable with `qwen2.5:3b`:** 5 real attempts, none saved. Its checks refused unsupported claims ("no prior experience") and near-identical variations. The review, approval and download screens work. |

**Known limitations**
- The small local model is slow. With it, marketing campaigns are usually refused, and Arabic output sometimes fails the checks. A larger `MARKETING_LLM_MODEL` is the suggested fix (not yet tested).
- The Arabic curriculum translation sometimes leaves a title in English. Validation catches it, and the admin fixes it by editing.
- If the Curriculum Agent asks a question in the middle of a chained request, the later steps must be asked for again.
- There is one shared admin passcode, not individual accounts.
- The course catalogue is demonstration data, not real academy data.

---

## 9. Where to find more

| Document | Contents |
|---|---|
| [README.md](README.md) | Full setup, configuration, features and testing |
| [docs/architecture.md](docs/architecture.md) | Workflows, routing, orchestrator, security, storage |
| [docs/PROJECT_TECHNICAL_GUIDE.md](docs/PROJECT_TECHNICAL_GUIDE.md) | Component-by-component technical guide |
| [docs/CURRICULUM_AGENT_ARCHITECTURE.md](docs/CURRICULUM_AGENT_ARCHITECTURE.md) | Curriculum Agent design |
| [docs/MANUAL_TEST_PLAN.md](docs/MANUAL_TEST_PLAN.md) | Step-by-step manual and browser tests |
