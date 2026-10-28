# CLAUDE.md — REX AI project memory

Read this file first in every session. For detail on architecture, RAG, models or learning material, read [docs/PROJECT_TECHNICAL_GUIDE.md](docs/PROJECT_TECHNICAL_GUIDE.md). Last updated: **2026-10-28**.

## 1. Project identity

- **REX AI**: Andalusia Academy graduation project, a multi-agent system (MVP). Final submission **2026-11-05**.
- **Authoritative brief:** "REX AI – Smart Academy Multi-Agent System – AI Graduation Project – Scope of Work & WBS" (PDF, 11 pages). It is **not in the repo**; it was provided as an attachment. Ask the user for it if requirements are in doubt. The implementation must meet the brief; when the code and the brief differ, report the gap instead of assuming.
- **Agents in the brief:** 1 REX Assistant (course chatbot), 2 RFP/Tender Research (Egypt and Saudi Arabia only), 3 Curriculum Research (originally teammate work; **rebuilt 2026-10-27/28 at the user's request**, see §2 and `docs/CURRICULUM_AGENT_ARCHITECTURE.md`), 4 Document Generation, 5 Marketing, plus the Orchestrator.
- **Minimum acceptance criteria include:** at least one working workflow between two or more agents, and a test case for every major feature.

## 2. Current architecture (verified in code)

- **UI:** one Streamlit app, `rex_ai_app.py`. The welcome screen offers User or Admin; that choice is navigation only and grants nothing.
- **User path:** `agents/user_service.py` `RexUserService` → REX Assistant directly. It **never** goes through the orchestrator.
- **Admin path:** passcode checked on the server in `agents/admin_service.py` (`REX_ADMIN_PASSCODE`, constant-time compare, in-memory token, 30-minute idle timeout, 5 failures → 60 s lockout). Every action calls `_require(token)`. Requests go to `create_admin_orchestrator()` in `agents/register_agents.py`, which uses `Orchestrator(excluded_agents=("rex_assistant",))` with RFP Research and Curriculum Research registered.
- **Legacy path:** `create_orchestrator()` (REX included) is kept for the tests and earlier callers. It is not used by the app.
- **Orchestrator:** `agents/orchestrator.py`, a LangGraph plan→execute loop with keyword routing in `_make_plan()`. Document Generation and Marketing have routes but **no agents** (❌), so the curriculum→marketing workflow cannot complete.
- **REX:** `agents/rex_assistant/` LangGraph with nodes `detect_language`, `understand`, `ask_question`, `general`, `retrieve`, `generate`, `handle_consent`.
  - Two modes: discovery (6 fields from the brief) and direct course or academy questions.
  - Reaches the KB **only** through the MCP tool `search_knowledge_base` (`agents/rex_assistant/mcp_client.py` `RagMcpClient` → subprocess `python -m agents.rag_mcp_server`, over stdio). REX needs no other tools: everything else in it is its own reasoning.
- **MCP:** three servers, one shared client class. `agents/mcp_tool_client.py` `McpToolClient` (persistent stdio session on a background thread, restart after a crash, JSON parsing, MCP error flag) is the base of `RagMcpClient`, `agents/rfp_research/mcp_client.py` `TenderMcpClient` and `agents/curriculum_research/mcp_client.py` `CurriculumMcpClient`. A server subprocess gets only MCP's safe default environment plus what the client passes; keys are read from `.env` inside the server.
  - Then **writes the answer itself**: `generate_answer()` in `graph.py` builds the prompt from the chunks (`prompts.py`), calls the LLM, and for recommendations runs `recommendation.check_recommendation()`. `knowledge_base/` contains no answer-writing code (the old `answer_generator.py` was moved into REX on 2026-10-27).
  - Leads are saved only after a clear "yes" (`JSONLLeadStore`, a placeholder).
- **RFP Research:** `agents/rfp_research/` LangGraph `validate_request → query_apis → web_research → filter_relevant → detect_duplicates → assess_and_filter → apply_business_rule → store_opportunities → compose_response` (reworked 2026-10-27).
  - Sources: World Bank API (Egypt, live); Etimad (Saudi) has no public API and is reported `unavailable`; **web research for both countries** (`web_tenders.py`, Tavily, `TAVILY_API_KEY`). Per country: the official portal (`etenders.gov.eg` / `tenders.etimad.sa`, Arabic query), 8 tender aggregator sites, and the open web in English and Arabic. Full page text (`include_raw_content`) goes to `tools.build_extraction_chain` (LLM → `TenderPage` of up to 5 `ExtractedTender`s). Code checks the country, normalises dates, finds document links, labels the training category, and lists missing fields; nothing is guessed.
  - Filters (counted per reason in `filter_report` and shown in the reply): expired or old (`is_stale`, no deadline and more than 365 days); **not training by title** (`state.names_training`: the title must name training — the description is not used because it often names the buyer, e.g. a training corporation buying archiving; our own category label is never used); works/supply contracts and any rental or lease (`_is_works_contract`: a training word overrides a works word only outside an organisation's or place's name — `_without_institution_names` strips "at/for/of the … Authority/Centre/Institute/Ministry…" and "مؤسسة/هيئة/مركز/معهد/جامعة/وزارة…" first); the admin's deadline range (unknown deadlines kept, with a ⚠ line in the reply); the admin's categories. The AI's category label is kept only if the tender's text contains that category's words (`web_tenders._supported_category`).
  - **Tender titles** (`web_tenders.page_to_records`): a generic heading or menu label (`GENERIC_TITLES`, e.g. "تفاصيل المنافسة", "Tender Details"; or, on a single-tender page, a title that appears only as link text) or a heading naming a group of tenders ("… Training Tenders", "مناقصات …") is never a title. It is replaced only by an explicit title field (`title_from_labels`: "اسم المنافسة", "Tender Name"…) on a single-tender page; otherwise the tender is skipped and the page counted once in `web_research.unreadable_pages` (reported on its own ℹ line, apart from the filter counts). Etimad pages fetched by Tavily are empty shells (details load with JavaScript; the report page is behind a CAPTCHA, never bypassed), so they always count as unreadable; Saudi tenders come from the listing sites. Duplicates: this conversation, this run, and the stored catalogue (`OpportunityStore.find_duplicate`: same URL, or same title+country with the same or an unknown issuer).
  - Relevance and qualification are two separate LLM chains; the admin's eligibility notes go to qualification (`{academy_notes}`).
  - Business rule: Egypt → `flagged_for_immediate_review`, Saudi → `monitored_pipeline`; `priority_match` when the admin's priority organisations or sectors appear.
  - **One review queue:** every tender, from the API or the web, is saved in `OpportunityStore` (`rfp_opportunities.sqlite3`) as `approval_status = pending` → the admin approves or rejects (`opportunity_audit`) → approved tenders form the searchable **Tender dashboard**. Tenders **never** enter the course knowledge base (`review_store.py` and `approved_ingestion.py` were removed). In-app alerts: `unseen_alerts()` / `mark_alerts_seen()`.
  - **Tender MCP tools** (`agents/tender_mcp_server.py` → logic in `agents/rfp_research/tender_tools.py`): `search_procurement_notices(country, query)`, `search_tender_web_pages(countries, categories, priority)`, `find_known_tenders(tenders)`, `save_discovered_tenders(tenders)` (pending only; cannot approve, reject or submit). The graph's `query_apis`, `web_research`, `detect_duplicates` and `store_opportunities` call them through an injected `TenderTools` (`TenderMcpClient` by default, pointed at the agent's own DB with `REX_TENDER_DB_PATH`; tests use `tender_tools.LocalTenderTools`). LLM chains, filters, business rule and reply stay in the agent. A failed save is never reported as "waiting for review".
  - **Applications after approval** (`agents/rfp_research/applications.py`, added 2026-10-27): `AdminService.decide(..., "approved", authorize_submission)` saves and audits the approval, then `ApplicationWorkflow.start()` runs `detect_route()` → `OpportunityStore.claim_submission()` (atomic) → adapter → `finish_attempt()`. Routes: World Bank EOI with a contact email **from the official API** → `EmailExpressionOfInterestAdapter`, **dry run only** (prepares the email from the application profile; no sending code exists; `REX_SUBMISSION_MODE` other than `dry_run` is ignored); Etimad, `.gov.eg`, listing sites, web pages, World Bank bids → `ManualRouteAdapter` (official steps). Every result today is `requires_manual_action`. No source has a submission API (Etimad needs Nafath + the authorised signer).
  - **Three separate statuses:** `approval_status` (pending/approved/rejected), `submission_status` (not_started/in_progress/submitted/submission_failed/requires_manual_action/outcome_unknown), `application_status` (not_applied/applied/awarded/not_awarded/withdrawn). `applied` only from a confirmed non-dry-run submission or the admin's `record_manual_submission` (with a confirmation tick in the UI). `application_attempts` table holds every run; `outcome_unknown` (incl. interrupted runs, `recover_interrupted_attempts`) blocks retries until the admin resolves it; an applied/in-progress tender can't be rejected.
  - **Application profile** (`agents/rfp_research/application_profile.py`): 10 admin-entered fields + documents (PDF/DOCX/PNG/JPG ≤ 10 MB, signature-checked, random file names) in `agents/rfp_research/application_documents/` (gitignored). Never sent to an LLM or MCP.
  - Admin UI tabs: Tender review (Approve opens a dialog: route + authorisation checkbox), New tender search, Ask agents, Tender dashboard (submission/application columns and filters, per-tender application panel: missing items, steps, prepared email, manual submission and outcome forms, history), **Curricula** (Agent 3), Application profile, Audit trail.
- **Curriculum Agent (Agent 3, rebuilt; design, decisions D1–D5, verified sources, model comparison and coverage matrix in `docs/CURRICULUM_AGENT_ARCHITECTURE.md`):** `agents/curriculum_research/curriculum_agent.py` `CurriculumAgent` → LangGraph `graph.py` (understand → academy_context → clarify | research → generate (3 structured calls) → validate → one repair → [attach_alignment] → save → render_pdf → reply; plus revise, suggest (Workflow B), show, align, refusals). LangChain `ChatOllama` (`llm.py`, `CURRICULUM_LLM_MODEL`, default = REX's `qwen2.5:3b`; calls capped at 4,096 tokens / 240 s; schemas with required fields and size limits).
  - Tools only through the **Curriculum MCP server** `agents/curriculum_mcp_server.py` (13 tools in `curriculum_tools.py`: catalogue read from the existing index, overlap, gap report, allowlisted research, frameworks, validation, save draft, get/list, render PDF). **No approve tool.** Store `curricula.sqlite3` (versions, documents, audit; gitignored) and PDFs in `agents/curriculum_research/documents/` (gitignored); paths via `CURRICULUM_DB_PATH` / `CURRICULUM_DOCUMENTS_DIR`.
  - Deterministic: `understanding.py` (rules over the reused `request_parser.py`), `validation.py` (measurable outcomes, hours, links, sources, alignment evidence, no accreditation claims, per-field language), `revisions.py`, `alignment.py` (evidence required, else *unable to determine*), `pdf.py` (ReportLab; Arabic via `arabic-reshaper` + `python-bidi`; bundled Noto fonts, embedded).
  - Statuses `draft` / `pending_review` / `changes_requested` / `approved`; only `AdminService.approve_curriculum` / `request_curriculum_changes` (token + audit) change them. Research is **off** (`CURRICULUM_RESEARCH_DOMAINS` empty) until the owner approves the verified list (§16 of that doc); **no framework file is loaded** (D4), so alignment answers *unable to determine*.
  - Routing: curriculum keywords, course creation, and the catalogue-planning rule in `orchestrator.py` (new courses + academy cue, catalogue gaps, "generate suggestion N"); learner phrases still go to REX. Continuation check `wants_follow_up`.

## 3. RAG architecture (`knowledge_base/`, generic and data-agnostic)

- **Ingestion** (`pipeline.ingest_directory`, CLI `python -m knowledge_base.pipeline`): `file_loader` → `parsers` → `cleaner` → `chunker` → `metadata_extractor` → `embedding_service` → `vector_store`.
- **Formats:** pdf, docx, pptx, txt, md, csv, xlsx, json, html, htm. Tested end to end only with generated test files plus the real PDF/CSV/XLSX/JSON/MD catalogue. DOCX/PPTX/HTML with real-world files and Arabic-script PDFs are **unverified**.
- **Chunking:** document-aware headings and records → recursive split → semantic splitting (strategy `auto`/`document`/`semantic`).
  - `max_chars=1000` (the embedder reads at most 256 tokens), `min_chars=250`, `overlap_chars=120`, semantic breakpoint percentile 85.
  - Every chunk starts with its section path; records are never merged.
- **Metadata:** rule-based only, no LLM. `Key: value` facts become `field_<name>` scalars in Chroma (filterable, e.g. `field_fee_egp`). Sources are stored as project-relative paths; each chunk records `embedding_model` and `content_hash`.
- **Models:**
  - embedder `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, local);
  - cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` (local);
  - translator `Helsinki-NLP/opus-mt-ar-en` (MarianMT, local, lazy);
  - LLM via Ollama (local service, temperature 0): **REX uses `qwen2.5:3b`** (default in `agents/config.py`; chosen over `qwen2.5:7b` after live tests on the 6 GB laptop GPU: as accurate, 2–3× faster; override with `REX_LLM_MODEL` in `.env`; `num_ctx=8192`; thinking mode is switched off automatically for models that have one) for both profile extraction and answer writing; RFP Research stays on `qwen2.5:3b`. One model per agent avoids swapping models in the 6 GB GPU.
- **Retrieval** (`retriever.Retriever`, `top_k=5`, `candidate_multiplier=10`): 50 vector candidates → drop duplicates and low-information chunks → cross-encoder rerank → keep score ≥ `MIN_RELEVANCE_SCORE = -9.5` → top 5. It refuses an index built with a different embedding model.
- **Arabic:** `language.prepare_retrieval_query` keeps the original question, normalises Egyptian words, Arabizi, digits and Arabic-spelled tech names, translates **only the Arabic part**, keeps Latin names, codes and numbers verbatim, and reports `translated`/`not_needed`/`failed`. Documents are never translated.
- **MCP interface:** `search_knowledge_base(question, retrieval_query=None, metadata_filter=None)` returns:
  - `status` (`ok`, `no_results`, `empty_index`, `invalid_request`, `error`);
  - `question`, `retrieval_query`, `query_language`, `translation`, `translation_error`, `error`;
  - `chunks[]` with `reference`, `text`, `source` (file name only), `page`, `page_end`, `section`, `section_path`, `record_id`, `chunk_id`, `relevance_score`, `fields`.

  It is the only tool; `ask_knowledge_base` and `RAGAnswerGenerator` were removed on 2026-10-27 as a duplicate answer path.
- **REX receives the chunks and generates the final answer from them** (verified live by trace). No evidence → a fixed bilingual message and no LLM call.

## 4. Architectural decisions (keep them)

1. Improve the existing RAG **in place**; no redesign unless the user explicitly authorises it. Reuse `translator.py`, `retriever.py`, `evidence.py`, `language.py` (RAG) and REX's `prompts.py`/`generate_answer()` (answering); never create duplicates.
2. No course-specific assumptions in `knowledge_base/`. REX rules live in `agents/rex_assistant/recommendation.py`, `prompts.py` and `agents/course_intent.py`. The RAG retrieves; the agent generates.
3. Metadata extraction stays rule-based (Python/regex), never an LLM.
4. Claim support only for formats that are implemented **and** tested.
5. Keep the original (Arabic) query for conversation and answering; use the translated query only for embedding.
6. Keep the MCP server a separate process reached through `RagMcpClient`; don't force it into a LangGraph node.
7. Synthetic data must stay labelled synthetic.
8. Every tender waits for an admin decision; tenders never become RAG knowledge.
9. Retrieval returns evidence; the agent writes the answer.
10. Only meaningful external-service or shared-data capabilities are MCP tools; agent reasoning stays in the agent. Admin decisions and application actions are **never** MCP tools: an MCP server can't identify its caller, so they stay behind `AdminService._require(token)`. The same holds for curriculum approval.
11. Approval ≠ submission ≠ award. Nothing is submitted without the admin's explicit authorisation, nothing is ever sent in this version (dry run), and nothing in an application is invented.
12. Agent 3: framework alignment is never accreditation; alignment needs approved official framework files and evidence; external research only from an owner-approved allowlist (SFIA excluded: its licence forbids external mapping); catalogue facts are labelled synthetic; the model writes content, code decides (validation, hours, statuses).

## 5. Protected areas and rules

- **`agents/curriculum_research/`** was rebuilt with the user's authorisation (2026-10-27/28); the old "protected" rule no longer applies. Follow `docs/CURRICULUM_AGENT_ARCHITECTURE.md`; ask before material changes to its approved architecture. The registration contract `register_curriculum_research(orchestrator, store)` is unchanged.
- **Teammate work:** preserve it, along with existing tests, deliverables, the RAG/MCP components and the language components. `PROJECT_HANDOVER.md` is the user's file; keep it.
- **Git:** before any change, run `git status` and check staged, unstaged and untracked files and `.git/MERGE_HEAD`. **Never** stage, commit, push, reset, revert, discard, switch branches, or resolve/abort the merge without explicit permission. Don't commit SQLite databases or `.env`.
- **Data safety:**
  - Before touching the real index (`knowledge_base/vector_db/`), stop the running app and its MCP servers (with the user's OK) and back up `vector_db/` and `metadata/`.
  - Do live experiments on a **copy** of the index.
- **Secrets:** never print or commit `TAVILY_API_KEY`, `REX_ADMIN_PASSCODE` or any key. Guide the user to edit `.env` themselves; never ask them to paste secrets into chat.

## 6. Current implementation state (2026-10-27)

- **Tests:** `python -m pytest -q` → **1284 passed** (2026-10-28; mocked LLM/APIs, temp stores; `tests/conftest.py` keeps every test off the real tender DB and documents folder, off the network, on in-process tender tools, and gives Agent 3 in-process tools on temporary storage with a model that fails if called; real stdio MCP subprocesses in `test_mcp_client.py`, `test_tender_mcp_server.py` and `test_curriculum_mcp_server.py`). Live-verified by manual runs: the REX→MCP→answer trace on the real index, the reranking trace, Tavily, World Bank, the RFP agent through the real tender MCP server (Egypt + Saudi, temp DB), approval → workflow on those live tenders (7 approved, all `requires_manual_action`, 0 applied, temp DB), and **Agent 3 end to end over the real Curriculum MCP server with `qwen2.5:3b`** (index copy, temp stores: EN generate → revise → align → refused approval, AR generation, suggestions → "generate suggestion 1", admin approval + PDF; 2026-10-28). Agent 3 model comparison: `docs/CURRICULUM_AGENT_ARCHITECTURE.md` §17.
- **Real tender DB:** migrated 2026-10-27 18:01 (additive; 17 tenders and decisions identical; backup of the pre-migration file in the session scratchpad `real_db_backup_before_migration/`).
- **Model comparison** (2026-10-27, 36 checks, index copy): qwen2.5:3b 30/36, qwen3:4b 35/36 but ~54 s/answer, qwen3.5:4b 34/36 at ~3.6 s/answer, gemma3:4b 31/36. REX stays on qwen2.5:3b (the user's choice); the app was running during the run, so the 3b timing may be inflated.
- **Retrieval evaluation:** `knowledge_base/datasets/andalusia_catalogue/eval_set.json` (32 questions (21 English, 8 Arabic, 3 mixed; two listing items carry REX's own `retrieval_query` with its listing hint)) gives **Recall@5 1.0, MRR 0.964** on the index copy with `course_list.md` (the real index still needs re-ingesting to add it); the original pipeline scored 0.333 / 0.218. The set is small and was written alongside the data.
- **Dataset:** a synthetic Andalusia Academy catalogue (**14 courses**, 8 instructors, fees in EGP, intakes, promotions, FAQs, admission), generated from `knowledge_base/datasets/andalusia_catalogue/catalogue_source.json` by `build_documents.py` into 10 files in `knowledge_base/documents/` (including `course_list.md`, one line per course grouped by level, added 2026-10-27 so listing questions retrieve every course).
- **Real index:** rebuilt 2026-10-27 — 94 chunks from 9 files; **`course_list.md` is not in it yet** (stop the app, then `python -m knowledge_base.pipeline` → expect 10 files, 98 chunks), 0 old "NileBridge" chunks. The pre-rebuild index backup is in the session scratchpad (outside the repo); the old PDFs are in git history.
- **Known limitations:**
  - 3B model wording, especially in Arabic, with occasional citation slips;
  - Saudi tenders come only from web research (no Etimad API); web extraction quality depends on the 3B model and on what aggregator pages show (often no issuer or eligibility); RFP alerts are in-app only (no email);
  - no tender source offers a submission API, so applications end as `requires_manual_action`; the EOI email is dry run only; a tender saved through the MCP save tool could claim to be a World Bank record (only our own agent calls it; web extraction can't set those fields; the result would still only be a dry-run draft);
  - no Document Generation or Marketing agents, and no completed two-agent workflow;
  - Agent 3: `qwen2.5:3b` content quality is modest (in the comparison 5/10 curricula valid vs 7/10 for `qwen3.5:4b`, which is 3.6× slower; owner decision pending); some curricula end as drafts after the one repair pass; research off and no framework files until the owner provides them; English change summaries inside Arabic replies/PDFs;
  - JSONL lead store placeholder, with a path relative to the working directory;
  - one shared admin passcode; sessions in memory.
- **Unverified:** browser click-through of the admin dashboard (incl. the approve dialog, application panel, profile upload and the Curricula tab — covered only by AppTest); DOCX/PPTX/HTML real files; Arabic-script PDFs as RAG *input* (Agent 3's Arabic PDF *output* is verified visually and by tests).
- **Housekeeping:**
  - Removed on 2026-10-27 at the user's request: `knowledge_base/chunker_backup.py`, `RAGAnswerGenerator`/`get_chat_model`, MCP tool `ask_knowledge_base`, `RagMcpClient.ask_knowledge_base`, `language.build_retrieval_query`, `language.resolve_response_language`.
  - `requirements.txt`: the unused `fastmcp`, `ollama`, `langchain` (meta-package) and `lingua-language-detector` were removed on 2026-10-28 after an import check with them blocked (`langchain-ollama` still installs the `ollama` client itself); `arabic-reshaper` and `python-bidi` were added for Agent 3's Arabic PDFs (D1). `.env.example` lists every variable the code reads (no values).
  - Removed on 2026-10-28 (Agent 3 rebuild, after the switch-over): the old rule-based `agent.py`, `curriculum_builder.py`, `refinement.py`, `data.py`, `tools.py`, `server.py`, `agent_card.py`, `data/curriculum_examples.json`, and their tests (report in §19 of the Agent 3 doc; copies in the session scratchpad `agent3_removed_backup/`).

## 7. Current stopping point

- **Done:** Agent 3 rebuilt end to end (2026-10-27/28, Phases 0–6: LangGraph + LangChain + Curriculum MCP server, validation, revisions, suggestions, alignment, EN/AR PDFs, AdminService review + Curricula tab, routing, old code removed; report and coverage matrix in `docs/CURRICULUM_AGENT_ARCHITECTURE.md`); MCP-first refactor (shared client, tender MCP server with 4 tools, RFP graph on the tools) and the approval-triggered application workflow (dry run, profile, three statuses, attempts, dashboard), real tender DB migrated; RFP agent rework (web research for both countries, one admin review queue, search form, alerts, dashboard); RAG fixed in place (loaders, chunker, metadata, translation, reranking guard, domain logic moved to REX), new synthetic catalogue, real index rebuilt, REX routing for academy questions, and this documentation (`CLAUDE.md` + `docs/PROJECT_TECHNICAL_GUIDE.md`). The Etimad direct-data question is paused (not cancelled).
- **Owner decisions pending (Agent 3):** approve or edit the verified research allowlist (§16 of the Agent 3 doc; research stays off until then); keep `qwen2.5:3b` or set `CURRICULUM_LLM_MODEL=qwen3.5:4b` (§17); provide official framework files for alignment (D4).
- **Git:** branch `Mena`, **merge in progress** (`MERGE_HEAD` aa7e00cf69). Staged set unchanged by Claude (16 files: the merge's curriculum work, `app.py`, `agents/text_utils.py`, and so on). Many unstaged and untracked changes are Claude's work and are awaiting the user's review. Nothing has been committed.
- **The user should next:**
  1. Restart the app with `python -m streamlit run rex_ai_app.py` (it was stopped for the index rebuild) and test it in a browser with the manual test list.
  2. Review the changes and decide what to commit, including whether to commit the rebuilt `knowledge_base/vector_db/chroma.sqlite3` and `knowledge_base/metadata/*.json`.
  3. Plan the Document Generation and Marketing agents for the brief's acceptance criteria.
  4. Run section J3 of `docs/MANUAL_TEST_PLAN.md` (Curriculum Agent) in the browser.
- **No implementation task is in progress.**

**Commands to resume:**
```powershell
.\.venv\Scripts\Activate.ps1
git status; Test-Path .git/MERGE_HEAD
python -m pytest -q
python -m knowledge_base.evaluation --eval-set knowledge_base/datasets/andalusia_catalogue/eval_set.json
python -m streamlit run rex_ai_app.py
# Rebuild the index only with the app stopped and a backup taken:
python -m knowledge_base.pipeline
```

## 8. Session startup instructions

1. Read `CLAUDE.md`. Read [docs/PROJECT_TECHNICAL_GUIDE.md](docs/PROJECT_TECHNICAL_GUIDE.md) for architecture, RAG, models, or when the user wants to learn.
2. Inspect the actual code before explaining or changing it; this file may be out of date. Reconcile any differences and say so.
3. Check git status, staged/unstaged/untracked files and the merge state before any change.
4. Explain proposed changes in simple English; make targeted changes that preserve the architecture; ask before risky or destructive steps.
5. Never claim something was tested or verified unless it actually was; separate mocked tests from live checks.
6. After changes, report what changed, why, and how it was tested.
7. Update this file after major architectural changes, completed phases or decisions, and update the technical guide when architecture or important implementation details change. Keep a single project-memory file (this one).
