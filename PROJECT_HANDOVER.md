# REX AI — Project Handover

**Prepared:** 2026-10-26
**Prepared by:** Claude (Anthropic), working session on this repository
**Purpose:** Let a new AI assistant, with no memory of prior sessions, understand this project and continue work safely and correctly.
**Based on:** Direct inspection of the actual repository (files, git state, test runs) and live, unmocked API calls executed during this session — not just a summary of past conversation. Every number in this document was either just re-verified by running real code, or is explicitly marked as unverified/reported.

---

## 0. Read this first

- **`app.py` belongs to a teammate. Do not modify, delete, rename, or reformat it, ever, unless the user explicitly authorizes it in that specific conversation.** This rule has been given repeatedly across many prior sessions and must be treated as permanent.
- **The user's own UI file is `app_test.py`** (exact capitalization). This is the file to change when working on the orchestrator/agent integration or UI.
- Jump to **Section 11 — START HERE** for the next assistant's action checklist.

---

## 1. Project Overview

### Purpose
REX AI is a graduation project: a multi-agent system for "Andalusia Academy" (a fictional academy used throughout — all academy branding/data in this repo is illustrative, not a real institution) that automates course recommendations, tender/RFP discovery, curriculum research, document generation, and marketing content. The original project brief defines 5 required agents; only 3 have any implementation so far (see below).

### Agents and their status (verified by inspecting the repo)

| # | Agent | Status | Where |
|---|---|---|---|
| 1 | REX Assistant (course recommendation chatbot) | **Implemented** | `agents/rex_assistant/` |
| 2 | Educational RFP & Tender Research Agent | **Implemented, API-first, Egypt partially working, Saudi Arabia honestly blocked** | `agents/rfp_research/` |
| 3 | Curriculum Research & Accreditation Alignment Agent | **Implemented** | `agents/curriculum_research/` — **out of scope for this assistant unless the user explicitly says otherwise; a standing instruction across prior sessions says not to touch it** |
| 4 | Educational Document Generation Agent | **Not started** | No files exist |
| 5 | AI Marketing Agent | **Not started** | No files exist |

### Technologies/frameworks/libraries actually used (verified in code, not assumed)
- **Python**, **Streamlit** (UI)
- **LangGraph** (`langgraph`) — internal workflow graphs for each agent
- **LangChain** (`langchain-core`, `langchain-ollama`) — prompt templates, structured output, `ChatOllama`
- **Ollama**, model `qwen2.5:3b` — local LLM used by REX Assistant and Agent 2's relevance/qualification judgment
- **ChromaDB**, `sentence-transformers` — RAG pipeline for REX Assistant's knowledge base (`knowledge_base/`)
- **SQLite** (Python stdlib `sqlite3`) — all persistence (`ConversationStore`, `OpportunityStore`)
- **World Bank Procurement Notices API** (`search.worldbank.org/api/procnotices`) — Agent 2's one real, working external API
- **pytest** — test framework
- `pydantic` — structured LLM output schemas and data models

### Agent 2's role and integration (high level — detail in Section 3)
Agent 2 discovers real tender/RFP opportunities for Egypt and Saudi Arabia, filters by relevance and recency, deduplicates, assesses whether the academy can plausibly qualify, applies a business rule (Egypt = flagged for immediate review, Saudi Arabia = monitored pipeline), stores results in a local SQLite catalog, and returns a structured response. It's registered with the same central orchestrator as the other agents (`agents/orchestrator.py`) under the name `"rfp_research"`, invoked automatically when a user's message matches tender/RFP-related keywords (English or Arabic).

### Intended user workflow
A user types a request into the Streamlit chat (`app_test.py`). The orchestrator detects which agent should handle it (keyword-based routing) and calls that agent. For Agent 2, a request like *"Find education and training tenders in Egypt"* triggers a live API query, filtering/assessment pipeline, and returns a chat message listing: which API(s) were queried and their status, any individual opportunities found (with qualification judgment), and duplicate/new counts.

---

## 2. Project Architecture

### Directory structure (relevant parts, verified via `find`)

```
rex-ai/
├── app.py                         # LEGACY/teammate-owned. DO NOT TOUCH.
├── app_test.py                    # ⭐ The actual working UI + orchestrator integration test harness
├── requirements.txt
├── agents/
│   ├── orchestrator.py            # Central LangGraph-based router (Orchestrator class)
│   ├── register_agents.py         # Wires up all agents + shared stores; create_orchestrator()
│   ├── conversation_store.py      # Generic SQLite per-(conversation_id, agent_name) state store
│   ├── rex_assistant/             # Agent 1
│   ├── curriculum_research/       # Agent 3 (do not modify per standing instruction)
│   └── rfp_research/              # Agent 2 — see Section 3
│       ├── agent.py               #   RFPResearchAgent.process_request() — orchestrator entry point
│       ├── graph.py               #   LangGraph workflow (the actual pipeline logic)
│       ├── tools.py               #   LLM relevance/qualification judgment (LangChain)
│       ├── state.py               #   RFPResearchState TypedDict + constants
│       ├── opportunity_store.py   #   SQLite catalog of discovered opportunities
│       └── api_clients/
│           ├── base.py            #   NormalizedTender, APIQueryResult (shared data model)
│           ├── world_bank.py      #   REAL, working Egypt connector
│           └── etimad.py          #   Honest "unavailable" Saudi Arabia connector (no network call)
├── knowledge_base/                # RAG pipeline backing REX Assistant (unrelated to Agent 2)
├── data/                          # Synthetic/sample datasets for other agents
│   └── README.txt                 # Explains what each dataset file is
└── tests/                         # pytest suite, 96 tests (verified this session, see Section 5)
```

### How the application starts

```
streamlit run app_test.py
```

This is the correct, working entry point for testing the integrated multi-agent system. `app.py` is a separate, older, teammate-owned prototype — **not** part of this assistant's working set.

### `app.py` vs `app_test.py` — the relationship

- `app.py`: legacy prototype, predates the orchestrator, single-agent, bypasses `agents/orchestrator.py` entirely. Owned by a teammate. **Off-limits.**
- `app_test.py`: the real integration surface. Creates one orchestrator via `agents.register_agents.create_orchestrator()` (cached with `@st.cache_resource`), assigns a per-browser-session `conversation_id`, and calls `orchestrator.handle_request(user_input, context={"conversation_id": ...})` per message. Renders results differently depending on which agent responded (`curriculum_research`, `rex_assistant`, or `rfp_research` each have their own rendering branch).

### The orchestrator's role

`agents/orchestrator.py` — `Orchestrator` class, built on `langgraph.graph.StateGraph`. `handle_request()`:
1. Detects Arabic/English (`detect_language`, simple Unicode-range check).
2. Picks an agent via keyword matching (`_make_plan`) — e.g. "tender", "rfp", "مناقصة" → `rfp_research`; "recommend a course" → `rex_assistant`; "منهج"/"curriculum" → `curriculum_research`. Multi-agent plans are supported (e.g. curriculum + marketing) but only single-agent plans are currently reachable since marketing doesn't exist.
3. Calls the registered handler for that agent via `agents.register_agents.create_orchestrator()`'s registrations, passing a common request dict `{request_id, agent_name, user_input, context}`.
4. Normalizes whatever the handler returns into a common response dict `{request_id, agent_name, status, result, error_message}`.
5. `context["previous_results"]` carries prior agents' results forward within one multi-agent plan (a **snapshot**, not a live reference — this was a confirmed-and-fixed bug earlier in this project's history).

`RFPResearchAgent.process_request(request: dict) -> dict` is written to directly match the orchestrator's expected handler signature (same pattern as `CurriculumResearchAgent`) — **no adapter needed**, unlike REX Assistant, whose `handle_message()` signature required a small adapter function (`make_rex_assistant_handler` in `register_agents.py`).

### Agent 2 data flow (verified against the actual code, not assumed)

```
User message ("Find education and training tenders in Egypt")
        │
        ▼
agents/orchestrator.py  (keyword routing → "rfp_research")
        │  common request dict
        ▼
agents/rfp_research/agent.py :: RFPResearchAgent.process_request()
        │  builds initial RFPResearchState, invokes the LangGraph
        ▼
agents/rfp_research/graph.py  (compiled LangGraph, nodes run in order):

  validate_request
      │  parses which countries were asked for (Egypt/Saudi Arabia/both);
      │  detects an explicitly out-of-scope country (e.g. "Germany") and
      │  short-circuits with a bilingual rejection message, no API calls
      ▼
  query_apis
      │  Egypt  → agents/rfp_research/api_clients/world_bank.py
      │             query_world_bank_egypt("training")  [REAL HTTP call]
      │  Saudi  → agents/rfp_research/api_clients/etimad.py
      │             query_etimad_saudi_arabia(...)  [NO network call,
      │             always returns status="unavailable" — see Section 4]
      │  Both connectors return an APIQueryResult (status/message/records)
      ▼
  filter_relevant
      │  lightweight keyword pre-filter (CATEGORY_KEYWORDS in state.py)
      │  on title/scope_of_work/category, plus expiry check
      │  (THIS IS WHERE THE CURRENT UNRESOLVED PROBLEM LIVES — see §4/§6)
      ▼
  detect_duplicates
      │  keyed by "{source_name}::{source_id}" (the API's own stable ID),
      │  checked against agents/conversation_store.py, scoped per
      │  conversation_id
      ▼
  assess_and_filter
      │  agents/rfp_research/tools.py :: build_assessment_chain()
      │  ONE LLM (qwen2.5:3b via LangChain ChatOllama) call per candidate:
      │  judges is_relevant (bool) + qualification_status (3-state) +
      │  summary + reason. is_relevant=False → record dropped here too.
      ▼
  apply_business_rule
      │  Egypt → review_status="flagged_for_immediate_review"
      │  Saudi Arabia → review_status="monitored_pipeline"
      │  (this label is PRIORITY ONLY — never implies verified/qualified)
      ▼
  store_opportunities
      │  agents/rfp_research/opportunity_store.py :: OpportunityStore.save()
      │  persists to SQLite, reports whether each record is genuinely new
      ▼
  compose_response
      │  builds the final {message, opportunities, api_results,
      │  duplicate_count, newly_discovered_count} dict
      ▼
Back through agent.py → orchestrator.py → app_test.py's rfp_research
render branch (shows API source status, per-opportunity details,
qualification labels, missing-info notes)
```

### Database/persistence design

Two separate, intentionally non-interchangeable SQLite stores (both plain Python stdlib `sqlite3`, no ORM):

| Store | File | Scope/lifetime | Used by |
|---|---|---|---|
| `ConversationStore` | `agents/conversation_state.sqlite3` | Per `(conversation_id, agent_name)` — conversation memory | REX Assistant (learner profile/consent/history); Agent 2 (which `source_id`s were already shown in this conversation, for dedup) |
| `OpportunityStore` | `agents/rfp_research/rfp_opportunities.sqlite3` | Project-wide catalog, independent of any conversation | Agent 2 only — the "searchable dashboard" requirement, implemented as a real queryable SQLite table (not a Google Sheet — no credentials for that exist) |

Both stores use **non-destructive, additive schema migration** (`ALTER TABLE ... ADD COLUMN`, guarded by `PRAGMA table_info`) specifically so that a live, already-running Streamlit session's open database file is never broken or deleted by a code update. This was deliberately engineered after a near-miss earlier in this project where an assistant almost deleted the user's live database file, mistaking their running `streamlit run app_test.py` process for an orphaned background process — **always check `Get-Process`/`ps` command lines before touching any `.sqlite3` file that might be locked.**

---

## 3. Agent 2 — Current Implementation (verified against the actual code)

**Purpose:** Discover real, actionable educational/training tender opportunities in Egypt and Saudi Arabia that the academy could plausibly bid on.

**Countries supported:** Egypt and Saudi Arabia only, enforced in `agents/rfp_research/state.py::ALLOWED_COUNTRIES` and checked at multiple pipeline stages. A request naming an explicitly out-of-scope country (USA, Europe, UK, Germany, France, Canada, Australia, Spain, Italy — see `_DISALLOWED_COUNTRY_HINTS` in `graph.py`) is rejected immediately with a bilingual message, before any API call.

**Relevant categories** (`CATEGORY_KEYWORDS` in `state.py`): education, training, technical training, professional development, digital transformation, software, IT training, corporate learning, e-learning, curriculum, capacity building, skills. These are used for a **keyword pre-filter** before the LLM relevance judgment — see the unresolved problem in Section 6.

**Fields captured per tender** (`NormalizedTender` in `agents/rfp_research/api_clients/base.py`): `title`, `reference_number`, `organization`, `country`, `source_id`, `source_url`, `publication_date`, `deadline`, `status` (as reported by the source API, not inferred), `category`, `scope_of_work`, `eligibility` (list), `submission_instructions`, `submission_url`, `available_documents` (list), `retrieved_at`, `source_name`, `verification_status`, `missing_information` (list). **A field the API/record doesn't actually state is left `None`/empty and listed in `missing_information` — never invented.**

**Expired opportunities:** `_is_expired()` in `graph.py` parses `deadline` as ISO date and excludes anything in the past. A deadline that can't be parsed is treated as "not provably expired" (kept, not guessed-away).

**Duplicate detection:** keyed by `f"{source_name}::{source_id}"` — the API's own stable identifier, not a scraped URL (this changed when the scraping approach was removed — see Section 4). Tracked per `conversation_id` via `ConversationStore`.

**Qualification assessment — the 3-state system (verified present in code):**
Defined in `agents/rfp_research/tools.py::TenderAssessment.qualification_status`, `Literal["qualified", "not_qualified", "unknown"]`:
- `qualified` — stated eligibility requirements are fully consistent with the academy's known capabilities.
- `not_qualified` — a stated requirement is clearly NOT met.
- `unknown` — eligibility isn't stated, or requires info about the academy (registrations/certifications/geographic licensing) that isn't available. **The prompt explicitly instructs: never default to `qualified` just because the topic is education/training, and never default to `not_qualified` merely because eligibility wasn't stated — that case is `unknown`.**

The academy's "known capabilities" used for this judgment (`ACADEMY_CAPABILITIES_SUMMARY` in `tools.py`) is a short, honest statement: the academy designs/delivers technical and professional training; **no information is available about formal registrations, certifications, financial capacity, or geographic licensing in Egypt/Saudi Arabia specifically.**

**Business rule (verified in `apply_business_rule` node, `graph.py`):**
- Egypt → `review_status = "flagged_for_immediate_review"`
- Saudi Arabia → `review_status = "monitored_pipeline"`
- Explicitly documented in code comments and shown in the UI as **priority-only** — never implies the tender is verified open or that the academy qualifies (that's `qualification_status`'s job, a separate field).

**Current limitations (honest, not glossed over):**
1. **The Egypt relevance filter is currently rejecting everything a live run finds — see Section 6, this is the #1 open problem.**
2. The World Bank API only covers World Bank-*financed* projects in Egypt — a real but partial subset of all Egyptian government procurement, not the national portal.
3. No working Saudi Arabia API exists (architecture ready, not operational — see Section 4).
4. At least one confirmed case of the LLM (`qwen2.5:3b`) producing an internally inconsistent judgment: a summary stating a tender "aligns with the academy's expertise" while simultaneously setting `is_relevant=False`. Not a code bug — a model reliability limitation, consistent with similar issues documented elsewhere in this project (e.g. REX Assistant's language-instruction adherence).
5. No Google Sheets/dashboard UI, no real notification delivery (email/Slack) — `OpportunityStore` provides the underlying searchable persistence, but no outward-facing dashboard or alerting exists.

---

## 4. API Migration and Cleanup (verified: files deleted, dependencies removed)

### Why the scraping approach was removed
An earlier version of this agent used DuckDuckGo web search (`ddgs` package) plus HTML scraping (`beautifulsoup4`) to discover tenders. This was explicitly rejected by the project owner: the brief requires discovering **genuine** opportunities from external sources, and most scraped results turned out to be directory/category pages mislabeled as individual tenders, with unreliable field extraction. The owner explicitly required an **API-first** architecture instead.

### What was added
- `agents/rfp_research/api_clients/base.py` — `NormalizedTender` (pydantic model, the common tender shape) and `APIQueryResult` (status: `"success" | "unavailable" | "error" | "awaiting_credentials"`, with `message`, `records`, `raw_record_count`). **This 4-state status model is the core mechanism preventing an API failure from ever being silently shown as "zero tenders found."**
- `agents/rfp_research/api_clients/world_bank.py` — real, working Egypt connector. See below.
- `agents/rfp_research/api_clients/etimad.py` — honest "unavailable" Saudi Arabia connector. See below.

### What was deleted (verified via `grep` showing zero remaining importers before deletion)
- `agents/rfp_research/search.py` — the DuckDuckGo + BeautifulSoup scraper. Fully removed.
- `agents/rfp_research/data.py` — a synthetic-tender-data loader that had already become dead code (zero importers) in an earlier refactor. Fully removed.
- `data/tenders.json` — the synthetic/fixture tender dataset (10 fictional records) this loader read. Fully removed. `data/README.txt` was updated to remove the reference and note that Agent 2 now uses real APIs.
- `ddgs` and `beautifulsoup4` removed from `requirements.txt` (confirmed no longer imported anywhere in the codebase).

### API connector behavior when unavailable
Both connectors always return an `APIQueryResult`, never raise past the connector boundary for ordinary failure modes, and never return a `"success"` status with an empty/fabricated record list. `graph.py`'s `compose_response` node explicitly checks `any_api_succeeded` and, if false, returns a message that starts with **"No tender data could be retrieved -- this is NOT the same as finding zero relevant tenders."** followed by each connector's own status and message. This was verified live for Saudi Arabia in this session (see Section 6).

### Environment variables
- `ETIMAD_API_KEY` — read by `agents/rfp_research/api_clients/etimad.py`. **Currently has no functional effect** — there is no confirmed real endpoint to call with it yet. It exists only so a future real Etimad integration can be dropped in without redesigning the connector interface. If set, the connector's message notes it's set but still reports `"unavailable"` with an explanation.
- No other environment variables are used by Agent 2. **No API keys are required or hardcoded for the World Bank connector** — it's a public, unauthenticated API.

### Tested against real endpoints or only mocked?
**Both, clearly separated:**
- **Automated test suite** (`tests/test_rfp_api_clients.py`, `tests/test_rfp_research.py`): fully mocked at `urllib.request.urlopen` (for `world_bank.py`) and at `agents.rfp_research.graph.query_world_bank_egypt` / `query_etimad_saudi_arabia` / `build_assessment_chain` (for the graph pipeline). These tests prove the *code's logic* is correct given a certain API response shape — they do **not** prove the live API currently behaves that way.
- **Live, unmocked verification** (run directly in this session, not via pytest): real HTTP calls to `search.worldbank.org/api/procnotices`, real LLM calls to a local Ollama `qwen2.5:3b`. See exact figures in Section 6 — these were just re-verified again while preparing this document (not just recalled from memory).

### World Bank Procurement Notices API — what's confirmed

| Fact | Status |
|---|---|
| Endpoint `https://search.worldbank.org/api/procnotices` exists and responds with real JSON | **Verified** (direct `curl` and Python calls, multiple times across sessions) |
| No API key/auth required | **Verified** — confirmed working with zero credentials |
| `qterm` free-text search works, but appears to use AND-semantics across space-separated words | **Verified** — a long compound query (`"Egypt education training technical IT software digital skills professional development"`) returned `total: 0`; a short query (`"Egypt training"`) returned `total: 210`. The code was fixed accordingly (`CATEGORY_QUERY = "training"` in `graph.py`, with a code comment explaining why). |
| Covers only World Bank-financed projects, not Egypt's full national tender portal | **Verified by absence** — no Egypt-wide government portal API was found despite investigation; this is a known, accepted scope limitation, not a bug |
| Response includes real reference numbers, organizations, deadlines (when published), notice types | **Verified** in live response payloads |

### Saudi Arabia / Etimad — what's confirmed

| Fact | Status |
|---|---|
| Official Etimad Developer Portal exists (`apiportal.etimad.sa`) | **Verified** (portal exists, page content inspected) |
| Its listed products ("Contracts Plus", "Bank Contracts Inquiry") are historical-contract lookups requiring an already-known contract/beneficiary ID, not tender discovery | **Verified** via portal page content |
| These products appear to require a commercial agreement | **Reported in portal content; not independently confirmed with pricing/contract details** |
| A public, documented, open tender-*listing* API | **Not found.** Investigated via web search and direct fetch attempts (which returned connection errors for the national portal `etenders.gov.eg`-equivalent Saudi sites) |
| Third-party wrappers (e.g. "Tenders Alerts") exist | **Verified they exist**; explicitly **not integrated** — they require vendor registration for an API key and are not an official government source, which the project owner explicitly said not to disguise as "the real integration" |

**Conclusion, stated plainly: the Saudi Arabia connector is an honest placeholder, not a working integration. Nothing should claim otherwise.**

---

## 5. Testing and Verification

### Framework and commands
```bash
# Run the whole suite
python -m pytest tests/ -q

# Run just Agent 2's tests
python -m pytest tests/test_rfp_research.py tests/test_rfp_api_clients.py tests/test_opportunity_store.py -v

# Launch the application (the correct UI file)
streamlit run app_test.py
```
(On this Windows machine, the virtualenv's Python is at `.venv/Scripts/python.exe` — use that explicitly if `python`/`streamlit` aren't on PATH in a fresh shell.)

### Latest verified test result
**Just re-run while preparing this document**: `python -m pytest tests/ -q` → **96 passed, 0 failed.** (This matches the previously reported figure of 96 — independently re-confirmed, not merely repeated from memory.)

### Important test files for Agent 2
| File | Covers |
|---|---|
| `tests/test_rfp_api_clients.py` | `world_bank.py` (mocked HTTP: success, non-Egypt filtering, historical-notice exclusion, missing-field honesty, network/malformed-response errors) and `etimad.py` (always-unavailable, no network call, `ETIMAD_API_KEY` presence noted but inert) |
| `tests/test_rfp_research.py` | Full graph pipeline (mocked connectors + LLM): successful flow, API-error-not-shown-as-empty, Saudi unavailable-not-empty, no-fallback-to-deleted-search-module, keyword pre-filter, LLM relevance exclusion, expiry exclusion, out-of-scope rejection, duplicate detection by `source_id`, `unknown` qualification when eligibility unstated, persistence, assessment-failure graceful handling, empty-input rejection |
| `tests/test_opportunity_store.py` | SQLite catalog: save/search, country/keyword search, list round-tripping, the fixed "unknown ≠ not_qualified" persistence bug, API-sourced records without a literal URL, cross-instance persistence |

### Known gap between automated test success and real-world functionality
**This is the central open issue.** All 96 tests pass, and they correctly verify the *pipeline's logic* — but a live run against the real World Bank API currently returns **0 final opportunities** despite 18 genuinely relevant-looking Egypt records being retrieved (see Section 6). The mocked tests cannot catch this because they control what the LLM "returns" — they don't exercise whether the real `qwen2.5:3b` model actually judges real records as relevant. **Passing tests here does not mean the live agent is useful yet.**

---

## 6. Current Live Test Results (just re-verified, not recalled)

### Egypt query
**Input:** `"Find education and training tenders in Egypt"`

**Actual output (re-run live during this handover's preparation):**
```
MESSAGE: Found 0 relevant tenders via API (0 newly discovered, 0 duplicate(s) skipped).
World Bank Procurement Notices API (Egypt: World Bank-financed projects only) | success | 25 raw
  → Retrieved 25 raw record(s); 18 confirmed as genuinely Egypt-specific, current opportunities.
opportunities: 0
```
This **exactly matches** the figures the project owner reported (25 raw, 18 confirmed Egypt-specific, 0 final relevant tenders) — independently re-verified, not merely copied from a prior report. Note: the World Bank API is a live, changing dataset, so these exact counts could drift slightly on a future run; the *pattern* (raw > confirmed > 0 after relevance filtering) is the reproducible, important fact.

**Diagnosis already done in a prior session, not yet fixed:** Inspecting one of the 18 records directly (title: *"AQM Skills Training, and developing University Curriculum"*) showed the LLM's own generated summary stating the tender **"aligns with the academy's expertise in designing and delivering technical and professional training courses and curricula,"** while the same structured-output call set `is_relevant=False`. This is an internal inconsistency in the model's output, not (as far as has been verified) a prompt-logic or code bug — but it has **not been systematically investigated across more than one record**, and the keyword pre-filter's own precision/recall hasn't been separately audited either. **Both are candidate root causes and neither has been ruled out. See Section 7, Immediate Next Steps.**

### Saudi Arabia query
**Input:** `"Find education and training tenders in Saudi Arabia"`

**Actual output (re-run live during this handover's preparation):**
```
SAUDI MESSAGE: No tender data could be retrieved -- this is NOT the same as finding zero relevant tenders.
Etimad (Saudi Arabia) -- no verified tender-listing API: unavailable -- No documented, publicly
accessible API for listing open Saudi Arabia tenders was confirmed. [...] No ETIMAD_API_KEY is
configured. Set it once a documented tender-listing endpoint and credentials are obtained from
Etimad to enable this connector.
```
This confirms the explicit design goal: an unavailable API source is **never** presented as "we searched and found zero" — it's presented as "we could not search at all, here's why."

---

## 7. Exact Current Status and Remaining Work

### Completed (verified by repository inspection, tests, or just-run live output)
- [x] Central orchestrator with keyword-based routing to `rex_assistant`, `curriculum_research`, `rfp_research` (`agents/orchestrator.py`)
- [x] `agents/register_agents.py` wires up all three agents + shared `ConversationStore`
- [x] Agent 2 fully rebuilt as API-first (no scraping) — `agents/rfp_research/` package
- [x] World Bank Procurement Notices API connector — real, working, unauthenticated, verified live multiple times
- [x] Etimad (Saudi) connector — honest "unavailable" status, verified to make zero network calls, never silently empty
- [x] 3-state qualification model (`qualified`/`not_qualified`/`unknown`) present in code and tested
- [x] Egypt/Saudi business rule (`flagged_for_immediate_review` / `monitored_pipeline`) implemented and tested
- [x] Duplicate detection by stable `source_id`, scoped per conversation, via `ConversationStore`
- [x] `OpportunityStore` — searchable SQLite catalog, non-destructive schema migration, API-sourced-record key synthesis
- [x] Old scraping code and synthetic JSON dataset fully deleted; unused dependencies (`ddgs`, `beautifulsoup4`) removed from `requirements.txt`
- [x] `app_test.py` updated to show per-API-source status, qualification labels, missing-info notes
- [x] 96/96 automated tests passing (just re-verified)
- [x] `app.py` confirmed untouched throughout (checked via `git diff --stat app.py` returning empty, repeatedly, across sessions)

### In progress / needs verification
- [~] Egypt relevance filtering — pipeline runs end-to-end without error, but currently produces 0 usable results from 18 genuinely-Egypt, apparently-relevant candidates. **Root cause not yet isolated between the keyword pre-filter and the LLM judgment.**
- [~] Whether the keyword pre-filter (`_mentions_relevant_category` in `graph.py`) is even the bottleneck, or whether candidates are reaching the LLM and being rejected there — **this specific trace has not been done and is the literal next step.**

### Not started or blocked
- [ ] Document Generation Agent (Agent 4) — no files exist
- [ ] Marketing Agent (Agent 5) — no files exist
- [ ] Saudi Arabia tender API — blocked on either Etimad granting real documented access, or discovery of another legitimate government/multilateral source not yet found
- [ ] Google Sheets/dashboard UI for `OpportunityStore` — not attempted, explicitly deferred by the project owner's own instruction
- [ ] Real notification/alerting on new opportunities — not attempted, no channel/credentials exist

### Known bugs and risks
1. **Egypt relevance-filter problem (highest priority, unresolved).** 18 genuine candidates → 0 final results. Confirmed via live, repeated testing. Root cause not isolated.
2. **`qwen2.5:3b` structured-output inconsistency** — confirmed in at least one specific case (summary contradicts `is_relevant` flag). Not yet checked across a larger sample to establish how frequent this is.
3. **World Bank API `qterm` AND-semantics** — already found and worked around (see Section 4), but the chosen workaround (single broad term `"training"` + separate keyword/LLM filtering) has not been stress-tested against other category phrasings (e.g. "digital skills", "IT training" alone).
4. **Source coverage is partial** — World Bank-financed Egypt projects only; no Saudi source at all.
5. **Live API data changes over time** — exact record counts (25 raw / 18 confirmed) are a snapshot, not a fixed test fixture; expect them to drift.

### Immediate next steps (recommended order)
1. **Trace the 18 Egypt records through the full pipeline manually** — print what `filter_relevant` keeps vs. drops (keyword stage), then print the raw LLM input/output for each record that reaches `assess_and_filter`. Determine the actual drop point(s).
2. **Determine why relevant-looking opportunities are rejected** — is it the keyword pre-filter being too strict, or the LLM's `is_relevant` judgment being unreliable (as the one confirmed case suggests)? Test both hypotheses with real data, not assumptions.
3. **Improve relevance/qualification logic without loosening it into admitting unrelated tenders** — e.g., consider whether the LLM prompt in `agents/rfp_research/tools.py::build_assessment_chain` needs restructuring (perhaps separating the relevance judgment from the qualification judgment into two calls, since combining them may be where the inconsistency creeps in), or whether a larger/different Ollama model performs more reliably (there is precedent elsewhere in this project — `qwen2.5:7b` is already pulled locally and was found more reliable than `qwen2.5:3b` for a different instruction-following task in REX Assistant).
4. **Add realistic positive and negative regression tests** using real-looking data (e.g., a fixture built from the actual "AQM Skills Training" record) so this exact failure mode can't silently reappear.
5. **Rerun automated tests and live API checks** to confirm the fix actually produces relevant-looking results, not just that the tests pass.
6. **Investigate a legitimate, documented Saudi open-tender data source** — this has been investigated once (Section 4) and came up empty; a second, deeper pass (e.g. checking whether Etimad's Developer Portal has added new products, or whether a multilateral source like UNDP/other MDBs covers Saudi Arabia) could still be worthwhile, but don't invent an endpoint if none is found.
7. **Test the Streamlit interface and orchestrator** end-to-end (`streamlit run app_test.py`) for both countries, in Arabic and English.
8. **Review `git diff` and decide with the user whether/what to commit and push** — nothing has been pushed; see Section 9.

---

## 8. CRITICAL: Protected Files and Working Rules

> **These are standing project rules, repeated across many prior sessions. Treat them as permanent until the user says otherwise.**

- **`app.py` belongs to a teammate. DO NOT MODIFY, DELETE, RENAME, OR REFORMAT IT** unless the user explicitly authorizes it in that specific conversation. A past session nearly broke this rule by considering `app.py` "obsolete" — it was not; the user confirmed a teammate owns it.
- **The user's own UI file is `app_test.py`** (exact filename, exact capitalization — not `App_Test.py`, not `app-test.py`).
- **Changes should be limited to the files needed for the requested task.** Don't do a speculative broad cleanup unless asked.
- **Preserve existing database records.** Both `ConversationStore` and `OpportunityStore` use non-destructive, additive schema migrations specifically for this reason. Never `DROP TABLE`, never delete a `.sqlite3` file to "reset" it — if a schema change is needed, add a migration.
- **Before touching any `.sqlite3` file, check whether a live Streamlit process has it open** (e.g. `Get-CimInstance Win32_Process | Select ProcessId, CommandLine` on Windows, or `lsof`/`ps` on Unix) — a file lock error can mean the user's own running session, not an orphaned process.
- **Do not fabricate tender records, API results, endpoints, credentials, or successful test results.** If something can't be verified, say so explicitly.
- **Clearly distinguish live API results from mocked test data** in anything you report to the user.
- **Do not describe an unavailable API connector as a functioning integration.** The Etimad connector is a placeholder — say so, every time, not just once.
- **Review `git diff` before committing** (nothing has been committed by any assistant session on this project so far — see Section 9).
- **Do not commit secrets, API keys, local environment files, or unintended database changes.**

---

## 9. Git and Repository State (verified just now)

**Current branch:** `Mena`
**Remote:** `origin` → `https://github.com/MennaElsekily/rex-ai.git`

**Uncommitted changes (verified via `git status --short`):**
```
 M agents/register_agents.py
 M app_test.py
 M data/README.txt
 D data/tenders.json
 M requirements.txt
 M tests/test_register_agents.py
?? agents/conversation_state.sqlite3
?? agents/rfp_research/
?? tests/test_opportunity_store.py
?? tests/test_rfp_api_clients.py
?? tests/test_rfp_research.py
```
**`app.py` has zero diff** — confirmed via `git diff --stat app.py` returning empty output.

**Has this been pushed?** **No evidence of a push by any AI assistant session.** No `git commit` or `git push` command has been run as part of this work — every session this document's content is based on explicitly avoided committing per standing instructions ("do not commit or push automatically"). The last commits on this branch (`ea8de8d`, `c98fa75`, etc.) predate this work and appear to be the user's/teammate's own commits.

**Recommended safe commands for the next session (do NOT run these automatically — present them to the user, or run only with explicit confirmation):**
```bash
# Review what actually changed, carefully, before anything else
git status
git diff agents/register_agents.py app_test.py requirements.txt tests/

# Check there's nothing sensitive in the new files
git diff --stat
cat agents/rfp_research/api_clients/etimad.py | grep -i "key\|secret\|token"   # should show only the ETIMAD_API_KEY *name*, never a value

# Stage only the intended files (avoid a blind `git add -A`/`git add .`)
git add agents/register_agents.py app_test.py data/README.txt requirements.txt \
        tests/test_register_agents.py agents/rfp_research/ \
        tests/test_opportunity_store.py tests/test_rfp_api_clients.py tests/test_rfp_research.py
git rm data/tenders.json   # if not already staged by the add above

# Confirm app.py and agents/curriculum_research/ are NOT staged
git status

git commit -m "Agent 2: migrate from web scraping to API-first tender discovery"

# Only push if the user explicitly asks
git push origin Mena
```
**Do not stage `agents/conversation_state.sqlite3` or `agents/rfp_research/rfp_opportunities.sqlite3` without first checking with the user** — these are local runtime databases that may contain data from the user's own testing; whether they belong in version control is the user's call, not an automatic default.

---

## 10. START HERE — Instructions for the Next AI Assistant

1. **Read this entire file before making any changes.**
2. **Inspect the current repository and compare it against this handover** — run `git status`, `git log -5`, `python -m pytest tests/ -q`, and re-run the live Egypt query (see the Python snippet pattern in Section 6, or just use `streamlit run app_test.py`) to confirm the state described here still holds.
3. **If the repository has changed since this document was written, treat the actual code and current test/live output as authoritative over this document.** This document is a snapshot, not a source of truth that overrides reality.
4. **Preserve all protected-file rules from Section 8, especially never modifying `app.py` without explicit fresh authorization in your own conversation.**
5. **Start work by fixing and testing the Egypt relevance-filter problem** (Section 7, Immediate Next Steps #1–5) — this is the single most impactful, well-understood, and already-diagnosed-but-unfixed issue in the project.
6. **Avoid repeating work already completed** (Section 7, "Completed") unless your own verification shows something regressed.
7. **Ask the user for clarification only when a genuinely essential decision is missing** — e.g., whether to commit/push, whether to pursue a Saudi API lead that requires a paid/commercial agreement, or whether to swap the LLM model. Don't ask about things this document already answers.
8. **Explain commands in beginner-friendly, step-by-step language when guiding the user** — they have been directing highly detailed technical work across many sessions, but prefer to review and approve commands rather than have them run silently, especially anything touching git or databases.
9. **Report exactly what was changed and what tests were actually run** — distinguish mocked results from live API/model results explicitly, every time, the same way this document does.
10. **Continue the project from its current state. Do not restart, redesign, or rewrite working components** (the orchestrator, `ConversationStore`, `OpportunityStore`, the World Bank connector, the overall LangGraph pipeline shape) **unless you find and clearly explain a concrete defect that requires it.**
