# REX AI – Smart Academy Multi-Agent System

## 1. Project Overview

REX AI is a multi-agent system designed to support educational and training operations. It combines specialized AI agents to help students discover suitable courses, research educational tenders, design curricula, generate educational documents, and create marketing content.

This project is being developed as a graduation project.

## 2. Project Objectives

- Provide personalized course recommendations.
- Research and filter educational tenders and requests for proposals (RFPs) in Egypt and Saudi Arabia.
- Support curriculum development and accreditation alignment.
- Generate professional educational documents in PDF format.
- Assist with educational marketing and campaign content.
- Connect the agents through a shared workflow.

## 3. AI Agents

1. **REX Assistant:** Collects learner requirements and recommends suitable courses.
2. **RFP Research Agent:** Finds, extracts, and filters relevant educational tenders.
3. **Curriculum Agent:** Drafts structured curricula in English or Arabic, validates them, revises them as new versions, suggests new courses from the academy catalogue, compares curricula with approved competency frameworks (evidence-based, never an accreditation claim), and produces PDFs. An administrator approves them.
4. **Educational Document Generation Agent:** Generates professional educational PDFs.
5. **Marketing Agent:** Creates educational marketing and campaign content.

## 4. Technology Stack

- **Language:** Python
- **LLM runtime:** Ollama, model `qwen2.5:3b` (set `REX_LLM_MODEL` in `.env` to try another model for REX, `CURRICULUM_LLM_MODEL` for the Curriculum Agent)
- **Agent workflows:** LangGraph (state graphs) and LangChain (chat models, prompts, structured output)
- **Knowledge base (RAG):** ChromaDB, `all-MiniLM-L6-v2` embeddings, cross-encoder reranking, MarianMT Arabic→English query translation
- **Agent tools:** MCP (three servers: knowledge base for REX, tender tools for RFP Research, curriculum tools for the Curriculum Agent)
- **PDFs:** ReportLab with `arabic-reshaper` and `python-bidi` for Arabic, and bundled Noto fonts
- **External data:** World Bank Procurement Notices API, Tavily web search
- **User interface:** Streamlit (`rex_ai_app.py`), English and Arabic (RTL)
- **Testing:** pytest, Streamlit `AppTest`

## 5. Project Structure

```text
rex-ai/
├── agents/
│   ├── rex_assistant/        # Agent 1: course discovery and course Q&A
│   ├── rfp_research/         # Agent 2: tenders in Egypt and Saudi Arabia (World Bank API + web research), review queue, applications
│   ├── curriculum_research/  # Agent 3: curricula (LangGraph), validation, revisions, suggestions, alignment, PDFs, store
│   ├── orchestrator.py       # LangGraph router (Admin side; REX excluded)
│   ├── register_agents.py    # create_admin_orchestrator()
│   ├── user_service.py       # User interface backend: REX only
│   ├── admin_service.py      # Admin interface backend: passcode, sessions, tender review, applications, curriculum review
│   ├── mcp_tool_client.py    # shared MCP client code (all three MCP servers)
│   ├── rag_mcp_server.py     # MCP server exposing the knowledge base (REX)
│   ├── tender_mcp_server.py  # MCP server with the tender tools (RFP Research)
│   └── curriculum_mcp_server.py  # MCP server with the curriculum tools (Curriculum Agent)
├── knowledge_base/           # retrieval engine: ingestion, retrieval, Arabic query handling (REX writes the answers)
├── docs/architecture.md      # architecture, workflows, security, storage
├── docs/CURRICULUM_AGENT_ARCHITECTURE.md  # Curriculum Agent design, decisions and verification
├── tests/
├── rex_ai_app.py             # the application
└── requirements.txt
```

See [docs/architecture.md](docs/architecture.md) for the User and Admin workflows.

## 6. Setup

You need Python and [Ollama](https://ollama.com) installed. From the project root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull qwen2.5:3b
```

### Configuration (`.env`)

Create a file named `.env` in the project root (it is gitignored, so never commit it):

```text
TAVILY_API_KEY=your-tavily-key
REX_ADMIN_PASSCODE=choose-a-passcode
```

- Without `TAVILY_API_KEY`, RFP web research is skipped and the reply says so. World Bank research (Egypt) still works; Saudi Arabia needs web research.
- Without `REX_ADMIN_PASSCODE`, the Admin interface is disabled. The User interface still works.

## 7. Running

```powershell
python -m streamlit run rex_ai_app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`). The welcome screen has two choices:

- **I'm a learner:** chat with REX. Ask for a personalized recommendation (REX asks about your background, goal, level, online/offline preference, budget and time; any question can be skipped), or ask directly about a course. Write in English, Arabic or Egyptian Arabic.
- **I'm an admin:** enter the passcode to open the dashboard. From there you can:
  - run a **New tender search** (countries, training categories, priority organisations or sectors, eligibility notes, deadline range), or ask RFP Research or Curriculum Research in chat;
  - review every tender found (approve, reject, or back to pending), with all extracted fields, the source and submission links; Egypt tenders are flagged for immediate review, Saudi tenders are monitored;
  - see in-app alerts for new tenders;
  - browse the **Tender dashboard** of approved tenders and the audit trail;
  - ask the Curriculum Agent in chat (for example "Create an intermediate curriculum about SQL for business analysts, 24 hours", "make it 30 hours", "Suggest new courses for the Data Analytics department"), then review curricula in the **Curricula** tab: read each version, download its PDF, and approve it or request changes (the agent can never approve);
  - approving a tender starts its **application workflow**: REX AI shows how to apply (official portal steps, or a prepared expression-of-interest email for World Bank notices) and lists what is missing from the **Application profile** (the academy's real details and documents, entered by you). Nothing is ever sent: emails are dry runs, and a tender becomes *Applied* only when you record a verified submission.

The course catalogue is **synthetic demonstration data** for Andalusia Academy. Its courses, fees, schedules, instructors and offers are illustrative, not real.

### Knowledge base

```powershell
# 1. (Re)generate the demo catalogue documents from their structured source
python -m knowledge_base.datasets.andalusia_catalogue.build_documents

# 2. Ingest knowledge_base/documents/ into the vector index.
#    Safe to repeat: replaces changed documents, removes deleted ones, no duplicates.
python -m knowledge_base.pipeline

# 3. Check retrieval quality (Recall@5 and MRR on 32 questions)
python -m knowledge_base.evaluation --eval-set knowledge_base/datasets/andalusia_catalogue/eval_set.json
```

Stop the app before step 2, because the running app holds the index open. To use real academy data, put the files in `knowledge_base/documents/` and run step 2. Supported formats are PDF, DOCX, PPTX, XLSX, CSV, JSON, HTML, Markdown and TXT. See [docs/architecture.md](docs/architecture.md#knowledge-base-rag).

## 8. Testing

```powershell
python -m pytest -q
```

The test suite uses no network, no Ollama and no real database, and writes only to temporary files.
- **Mocked tests:** the chat model, the MCP server and Tavily.
- **Local integration tests:** the real parsers, chunker and pipeline with a real Chroma store in a temp folder. Embeddings are faked.
- **Web search** is offline in every test (`tests/conftest.py`).

Retrieval quality is measured separately by the evaluation command above.

## 9. Project Scope

The initial version will focus on demonstrating the core agent capabilities and establishing the architecture for multi-agent integration. Features will be implemented and tested incrementally.
