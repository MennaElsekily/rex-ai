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
3. **Curriculum Agent:** Supports curriculum development, research, and accreditation gap analysis.
4. **Educational Document Generation Agent:** Generates professional educational PDFs.
5. **Marketing Agent:** Creates educational marketing and campaign content.

## 4. Technology Stack

- **Programming language:** Python
- **LLM runtime:** Ollama
- **Initial language model:** Llama 3.2 3B
- **User interface:** Streamlit
- **Testing:** pytest
- **Version control:** Git and GitHub

Additional technologies for retrieval, data storage, document generation, and agent integration will be selected as the project develops.

## 5. Project Structure

```text
rex-ai/
├── agents/
├── data/
├── docs/
├── knowledge_base/
├── tests/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 6. Development Status

The project is currently in its initial development stage. Sprint 1 focuses on project planning, repository setup, architecture, agent interfaces, initial knowledge-base preparation, and a basic REX Assistant prototype.

## 7. Setup

Python and Ollama must be installed before running the application.

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The initial model can be downloaded through Ollama using:

```bash
ollama pull llama3.2:3b
```

## Running the REX Assistant

### Prerequisites
- Python installed.
- Ollama installed and running.
- The `llama3.2:3b` model downloaded.

### Setup

Open PowerShell in the project root and activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
pip install -r requirements.txt
```

Make sure the required Ollama model is available:

```powershell
ollama pull llama3.2:3b
```

### Start the Application

Run the following command from the project root:

```powershell
python -m streamlit run app.py
```

Streamlit will provide a local URL, usually `http://localhost:8501`. Open that URL in your browser to chat with REX Assistant.

### Current Prototype

The current prototype provides a chat interface, uses a local language model through Ollama, and uses the course data in `data/courses.json` to guide course recommendations. The available course records are sample data for initial development.
## 8. Testing

Run the project tests with:

```bash
python -m pytest -q
```

## 9. Project Scope

The initial version will focus on demonstrating the core agent capabilities and establishing the architecture for multi-agent integration. Features will be implemented and tested incrementally.
