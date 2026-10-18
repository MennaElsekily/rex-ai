
# REX AI – System Architecture

## Architecture Diagram


```mermaid
flowchart TB
    U[User] --> UI[User Interface]
    UI --> O[Orchestrator / Router]

    subgraph Agents["Specialized AI Agents"]
        direction LR
        A1[REX Assistant]
        A2[RFP Research Agent]
        A3[Curriculum Agent]
        A4[Document Generation Agent]
        A5[Marketing Agent]
    end

    O --> A1
    O --> A2
    O --> A3
    O --> A4
    O --> A5

    KB[Shared Knowledge Base]
    LLM[Large Language Model]
    DB[(Database)]
    EXT[External Services]
    LOG[Logging and Error Handling]

    A1 --> KB
    A2 --> KB
    A3 --> KB
    A4 --> KB
    A5 --> KB

    O -.-> LLM
    O -.-> DB
    O -.-> EXT
    O -.-> LOG
```


## Overview

REX AI uses an orchestrator to route user requests to five specialized AI agents. The agents use the large language model and shared knowledge base to support their tasks. The architecture also includes a database, external services, and logging and error handling.

## Request Flow

1. The user submits a request through the interface.
2. The orchestrator routes the request to the appropriate agent.
3. The agent processes the request using the LLM and relevant knowledge.
4. The agent accesses other services when needed.
5. The result is returned to the user.

## Implementation Status

This document describes the planned architecture. Components and integrations will be implemented incrementally.
