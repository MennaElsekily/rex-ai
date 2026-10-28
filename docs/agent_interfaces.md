
# REX AI – Agent Interfaces and Data Structures

## 1. Purpose

This document defines the common request and response structures used by the five REX AI agents. These structures will help the orchestrator route requests and handle agent results consistently.

## 2. Common Request Structure

Each agent receives a request containing:

| Field | Description |
|---|---|
| request_id | Unique identifier for the request |
| agent_name | Name of the agent handling the request |
| user_input | The user's request or instructions |
| context | Relevant information needed to process the request |

## 3. Common Response Structure

Each agent returns a response containing:

| Field | Description |
|---|---|
| request_id | Identifier linking the response to its request |
| agent_name | Name of the responding agent |
| status | Result status, such as success or error |
| result | The generated answer or output |
| error_message | Error details when processing fails |

## 4. Agent Responsibilities and Outputs

### 4.1 REX Assistant
- **Input:** Learner background, career goals, skill level, learning preferences, budget, and available study time.
- **Output:** Relevant course recommendations.

### 4.2 RFP Research Agent
- **Input:** Search criteria and filters for educational tenders and RFPs.
- **Output:** Extracted and filtered tender information relevant to Egypt and Saudi Arabia.

### 4.3 Curriculum Agent
- **Input:** Curriculum requirements, learning objectives, and relevant accreditation criteria.
- **Output:** Curriculum recommendations and accreditation gap analysis.

### 4.4 Educational Document Generation Agent
- **Input:** Document type, educational content, and formatting requirements.
- **Output:** A generated educational document in PDF format.

### 4.5 Marketing Agent
- **Input:** Campaign objectives, target audience, and marketing requirements.
- **Output:** Educational marketing and campaign content.

## 5. Orchestrator Responsibilities

The orchestrator:
1. Receives user requests.
2. Selects the appropriate agent or agents.
3. Passes requests using the common request structure.
4. Collects agent responses.
5. Returns results to the user.

## 6. Implementation Status

These structures define the planned interfaces. They may be refined as the agents are implemented and tested.
