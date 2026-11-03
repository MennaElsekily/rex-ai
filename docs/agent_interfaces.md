
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
- **Status:** implemented (`agents/document_generation/`, see docs/architecture.md). Nine generated types: `course_description`, `course_proposal`, `participant_guide`, `capstone_brief`, `outcomes_assessment_plan`, `instructor_guide`, `presentation_outline`, `question_bank`, `assessment_rubric`. The tenth required document, the curriculum PDF, is not generated: the admin downloads the Curriculum Agent's PDF of the approved version (`AdminService.approved_curriculum_pdf`).
- **Request:** `context["document_request"] = {curriculum_id, document_type, language?, extra?, document_id?}` or a chat request. `context["progress"]` may be a callable `(done, total, label)` that is told how far a multi-module document has got. The curriculum must be approved; the document is written in the curriculum's language (English or Arabic).
- **Result:** `{kind: "document", document_id, version, review_status, curriculum_id, curriculum_version, missing, validation, pdf}`. Errors use the common `status: "error"` with `error_message`; `result.kind` names the failure, for example `validation_failed` (the text still failed the checks after the repair attempts: one, or two for the question bank, so nothing was saved), `generation_failed` (the model failed) or `not_approved`.
- **Review:** a generated document is `pending_review`; only an administrator approves a version (`AdminService.approve_document`), which makes the approved PDF. Edits create new versions; approved versions are never overwritten.

### 4.5 Marketing Agent
- **Input:** Campaign objectives, target audience, and marketing requirements.
- **Output:** Educational marketing and campaign content.
- **Status:** implemented (`agents/marketing/`, see docs/architecture.md). It writes campaigns only for an *approved* course (read through its own MCP server, `agents/marketing_mcp_server.py`), never from an unapproved draft: in a "curriculum, then marketing" plan the draft in `previous_results` is ignored and the agent asks for an approved course.
- **Request:** `context["marketing_request"] = {curriculum_id, audience?, tone?, goal?, notes?, channels?, variations?, campaign_id?}` or a chat request ("Write a social media campaign for approved course CUR-..."). `audience` confirms or overrides the course's target audience; `tone` is `professional` (default), `friendly`, `inspiring` or `formal`; `channels` is any of `social_captions`, `ad_copy`, `linkedin_posts`, `email`, `whatsapp`, `headlines`, `ctas` (default all seven); `variations` is 1-3 (default 2); `campaign_id` writes a new version of that campaign. `context["progress"]` may be a callable `(done, total, label)`. The campaign is written in the course's language (English or Arabic) and is not translated.
- **Result:** `{kind: "campaign", campaign_id, version, review_status: "pending_review", curriculum_id, curriculum_version, audience, tone, variations, channels, strategy, validation}`. Questions come back as `status: "success"` with `kind: "clarify"` (which course?) or `kind: "need_audience"`. Errors use the common `status: "error"` with `error_message`; `result.kind` is one of `not_approved`, `not_found`, `invalid`, `missing_information`, `generation_failed` or `validation_failed` (a content type still failed the checks after its one repair, so nothing was saved; `failed_content` names it).
- **Review:** a generated campaign is `pending_review`; only an administrator approves a version (`AdminService.approve_campaign`) and only the approved version can be downloaded (Markdown or JSON, `AdminService.campaign_export`). Edits create new versions. Nothing is published, e-mailed or sent by any agent or tool.

## 5. Orchestrator Responsibilities

The orchestrator:
1. Receives user requests.
2. Selects the appropriate agent or agents.
3. Passes requests using the common request structure.
4. Collects agent responses.
5. Returns results to the user.

## 6. Implementation Status

These structures define the interfaces. All five agents are implemented and registered (REX Assistant on the User side; RFP Research, Curriculum, Document Generation and Marketing behind the Admin orchestrator), and the structures were refined as they were built; sections 4.4 and 4.5 describe the Document Generation and Marketing agents as implemented.

Which agents a chat request reaches is decided by the deliverables it asks for, not by the subjects it mentions (`agents/routing.py`): "Create a curriculum about digital marketing" is a curriculum request, "Create a social media campaign for approved course CUR-123" a marketing one, and a request that asks for several deliverables runs each requested agent in order. A follow-up question an agent asked (for example the campaign audience) is tied to the original request, expires after 15 minutes and is dropped by "cancel" or by a new request.
