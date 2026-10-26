REX AI sample development datasets

Synthetic data for developing and testing the six-agent REX AI project before official academy data is available.

Files:
- courses.json: 20 fictional course examples. Fees are fictional and must not be shown as academy prices.
- curriculum_examples.json: 5 illustrative curricula, not accreditation-approved.
- document_templates.json: 5 document structure templates.
- marketing_examples.json: 10 draft examples, not approved for publication.

Use these to test routing, filtering, validation, workflow context, and document generation. Replace or supplement them with verified academy/public-source data when available. No RAG is needed; agents can load and filter these JSON files directly.

Note: Agent 2 (RFP/Tender Research) no longer uses a synthetic JSON fixture -- it retrieves real records from external tender APIs (see agents/rfp_research/api_clients/).
