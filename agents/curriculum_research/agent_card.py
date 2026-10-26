
AGENT_CARD = {
    "name": "curriculum_research",
    "display_name": "Curriculum Research & Accreditation Alignment Agent",
    "description": (
        "Develops curriculum outlines, learning outcomes, modules, "
        "assessments, and preliminary gap analyses against supplied standards."
    ),
    "supported_languages": ["en", "ar"],
    "capabilities": [
        "search_curriculum_examples",
        "retrieve_curriculum_details",
        "draft_curriculum_outlines",
        "review_learning_outcomes",
        "compare_against_supplied_standards",
    ],
    "data_sources": [
        "data/curriculum_examples.json",
        "verified standards supplied by the user or retrieved through RAG",
    ],
    "limitations": [
        "Bundled curriculum examples are synthetic.",
        "Examples are not verified accreditation standards.",
        "Accreditation claims require verified evidence.",
        "Live research and RAG integration are not implemented here yet.",
    ],
}
