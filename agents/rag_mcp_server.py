
from typing import Any, Optional

from mcp.server.mcpserver import MCPServer

from knowledge_base.answer_generator import RAGAnswerGenerator


mcp = MCPServer("REX AI Knowledge Base")

rag = RAGAnswerGenerator()


@mcp.tool()
def ask_knowledge_base(
    question: str,
    retrieval_query: Optional[str] = None,
    metadata_filter: Optional[dict[str, Any]] = None,
    preferred_language: Optional[str] = None,
) -> dict:
    """
    Answer a question using REX AI's document knowledge base.

    Args:
        question: The user's original question, in Arabic or English.
        retrieval_query: Optional pre-built search query to use for
            retrieval instead of the question itself (e.g. a
            recommendation query assembled from a learner profile).
        metadata_filter: Optional filter over indexed chunk metadata
            fields (only applied when the referenced fields exist).
        preferred_language: Optional "ar" or "en" override for the
            response language, independent of the question's language.

    Returns an answer generated from retrieved documents, with source
    references. Use this tool whenever an agent needs information from
    the knowledge base rather than calling the RAG components directly.
    """
    try:
        return rag.answer(
            question,
            retrieval_query=retrieval_query,
            metadata_filter=metadata_filter,
            preferred_language=preferred_language,
        )
    except ValueError as exc:
        return {
            "error": str(exc),
            "answer": "",
            "sources": [],
        }
    except Exception:
        return {
            "error": "The knowledge-base request failed.",
            "answer": "",
            "sources": [],
        }


if __name__ == "__main__":
    mcp.run()
