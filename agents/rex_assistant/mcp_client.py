"""
Persistent MCP client for Agent 1 to reach the existing RAG knowledge
base through agents/rag_mcp_server.py, instead of importing
knowledge_base.answer_generator directly.

The MCP server process loads the embedding model, cross-encoder,
translator, and ChromaDB connection on startup (several seconds,
confirmed by manual testing). Opening a new stdio connection per
conversational turn would reload all of that on every message, so this
client starts the server subprocess once and keeps the MCP session
open across calls, running the async MCP client in a dedicated
background thread so the rest of Agent 1 (LangGraph nodes) can call it
synchronously.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class McpToolError(RuntimeError):
    """Raised when the MCP tool call itself cannot be completed."""


class RagMcpClient:
    """
    Keeps one MCP connection to agents/rag_mcp_server.py alive and
    exposes the ask_knowledge_base tool as a synchronous method.
    """

    def __init__(
        self,
        server_module: str = "agents.rag_mcp_server",
        startup_timeout: float = 120.0,
    ) -> None:
        self._server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", server_module],
            cwd=str(PROJECT_ROOT),
        )
        self._startup_timeout = startup_timeout

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session: Optional[ClientSession] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._ready_event = threading.Event()
        self._startup_error: Optional[BaseException] = None

    def start(self) -> None:
        """Start the server subprocess and open the MCP session once."""
        if self._thread is not None:
            return

        self._thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
        )
        self._thread.start()

        if not self._ready_event.wait(timeout=self._startup_timeout):
            raise McpToolError(
                "Timed out starting the RAG MCP server subprocess."
            )

        if self._startup_error is not None:
            raise McpToolError(
                f"Failed to start the RAG MCP server: {self._startup_error}"
            ) from self._startup_error

    def _run_event_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._session_lifecycle())
        finally:
            self._loop.close()

    async def _session_lifecycle(self) -> None:
        self._stop_event = asyncio.Event()

        try:
            async with stdio_client(self._server_params) as (
                read_stream,
                write_stream,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    self._session = session
                    self._ready_event.set()

                    await self._stop_event.wait()
        except BaseException as exc:  # noqa: BLE001
            self._startup_error = exc
            self._ready_event.set()

    def ask_knowledge_base(
        self,
        question: str,
        retrieval_query: Optional[str] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
        preferred_language: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Call the ask_knowledge_base MCP tool and return its parsed
        result. Starts the server connection on first use.
        """
        self.start()

        if self._session is None or self._loop is None:
            raise McpToolError("MCP session is not available.")

        arguments: dict[str, Any] = {"question": question}

        if retrieval_query is not None:
            arguments["retrieval_query"] = retrieval_query
        if metadata_filter is not None:
            arguments["metadata_filter"] = metadata_filter
        if preferred_language is not None:
            arguments["preferred_language"] = preferred_language

        future = asyncio.run_coroutine_threadsafe(
            self._call_tool(arguments),
            self._loop,
        )

        return future.result(timeout=self._startup_timeout)

    async def _call_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self._session.call_tool(
            "ask_knowledge_base",
            arguments,
        )

        text_parts = [
            item.text for item in result.content if hasattr(item, "text")
        ]

        if not text_parts:
            raise McpToolError(
                "The ask_knowledge_base tool returned no text content."
            )

        try:
            parsed = json.loads(text_parts[0])
        except json.JSONDecodeError as exc:
            raise McpToolError(
                f"Could not parse MCP tool response as JSON: {exc}"
            ) from exc

        if not isinstance(parsed, dict):
            raise McpToolError(
                "The ask_knowledge_base tool returned an unexpected format."
            )

        return parsed

    def close(self) -> None:
        """Stop the server subprocess and background thread."""
        if self._loop is not None and self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._stop_event.set)

        if self._thread is not None:
            self._thread.join(timeout=10)

        self._session = None
        self._thread = None
        self._loop = None
