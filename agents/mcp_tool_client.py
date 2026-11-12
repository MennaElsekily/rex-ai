"""
Shared MCP client: one persistent connection from an agent to one MCP
server, used by every agent-specific client:

    agents/rex_assistant/mcp_client.py  RagMcpClient     -> agents/rag_mcp_server.py
    agents/rfp_research/mcp_client.py   TenderMcpClient  -> agents/tender_mcp_server.py

An MCP server is a separate Python process reached over stdio (its
stdin/stdout). Starting it costs seconds (imports, database connection),
so this client starts the process once and keeps the MCP session open
across calls. The MCP library is asynchronous; the agents' LangGraph
nodes are synchronous, so the session runs in its own background thread
with its own event loop, and call_tool() waits for the result.

If the server process dies, the failed call raises McpToolError and the
next call starts a fresh server.

Every tool returns one JSON object; call_tool() parses it into a dict.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import sys
import threading
import weakref
from pathlib import Path
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class McpToolError(RuntimeError):
    """Raised when the MCP tool call itself cannot be completed."""


def _drop_none(arguments: dict[str, Any]) -> dict[str, Any]:
    """Omit unset optional arguments so the server's defaults apply."""
    return {key: value for key, value in arguments.items() if value is not None}


# Every client this process created and has not closed. When the application exits, atexit closes them, which ends the
# server subprocesses this process started (and only those: a server another process or the user started is never touched).
_OPEN_CLIENTS: "weakref.WeakSet[McpToolClient]" = weakref.WeakSet()


def close_all_clients() -> None:
    """Close every MCP client of this process. Safe to call more than once."""
    for client in list(_OPEN_CLIENTS):
        try:
            client.close()
        except Exception:  # one client failing to close must not keep the others' servers running
            pass


atexit.register(close_all_clients)


class McpToolClient:
    """Keeps one MCP connection to a server module alive; call_tool() is synchronous."""

    def __init__(
        self,
        server_module: str,
        startup_timeout: float = 120.0,
        server_name: str = "MCP",
        env: Optional[dict[str, str]] = None,
    ) -> None:
        """
        server_module: the server, run as "python -m <server_module>".
        server_name: used in error messages only.
        env: extra environment variables for the server process. The MCP
            SDK passes only a small safe set of this process's variables
            (PATH, TEMP, ...) plus these; servers read other settings from
            .env (agents.config.load_env).
        """
        self._server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", server_module],
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        self._startup_timeout = startup_timeout
        self._server_name = server_name

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session: Optional[ClientSession] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._ready_event = threading.Event()
        self._startup_error: Optional[BaseException] = None
        _OPEN_CLIENTS.add(self)

    def start(self) -> None:
        """
        Start the server subprocess and open the MCP session, unless a
        live session already exists. Also restarts after close(), after
        a failed start, and after the server process has died.
        """
        if (
            self._thread is not None
            and self._thread.is_alive()
            and self._session is not None
        ):
            return

        self._ready_event = threading.Event()
        self._startup_error = None
        self._session = None

        self._thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
        )
        self._thread.start()

        if not self._ready_event.wait(timeout=self._startup_timeout):
            raise McpToolError(
                f"Timed out starting the {self._server_name} MCP server subprocess."
            )

        if self._startup_error is not None:
            self._thread = None
            raise McpToolError(
                f"Failed to start the {self._server_name} MCP server: {self._startup_error}"
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
        finally:
            # The session is gone (closed or the server died); the next
            # call starts a fresh one.
            self._session = None

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Run one tool call and return its JSON result as a dict. Unset
        (None) arguments are left out so the server's defaults apply.
        Starts or restarts the server when needed.
        """
        return self._run(self._call_tool, tool_name, _drop_none(arguments))

    def list_tools(self) -> list[dict[str, Any]]:
        """The server's tools: name, description and input schema."""
        return self._run(self._list_tools)

    def _run(self, coroutine_function, *args) -> Any:
        # The coroutine is created only once the session exists, so a
        # failed start leaves nothing un-awaited.
        self.start()

        if self._session is None or self._loop is None:
            raise McpToolError("MCP session is not available.")

        future = asyncio.run_coroutine_threadsafe(coroutine_function(*args), self._loop)

        try:
            return future.result(timeout=self._startup_timeout)
        except McpToolError:
            raise
        except TimeoutError as exc:
            # str(TimeoutError()) is empty: say what timed out. The server is stopped, so the next call starts
            # a fresh one (a first call may be slow because the server loads its models on first use).
            what = f"the {args[0]} tool" if args and isinstance(args[0], str) else "the request"
            self.close()
            raise McpToolError(
                f"The {self._server_name} MCP server did not answer {what} within "
                f"{self._startup_timeout:g} seconds; it was stopped."
            ) from exc
        except Exception as exc:
            # Transport failure (e.g. the server process died): drop this
            # session so the next call starts a fresh server.
            self.close()
            raise McpToolError(f"MCP tool call failed: {exc}") from exc

    async def _call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        result = await self._session.call_tool(tool_name, arguments)

        text_parts = [
            item.text for item in result.content if hasattr(item, "text")
        ]

        if getattr(result, "is_error", False):
            # Rejected by MCP itself (e.g. arguments that don't match the
            # tool's schema) or the tool raised: the text is the reason.
            raise McpToolError(
                f"The {tool_name} tool reported an error: {' '.join(text_parts)[:500]}"
            )

        if not text_parts:
            raise McpToolError(
                f"The {tool_name} tool returned no text content."
            )

        try:
            parsed = json.loads(text_parts[0])
        except json.JSONDecodeError as exc:
            raise McpToolError(
                f"Could not parse MCP tool response as JSON: {exc}"
            ) from exc

        if not isinstance(parsed, dict):
            raise McpToolError(
                f"The {tool_name} tool returned an unexpected format."
            )

        return parsed

    async def _list_tools(self) -> list[dict[str, Any]]:
        result = await self._session.list_tools()
        return [
            {"name": tool.name, "description": tool.description or "",
             "input_schema": tool.input_schema}
            for tool in result.tools
        ]

    def close(self) -> None:
        """Stop the server subprocess and background thread. Safe to call more than once, and when the server already died."""
        loop, stop_event, thread = self._loop, self._stop_event, self._thread
        if loop is not None and stop_event is not None:
            try:
                loop.call_soon_threadsafe(stop_event.set)
            except RuntimeError:  # the loop is already closed: the session has ended by itself
                pass

        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=10)

        self._session = None
        self._thread = None
        self._loop = None
