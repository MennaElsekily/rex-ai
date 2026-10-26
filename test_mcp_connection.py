import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_ROOT = Path(__file__).resolve().parent

server_params = StdioServerParameters(
    command=sys.executable,
    args=["-m", "agents.rag_mcp_server"],
    cwd=str(PROJECT_ROOT),
)


async def main():
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            print(
                "Available MCP tools:",
                [tool.name for tool in tools_result.tools],
            )

            result = await session.call_tool(
                "ask_knowledge_base",
                {
                    "question": (
                        "What programmes are available in artificial "
                        "intelligence and machine learning?"
                    )
                },
            )

            print("\nMCP tool call result:")
            for item in result.content:
                if hasattr(item, "text"):
                    print(item.text)


if __name__ == "__main__":
    asyncio.run(main())
