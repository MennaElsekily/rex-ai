"""A tiny MCP server for tests/test_mcp_cleanup.py: it writes its process id to the file named by PID_FILE and offers one tool."""

import json
import os

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("pid test server")


@mcp.tool()
def ping() -> str:
    return json.dumps({"pid": os.getpid()})


if __name__ == "__main__":
    with open(os.environ["PID_FILE"], "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    mcp.run()
