"""
MCP server subprocesses are closed when the application exits and by close(): idempotent, only the subprocesses this
process started, and never a process it does not own. Uses a tiny real MCP server (tests/pid_mcp_server.py) that records its pid.
"""

import asyncio
import gc
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from agents import mcp_tool_client
from agents.mcp_tool_client import McpToolClient, close_all_clients

ROOT = Path(__file__).resolve().parent.parent
SERVER = "tests.pid_mcp_server"


def alive(pid: int) -> bool:
    if os.name == "nt":                                                                                      # os.kill would terminate it there
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def wait_gone(pid: int, seconds: float = 15) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.2)
    return not alive(pid)


def make(tmp_path, name="a"):
    pid_file = tmp_path / f"{name}.pid"
    env = {**os.environ, "PID_FILE": str(pid_file)}
    return McpToolClient(SERVER, startup_timeout=60, server_name="test", env=env), pid_file


def server_pid(client, pid_file) -> int:
    client.start()
    return int(pid_file.read_text())


def test_close_ends_the_subprocess_and_is_idempotent(tmp_path):
    client, pid_file = make(tmp_path)
    pid = server_pid(client, pid_file)
    assert alive(pid)

    client.close()
    client.close()                                                                                           # a second close is harmless
    assert wait_gone(pid)
    assert client._thread is None and client._session is None


def test_a_client_can_be_used_again_after_close(tmp_path):
    client, pid_file = make(tmp_path)
    first = server_pid(client, pid_file)
    client.close()
    assert wait_gone(first)
    second = client.call_tool("ping", {})["pid"]                                                             # a fresh server
    assert second != first and alive(second)
    client.close()
    assert wait_gone(second)


def test_closing_a_client_that_never_started_or_whose_loop_ended_does_nothing(tmp_path):
    never, _ = make(tmp_path)
    never.close()
    never.close()

    dead, _ = make(tmp_path, "b")
    dead._loop = asyncio.new_event_loop()
    dead._loop.close()                                                                                       # the server's loop already ended
    dead._stop_event = asyncio.Event()
    dead.close()
    assert dead._loop is None


def test_close_all_clients_closes_this_processes_clients_but_not_unrelated_processes(tmp_path, monkeypatch):
    one, one_file = make(tmp_path, "one")
    two, two_file = make(tmp_path, "two")
    pids = [server_pid(one, one_file), server_pid(two, two_file)]
    broken, _ = make(tmp_path, "broken")
    monkeypatch.setattr(broken, "close", lambda: (_ for _ in ()).throw(RuntimeError("boom")))                # must not stop the others

    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])                     # not ours: must survive
    try:
        close_all_clients()
        close_all_clients()
        assert all(wait_gone(pid) for pid in pids)
        assert unrelated.poll() is None and alive(unrelated.pid)
    finally:
        unrelated.kill()
        unrelated.wait()


def test_a_client_that_is_no_longer_referenced_is_not_kept_alive_by_the_registry(tmp_path):
    import weakref

    client, _ = make(tmp_path)
    assert client in mcp_tool_client._OPEN_CLIENTS
    probe = weakref.ref(client)
    del client
    gc.collect()
    assert probe() is None


def test_the_server_subprocess_is_gone_after_the_application_process_exits(tmp_path):
    """A separate Python process starts a client, never closes it and exits normally: the atexit hook ends the server."""
    script = textwrap.dedent("""
        import os
        from agents.mcp_tool_client import McpToolClient
        client = McpToolClient("tests.pid_mcp_server", startup_timeout=60, server_name="test", env=dict(os.environ))
        client.start()                                                       # no close(): the process just ends
    """)
    pid_file = tmp_path / "exit.pid"
    done = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env={**os.environ, "PID_FILE": str(pid_file)},
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-800:]
    pid = int(pid_file.read_text())
    assert wait_gone(pid), "the server subprocess outlived the application"
