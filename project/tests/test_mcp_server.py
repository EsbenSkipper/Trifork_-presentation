"""The MCP server exposes the same tools — and preserves isolation.

Verifies that server.py registers exactly the three tools over MCP and, crucially,
that `customer_id` is NOT a tool argument (the server owns the customer identity;
the connecting model can't choose it).
"""
import asyncio
import importlib.util
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_server():
    os.environ.setdefault("BANK_CUSTOMER_ID", "CUST001")
    spec = importlib.util.spec_from_file_location("mcp_server_under_test", ROOT / "MCP_BOT" / "server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mcp_exposes_the_three_tools_without_customer_id():
    srv = _load_server()
    tools = asyncio.run(srv.mcp.list_tools())
    names = {t.name for t in tools}
    assert names == {"query_spending", "explain_transaction", "file_dispute"}
    for tool in tools:
        props = tool.inputSchema.get("properties", {})
        assert "customer_id" not in props and "customerId" not in props


def test_mcp_tool_call_returns_the_same_pandas_figure():
    srv = _load_server()
    # force identity to CUST031 without touching a real session file
    srv._SESSION_FILE = pathlib.Path("/tmp/__no_such_session__")
    os.environ["BANK_CUSTOMER_ID"] = "CUST031"
    result = asyncio.run(srv.mcp.call_tool("query_spending", {"aggregation": "sum", "category": "Groceries"}))
    blocks = result[0] if isinstance(result, tuple) else result.content
    text = blocks[0].text
    assert "14683.23" in text  # CUST031's groceries total


def test_login_switches_identity_but_the_model_cannot(tools=None):
    # the human's session file changes who the tools query; there is no tool that
    # lets the model change it (no customer_id argument, no login tool).
    srv = _load_server()
    tmp = pathlib.Path("/tmp/__mcp_session_test__")
    srv._SESSION_FILE = tmp
    try:
        tmp.write_text("CUST002\n")
        assert srv._current_customer() == "CUST002"
        tmp.write_text("CUST003\n")
        assert srv._current_customer() == "CUST003"
    finally:
        tmp.unlink(missing_ok=True)
    tool_names = {t.name for t in asyncio.run(srv.mcp.list_tools())}
    assert "login" not in tool_names and "set_customer" not in tool_names