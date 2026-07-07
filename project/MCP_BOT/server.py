"""Banking transactions — MCP server (delivering Philosophy 2).

Serves the **exact same three tools** as `LLM_BOT/main_tools.py`, but over the
**Model Context Protocol** — so any MCP client (Claude Desktop, Claude Code, …)
can use them, not just our CLI. The tool bodies are reused verbatim; only the
delivery changes: **same code, different client.**

Identity is chosen by the **human**, never the model — resolved as the
`.current_customer` file (set with `login.py`) → `BANK_CUSTOMER_ID` env → default.
The model can't write that file or pass a `customer_id`, so it can never switch
who it is.

Run:  BANK_CUSTOMER_ID=CUST031 python server.py   (switch: python login.py CUST007)
"""
import functools
import importlib.util
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# --- reuse the exact deterministic tool bodies from main_tools.py ----------
_MAIN_TOOLS = Path(__file__).resolve().parent.parent / "LLM_BOT" / "main_tools.py"
_spec = importlib.util.spec_from_file_location("main_tools", _MAIN_TOOLS)
bot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bot)

# The file where the human records "who I am". The model can't touch it.
_SESSION_FILE = Path(__file__).resolve().parent / ".current_customer"


def _current_customer() -> str:
    """The authenticated customer — chosen by the human, never the model."""
    if _SESSION_FILE.exists():
        who = _SESSION_FILE.read_text().strip()
        if who:
            return who
    return os.environ.get("BANK_CUSTOMER_ID", "CUST001").strip()


def _scoped(fn):
    """Pin the tool to the current (human-chosen) customer before each call.

    The customer is read fresh from _current_customer() on every call, so
    `login.py` takes effect immediately — no restart. `customer_id` is still not
    a tool argument, so the model cannot override it.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        bot.CURRENT_CUSTOMER = _current_customer()
        return fn(*args, **kwargs)

    return wrapper


# --- serve those same functions over MCP -----------------------------------
mcp = FastMCP("banking-transactions")
mcp.add_tool(_scoped(bot.query_spending.func))
mcp.add_tool(_scoped(bot.explain_transaction.func))
mcp.add_tool(_scoped(bot.file_dispute.func))


if __name__ == "__main__":
    print(
        f"[banking-transactions MCP] current customer: {_current_customer()} "
        f"(switch with: python login.py CUST007)",
        file=sys.stderr,
    )
    mcp.run()  # stdio transport (the default MCP transport)
