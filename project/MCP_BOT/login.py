"""Choose which customer 'you' are for the MCP server.

This is the human logging in — it writes a small session file the MCP server
reads on its next tool call. No restart of Claude Desktop needed; just ask your
next question. The model can't run this or change the file, so it can never
switch customers on its own.

    python login.py            # show who you are now
    python login.py CUST007    # become CUST007
"""
import sys
from pathlib import Path

SESSION_FILE = Path(__file__).resolve().parent / ".current_customer"


def _known_customers() -> set[str]:
    """The valid ids, read straight from the dataset (best-effort)."""
    try:
        import csv

        csv_path = Path(__file__).resolve().parent.parent / "LLM_BOT" / "transactions_sample.csv"
        with open(csv_path) as f:
            return {row["customerId"].strip() for row in csv.DictReader(f) if row.get("customerId")}
    except Exception:
        return set()


def main() -> None:
    if len(sys.argv) < 2:
        current = SESSION_FILE.read_text().strip() if SESSION_FILE.exists() else "(default / BANK_CUSTOMER_ID)"
        print(f"You are currently: {current}")
        print("Switch with:  python login.py CUST007")
        return

    customer = sys.argv[1].strip().upper()
    known = _known_customers()
    if known and customer not in known:
        print(f"Warning: {customer} is not in the dataset — queries will come back empty.")

    SESSION_FILE.write_text(customer + "\n")
    print(f"You are now {customer}. Ask Claude Desktop again — no restart needed.")


if __name__ == "__main__":
    main()