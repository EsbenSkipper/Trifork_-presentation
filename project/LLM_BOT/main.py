"""LLM-Bot (Claude edition).

A tiny, end-to-end banking assistant over a CSV of card transactions. Same idea
as the original Azure/ChatGPT version — only the model provider changed to
**Claude** (Anthropic).

The whole design is one split:

    semantic / language work        -> the LLM (Claude)
    money, rules, customer isolation -> deterministic Python (pandas)

Claude does three things, each returning a **typed, structured object** we can
track as it flows through the program:

    classify_intent(...)    -> Intent + a normalized question
    parse_spend_query(...)  -> a structured SpendQuery (filters + aggregation)
    resolve_merchant(...)   -> one real merchantCode (fixes fuzzy names)

Everything else — filtering to one customer, summing, the 90-day / no-ATM
dispute rules, the final JSON — stays in Python so it is exact and safe.

Run:  export ANTHROPIC_API_KEY=sk-ant-...   then   python main.py
(No key? It still runs, using a simple deterministic fallback classifier.)
"""
import json
import os
import re
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import lru_cache
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, Field

# ============================================================
# 1. CONFIG
# ============================================================
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")
CSV_PATH = os.path.join(os.path.dirname(__file__), "transactions_sample.csv")
DISPUTE_WINDOW_DAYS = 90

# Grounded, human-friendly merchant descriptions for the "explain" path.
# (Only these get a nice explanation; anything else is treated as unknown.)
MERCHANT_MAP = {
    "MC5678 AAR ICA": "ICA supermarket in Aarhus (Groceries)",
    "MC2468 ONLINE NETFLX": "Netflix subscription (online recurring charge)",
    "MC1234 OSL REMA1000": "Rema 1000 supermarket, Oslo (Groceries)",
    "MC4321 CPH MCDON": "McDonald's, Copenhagen (FastFood)",
    "MC1357 OSL NORDNET": "Nordnet broker, Oslo (Investment)",
    "MC7777 ATM DNB": "ATM withdrawal, DNB (ATM, non-disputable)",
    "MC9999 OSL BMW": "BMW car dealership, Oslo (Auto)",
}

# The only four dispute reasons the assignment allows.
ALLOWED_DISPUTE_REASONS = [
    "Fraudulent transaction",
    "Duplicate charge",
    "Goods/services not received",
    "Wrong amount charged",
]

# In-memory, per-customer state (sticky customer flow).
LAST_TX_BY_CUSTOMER: dict[str, dict[str, Any]] = {}
DISPUTE_SESSIONS: dict[str, dict[str, Any]] = {}


# ============================================================
# 2. STRUCTURED SHAPES  (what Claude must return)
# ============================================================
class Intent(str, Enum):
    SPEND = "spend"
    EXPLAIN = "explain"
    DISPUTE = "dispute"


class IntentResult(BaseModel):
    intent: Intent
    normalized_question: str = Field(description="Typo-fixed, filler-free version of the question.")


class Aggregation(str, Enum):
    SUM = "sum"
    COUNT = "count"
    TOP_N = "top_n"
    GROUP_BY_CATEGORY = "group_by_category"
    MIN_DATE = "min_date"
    MAX_DATE = "max_date"


class DateRange(BaseModel):
    date_from: str = Field(description="YYYY-MM-DD, inclusive")
    date_to: str = Field(description="YYYY-MM-DD, inclusive")


class RelationalFilters(BaseModel):
    """Deterministic filters — map 1:1 to columns."""

    date_from: Optional[str] = None
    date_to: Optional[str] = None
    date_ranges: list[DateRange] = Field(default_factory=list)
    last_n_days: Optional[int] = None
    categories: list[str] = Field(default_factory=list, description="Exact values from the given list.")
    merchant_code_exact: Optional[str] = None


class SemanticFilters(BaseModel):
    """Fuzzy hints, resolved in a second pass."""

    merchant_text: Optional[str] = Field(default=None, description="e.g. 'netflix', 'that aarhus shop'")


class AggregationSpec(BaseModel):
    type: Aggregation
    n: Optional[int] = Field(default=None, description="only for top_n")


class SpendQuery(BaseModel):
    relational_filters: RelationalFilters = Field(default_factory=RelationalFilters)
    semantic_filters: SemanticFilters = Field(default_factory=SemanticFilters)
    aggregation: AggregationSpec


class MerchantMatch(BaseModel):
    match: str = Field(description="One merchantCode from the candidates, or 'NONE'.")


# ============================================================
# 3. CLAUDE CLIENT  (the only place we call the model)
# ============================================================
@lru_cache(maxsize=1)
def _client():
    """Return an Anthropic client, or None if no key is configured."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic

    return anthropic.Anthropic()


def llm_enabled() -> bool:
    return _client() is not None


def claude_parse(system: str, user: str, schema: type[BaseModel]):
    """Ask Claude for a validated structured object. Raises on any failure."""
    client = _client()
    if client is None:
        raise RuntimeError("No ANTHROPIC_API_KEY set.")
    resp = client.messages.parse(
        model=MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_format=schema,
    )
    if resp.parsed_output is None:
        raise RuntimeError("Claude returned no parseable output.")
    return resp.parsed_output


# ============================================================
# 4. DATA  (deterministic; the customer boundary lives here)
# ============================================================
@lru_cache(maxsize=1)
def get_transactions_df() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["customerId"] = df["customerId"].astype("string").str.strip()
    df["merchantCode"] = df["merchantCode"].astype("string").str.strip().str.upper()
    df["category"] = df["category"].astype("string").str.strip()
    df["id"] = df["id"].astype("string").str.strip()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["amountDkk"] = pd.to_numeric(df["amountDkk"], errors="coerce")
    return df


def known_categories() -> list[str]:
    """Categories are read from the data, not hard-coded (works on any CSV)."""
    return sorted(c for c in get_transactions_df()["category"].dropna().unique())


def customer_df(customer_id: str) -> pd.DataFrame:
    """THE isolation boundary — only this customer's rows."""
    df = get_transactions_df()
    return df[df["customerId"] == customer_id.strip()]


def data_now() -> datetime:
    """The dataset is historical, so relative dates ('last 7 days', the 90-day
    dispute window) are anchored to the newest transaction, not the wall clock.
    In production this would just be datetime.now(timezone.utc)."""
    latest = get_transactions_df()["timestamp"].max()
    return latest.to_pydatetime() if pd.notna(latest) else datetime.now(timezone.utc)


def tx_to_dict(row) -> dict[str, Any]:
    d = dict(row)
    ts = d.get("timestamp")
    d["timestamp"] = ts.isoformat() if isinstance(ts, pd.Timestamp) and pd.notna(ts) else None
    if pd.notna(d.get("amountDkk")):
        d["amountDkk"] = round(float(d["amountDkk"]), 2)
    d.pop("note", None)
    return d


# ============================================================
# 5. LLM CALLS  (intent, spend query, merchant) — with fallbacks
# ============================================================
def classify_intent(question: str) -> IntentResult:
    system = (
        "You route a banking card assistant. Classify the message into exactly "
        "one intent:\n"
        "- spend   : totals / counts / 'how much' / 'how many' / by period or category / first / last / top.\n"
        "- explain : 'what is this transaction' / 'what is <merchant>'.\n"
        "- dispute : 'I don't recognise this' / fraud / 'charged twice' / 'I want to dispute'.\n"
        "Also return normalized_question: fix typos, drop filler, keep the meaning. "
        "For a dispute, make it just the transaction id or merchant the user named."
    )
    try:
        return claude_parse(system, question, IntentResult)
    except Exception:
        return IntentResult(intent=detect_intent_legacy(question), normalized_question=question)


def parse_spend_query(question: str) -> SpendQuery:
    system = (
        "Turn a spending question into a structured query over a transactions "
        "table (columns: merchantCode, amountDkk, timestamp, category).\n"
        "relational_filters map 1:1 to columns: date_from/date_to (YYYY-MM-DD), "
        "date_ranges (many/excluded months), last_n_days, categories (ONLY exact "
        f"values from {known_categories()}), merchant_code_exact (only if a full "
        "code like 'MC5678 AAR ICA' was given).\n"
        "semantic_filters.merchant_text holds a fuzzy merchant name ('netflix').\n"
        "aggregation.type: sum(how much) | count(how many) | top_n(highest, set n) "
        "| group_by_category | min_date(first/earliest) | max_date(last/latest).\n"
        "Never invent categories or merchant codes."
    )
    try:
        return claude_parse(system, question, SpendQuery)
    except Exception:
        # Deterministic fallback: just sum everything (honest, safe default).
        return SpendQuery(aggregation=AggregationSpec(type=Aggregation.SUM))


def resolve_merchant(user_text: str, candidates: list[str]) -> Optional[str]:
    """Map fuzzy merchant text to one real merchantCode the customer actually has."""
    candidates = [c for c in candidates if c][:40]
    if not candidates:
        return None
    system = (
        "Given a fuzzy merchant description and a list of real merchantCode values "
        "from the customer's transactions, return exactly one code from the list "
        "that matches, or 'NONE'. Never invent a code."
    )
    try:
        match = claude_parse(system, f"user_said={user_text}\ncandidates={candidates}", MerchantMatch).match
        return match if match in candidates else None
    except Exception:
        low = user_text.lower()
        return next((c for c in candidates if low in c.lower()), None)


# ============================================================
# 6. SPEND  (deterministic execution + plain-language summary)
# ============================================================
def execute_spend_query(customer_id: str, q: SpendQuery) -> dict[str, Any]:
    df = customer_df(customer_id)
    rel, sem = q.relational_filters, q.semantic_filters

    # --- relational filters ---
    if rel.date_ranges:
        mask = pd.Series(False, index=df.index)
        for r in rel.date_ranges:
            start = pd.to_datetime(r.date_from, utc=True)
            end = pd.to_datetime(r.date_to, utc=True) + timedelta(days=1) - timedelta(seconds=1)
            mask |= (df["timestamp"] >= start) & (df["timestamp"] <= end)
        df = df[mask]
    else:
        if rel.last_n_days is not None:
            since = pd.Timestamp(data_now() - timedelta(days=int(rel.last_n_days)))
            df = df[df["timestamp"] >= since]
        if rel.date_from:
            df = df[df["timestamp"] >= pd.to_datetime(rel.date_from, utc=True)]
        if rel.date_to:
            end = pd.to_datetime(rel.date_to, utc=True) + timedelta(days=1) - timedelta(seconds=1)
            df = df[df["timestamp"] <= end]
    if rel.categories:
        wanted = {c.lower() for c in rel.categories}
        df = df[df["category"].str.lower().isin(wanted)]
    if rel.merchant_code_exact:
        df = df[df["merchantCode"] == rel.merchant_code_exact.strip().upper()]

    # --- semantic merchant (LLM-resolved, else substring) ---
    resolved = None
    if sem.merchant_text and not df.empty:
        resolved = resolve_merchant(sem.merchant_text, df["merchantCode"].dropna().unique().tolist())
        if resolved:
            df = df[df["merchantCode"] == resolved]
        else:
            df = df[df["merchantCode"].str.contains(sem.merchant_text, case=False, na=False, regex=False)]

    # --- aggregation ---
    agg = q.aggregation.type
    result: dict[str, Any] = {"type": agg.value, "transaction_count": int(len(df)), "resolved_merchant": resolved}
    if df.empty:
        result["empty"] = True
        return result
    if agg is Aggregation.SUM:
        result["total_amount_dkk"] = round(float(df["amountDkk"].sum()), 2)
    elif agg is Aggregation.TOP_N:
        n = q.aggregation.n or 3
        result["n"] = n
        result["transactions"] = [tx_to_dict(r) for _, r in df.sort_values("amountDkk", ascending=False).head(n).iterrows()]
    elif agg is Aggregation.GROUP_BY_CATEGORY:
        grouped = df.groupby("category")["amountDkk"].sum().round(2).sort_values(ascending=False)
        result["by_category"] = [{"category": c, "total_amount_dkk": float(v)} for c, v in grouped.items()]
    elif agg is Aggregation.MIN_DATE:
        result["transaction"] = tx_to_dict(df.sort_values("timestamp").iloc[0])
    elif agg is Aggregation.MAX_DATE:
        result["transaction"] = tx_to_dict(df.sort_values("timestamp", ascending=False).iloc[0])
    return result


def build_spend_explanation(q: SpendQuery, result: dict[str, Any]) -> str:
    """A plain sentence built from the numbers pandas computed (never the LLM),
    so the words can't disagree with the figures."""
    rel = q.relational_filters
    scope = []
    if rel.categories:
        scope.append("in " + ", ".join(rel.categories))
    if result.get("resolved_merchant"):
        scope.append(f"at {result['resolved_merchant']}")
    elif q.semantic_filters.merchant_text:
        scope.append(f"matching '{q.semantic_filters.merchant_text}'")
    if rel.date_from and rel.date_to:
        scope.append(f"from {rel.date_from} to {rel.date_to}")
    elif rel.last_n_days is not None:
        scope.append(f"in the last {rel.last_n_days} days")
    s = (" " + " ".join(scope)) if scope else ""
    n = result.get("transaction_count", 0)

    if result.get("empty"):
        return f"I couldn't find any transactions{s}."
    t = result["type"]
    if t == "sum":
        return f"You spent {result['total_amount_dkk']:.2f} DKK{s}, across {n} transaction(s)."
    if t == "count":
        return f"You made {n} transaction(s){s}."
    if t == "top_n":
        return f"Here are your top {result.get('n', 3)} transaction(s){s}."
    if t == "group_by_category":
        return f"Here is your spending by category{s}."
    if t == "min_date":
        return f"Your earliest matching transaction{s} was on {result['transaction']['timestamp']}."
    if t == "max_date":
        return f"Your latest matching transaction{s} was on {result['transaction']['timestamp']}."
    return "Here is your spending result."


def handle_spend(customer_id: str, normalized_question: str) -> dict[str, Any]:
    q = parse_spend_query(normalized_question)
    result = execute_spend_query(customer_id, q)
    return {
        "intent": "spend",
        "interpreted_query": q.model_dump(mode="json"),
        "result": result,
        "explanation": build_spend_explanation(q, result),
    }


# ============================================================
# 7. EXPLAIN  (customer-scoped lookup + grounded text)
# ============================================================
_TX_TOKEN = re.compile(r"\bTX\w+\b", re.IGNORECASE)
_MC_TOKEN = re.compile(r"\bMC\d{3,}\b", re.IGNORECASE)


def find_transactions_for_user(customer_id: str, query: str) -> list[dict[str, Any]]:
    df = customer_df(customer_id)
    if df.empty:
        return []

    def out(rows) -> list[dict[str, Any]]:
        return [tx_to_dict(r) for _, r in rows.sort_values("timestamp", ascending=False).iterrows()]

    # A transaction id or merchant code anywhere in the message -> exact match.
    # (Works even when the LLM hasn't normalised "I don't recognise TX123".)
    if (m := _TX_TOKEN.search(query)) is not None:
        return out(df[df["id"].str.upper() == m.group(0).upper()])
    if (m := _MC_TOKEN.search(query)) is not None:
        hit = df[df["merchantCode"].str.contains(m.group(0).upper(), na=False, regex=False)]
        if not hit.empty:
            return out(hit)

    # Otherwise: strip filler, substring-match on merchant text.
    q = query.strip()
    for prefix in ("what is", "what's", "explain", "hvad er", "forklar"):
        if q.lower().startswith(prefix):
            q = q[len(prefix):].strip(" ?:").strip()
            break
    return out(df[df["merchantCode"].str.contains(q, case=False, na=False, regex=False)]) if q else []


def build_grounded_explanation(tx: dict[str, Any]) -> str:
    code = str(tx.get("merchantCode", "")).strip()
    if code in MERCHANT_MAP:
        return (
            f"This is a transaction at {MERCHANT_MAP[code]}. "
            f"Amount: {tx.get('amountDkk')} DKK, dated {tx.get('timestamp')}, "
            f"category {tx.get('category')}."
        )
    return "I can't clearly identify that merchant — please check your statement or contact support."


def handle_explain(customer_id: str, question: str) -> dict[str, Any]:
    txs = find_transactions_for_user(customer_id, question)
    if not txs:
        return {"intent": "explain", "matches_found": 0,
                "explanation": "I couldn't find that transaction on your account. Try the exact id (TX...) or the merchant name."}
    tx = txs[0]
    LAST_TX_BY_CUSTOMER[customer_id] = tx  # remember for a follow-up dispute
    return {"intent": "explain", "matches_found": len(txs), "transaction": tx,
            "explanation": build_grounded_explanation(tx)}


# ============================================================
# 8. DISPUTE  (rules in Python; 4 reasons; interactive; JSON)
# ============================================================
def is_nondisputable(tx: dict[str, Any]) -> Optional[str]:
    if str(tx.get("category", "")).lower() in ("atm", "cash", "fee", "fees"):
        return "ATM, cash and fee transactions can't be disputed."
    ts = pd.to_datetime(tx.get("timestamp"), utc=True, errors="coerce")
    if pd.notna(ts) and ts.to_pydatetime() < data_now() - timedelta(days=DISPUTE_WINDOW_DAYS):
        return f"Transactions older than {DISPUTE_WINDOW_DAYS} days can't be disputed."
    return None


def parse_dispute_reason(text: str) -> Optional[str]:
    t = text.strip().lower()
    if t in ("1",) or "fraud" in t or "stolen" in t or "unauthor" in t or "someone" in t:
        return "Fraudulent transaction"
    if t in ("2",) or "duplicate" in t or "twice" in t or "double" in t:
        return "Duplicate charge"
    if t in ("3",) or "not received" in t or "never arrived" in t or "didn't get" in t or "didnt get" in t:
        return "Goods/services not received"
    if t in ("4",) or "wrong amount" in t or "incorrect amount" in t or "overcharg" in t or "too much" in t:
        return "Wrong amount charged"
    return None


def _reason_menu() -> str:
    return "Which is it?\n" + "\n".join(f"{i}) {r}" for i, r in enumerate(ALLOWED_DISPUTE_REASONS, 1))


def handle_dispute(customer_id: str, question: str) -> dict[str, Any]:
    session = DISPUTE_SESSIONS.get(customer_id)

    # --- continue an in-progress dispute ---
    if session:
        if session["status"] == "awaiting_reason":
            reason = parse_dispute_reason(question)
            if not reason:
                return {"intent": "dispute", "needs_input": True, "message": _reason_menu()}
            session["reason"] = reason
            session["status"] = "awaiting_details"
            return {"intent": "dispute", "needs_input": True,
                    "message": "Got it. Briefly, what happened?"}
        # awaiting_details -> finalise
        tx = session["transaction"]
        ticket = {
            "TransactionId": tx.get("id"),
            "MerchantCode": tx.get("merchantCode"),
            "AmountDkk": tx.get("amountDkk"),
            "Reason": session["reason"],
            "CustomerDescription": question.strip(),
            "SubmittedAt": datetime.now(timezone.utc).isoformat(),
            "Status": "submitted",
        }
        DISPUTE_SESSIONS.pop(customer_id, None)
        return {"intent": "dispute", "needs_input": False, "dispute": ticket,
                "message": f"Your dispute for {ticket['TransactionId']} was submitted as '{ticket['Reason']}'."}

    # --- start a new dispute ---
    txs = find_transactions_for_user(customer_id, question)
    tx = txs[0] if txs else LAST_TX_BY_CUSTOMER.get(customer_id)
    if not tx:
        return {"intent": "dispute", "needs_input": True,
                "message": "Which transaction? Give me the id (TX...) or the merchant name."}
    blocked = is_nondisputable(tx)
    if blocked:
        return {"intent": "dispute", "needs_input": False, "non_disputable": True, "message": blocked}
    DISPUTE_SESSIONS[customer_id] = {"transaction": tx, "status": "awaiting_reason"}
    return {"intent": "dispute", "needs_input": True,
            "message": f"Starting a dispute for {tx.get('id')} ({tx.get('amountDkk')} DKK). {_reason_menu()}"}


# ============================================================
# 9. LEGACY INTENT  (fallback when the LLM is unavailable)
# ============================================================
def detect_intent_legacy(question: str) -> Intent:
    q = question.lower()
    if any(w in q for w in ("dispute", "don't recognise", "dont recognize", "fraud", "charged twice", "chargeback")):
        return Intent.DISPUTE
    if any(w in q for w in ("how much", "how many", "spent", "spend", "total", "top ", "biggest",
                            "first", "last", "by category", "average")):
        return Intent.SPEND
    return Intent.EXPLAIN


# ============================================================
# 10. ROUTER  (one place; enforces the semantic/deterministic split)
# ============================================================
def handle_user_query(customer_id: str, question: str) -> dict[str, Any]:
    # A dispute in progress captures the next turns (so "2" is dispute input).
    if DISPUTE_SESSIONS.get(customer_id):
        return handle_dispute(customer_id, question)

    routed = classify_intent(question)
    print(f"  [intent={routed.intent.value}  normalized='{routed.normalized_question}']")
    nq = routed.normalized_question or question

    if routed.intent is Intent.SPEND:
        return handle_spend(customer_id, nq)
    if routed.intent is Intent.DISPUTE:
        return handle_dispute(customer_id, nq)
    return handle_explain(customer_id, nq)


# ============================================================
# 11. CLI  (sticky customer, like a real banking session)
# ============================================================
def run_cli() -> None:
    print("LLM-Bot (Claude) — ask about your card transactions. Ctrl+C to exit.")
    print(f"Model: {MODEL}   LLM: {'on' if llm_enabled() else 'off (deterministic fallback)'}")
    customer_id: Optional[str] = None
    while True:
        try:
            if not customer_id:
                customer_id = input("\nCustomerId (e.g. CUST001): ").strip()
                continue
            question = input(f"[{customer_id}] ask: ").strip()
            if not question:
                continue
            if question.lower().startswith((":id ", "customer ")):
                customer_id = question.split()[-1].strip()
                print(f"-> switched to {customer_id}")
                continue
            response = handle_user_query(customer_id, question)
            print(json.dumps(response, indent=2, default=str))
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            break


if __name__ == "__main__":
    run_cli()
