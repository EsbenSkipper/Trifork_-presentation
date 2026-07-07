"""Deterministic core of the tool-calling bot (LLM_BOT/main_tools.py).

Same guarantees as the router bot, verified through the tool bodies. We call the
underlying functions via `.func` (the plain Python the `@beta_tool` decorator
wraps) and parse their JSON-string return.
"""
import json

import pandas as pd

CUST = "CUST001"


def _rows(tools, cust=CUST):
    df = tools._df()
    return df[df["customerId"] == cust]


def _spend(tools, **kwargs):
    return json.loads(tools.query_spending.func(**kwargs))


# --- customer isolation ----------------------------------------------------
def test_cannot_find_another_customers_transaction(tools):
    df = tools._df()
    b_tx = df[df["customerId"] == "CUST002"]["id"].iloc[0]
    assert tools.find_tx("CUST001", f"what is {b_tx}") is None
    assert tools.find_tx("CUST002", f"what is {b_tx}") is not None


# --- spend aggregations vs pandas ------------------------------------------
def test_sum_matches_pandas(tools):
    tools.CURRENT_CUSTOMER = CUST
    assert _spend(tools, aggregation="sum")["total_amount_dkk"] == round(float(_rows(tools)["amountDkk"].sum()), 2)


def test_count_matches_pandas(tools):
    tools.CURRENT_CUSTOMER = CUST
    assert _spend(tools, aggregation="count")["transaction_count"] == len(_rows(tools))


def test_category_filter_matches_pandas(tools):
    tools.CURRENT_CUSTOMER = CUST
    rows = _rows(tools)
    cat = rows["category"].mode().iloc[0]
    out = _spend(tools, aggregation="sum", category=cat)
    assert out["total_amount_dkk"] == round(float(rows[rows["category"] == cat]["amountDkk"].sum()), 2)


def test_top_n_is_sorted(tools):
    tools.CURRENT_CUSTOMER = CUST
    txs = _spend(tools, aggregation="top_n", n=3)["transactions"]
    amounts = [t["amountDkk"] for t in txs]
    assert len(amounts) == 3 and amounts == sorted(amounts, reverse=True)


def test_date_range_matches_pandas(tools):
    tools.CURRENT_CUSTOMER = CUST
    out = _spend(tools, aggregation="sum", date_from="2025-01-01", date_to="2025-06-30")
    rows = _rows(tools)
    m = (rows["timestamp"] >= pd.Timestamp("2025-01-01", tz="UTC")) & (rows["timestamp"] <= pd.Timestamp("2025-06-30 23:59:59", tz="UTC"))
    assert out["total_amount_dkk"] == round(float(rows[m]["amountDkk"].sum()), 2)


# --- dispute rules enforced in the tool ------------------------------------
def test_file_dispute_enforces_rules(tools):
    df = tools._df()
    good = df[(df["timestamp"] >= pd.Timestamp(tools.data_now() - pd.Timedelta(days=60)))
              & (~df["category"].str.lower().isin(["atm", "cash", "fee", "fees"]))].iloc[0]
    tools.CURRENT_CUSTOMER = good["customerId"]
    assert json.loads(tools.file_dispute.func(good["id"], "Duplicate charge"))["Status"] == "submitted"
    assert json.loads(tools.file_dispute.func(good["id"], "not a real reason"))["error"] == "invalid_reason"

    atm = df[df["category"].str.lower() == "atm"].iloc[0]
    tools.CURRENT_CUSTOMER = atm["customerId"]
    assert json.loads(tools.file_dispute.func(atm["id"], "Fraudulent transaction"))["error"] == "non_disputable"


def test_explain_known_vs_unknown(tools):
    df = tools._df()
    netflix = df[df["merchantCode"] == "MC2468 ONLINE NETFLX"].iloc[0]
    tools.CURRENT_CUSTOMER = netflix["customerId"]
    out = json.loads(tools.explain_transaction.func(netflix["id"]))
    assert "Netflix" in out["merchant"]
    assert json.loads(tools.explain_transaction.func("TX00000"))["error"] == "transaction_not_found"


# --- customer_id can't be chosen by the model ------------------------------
def test_customer_id_is_not_a_tool_argument(tools):
    # the tool schema must not expose customer_id — isolation is structural.
    assert "customer_id" not in tools.query_spending.input_schema["properties"]
    assert "customerId" not in tools.query_spending.input_schema["properties"]