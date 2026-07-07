# LLM-Bot (Claude)

A banking assistant over a CSV of card transactions, on **Claude**. The focus here
is **[`main_tools.py`](main_tools.py)** — the **tool-calling** version, where Claude
decides which action to take. There's also [`main.py`](main.py), a Claude port of
the older **router** approach, kept as an offline reference.

> For the full **router vs. tool calling** comparison, architecture diagrams, and
> the semantic-vs-deterministic idea, see the parent folder's
> [`../README.md`](../README.md). This file is just the quick guide to running the
> two Claude entry points.

---

## `main_tools.py` — tool calling (the focus)

You hand Claude three tools and it decides which to call — no hand-written router.

- `query_spending(aggregation, category, date_from/to, date_ranges, last_n_days, merchant_text, n)`
- `explain_transaction(reference)`
- `file_dispute(reference, reason)`

The tool bodies are ordinary deterministic Python (pandas + the dispute rules), so:

- Claude chooses the action, but **Python still owns the numbers and rules** —
  `file_dispute` rejects an ATM charge or a non-whitelisted reason; the model can't
  override it.
- `customer_id` is **not** a tool argument (the tools read the session's fixed
  customer), so Claude literally cannot query another customer.
- The `@beta_tool` decorator builds each tool's JSON schema from the function
  signature + docstring — "you write Python, the SDK writes the schema."

The file is **standalone** (it carries its own small copy of the deterministic
core), so you can read it top to bottom on its own.

```bash
cd LLM_BOT
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...     # required (model-driven)
python main_tools.py
```

Optional: `export ANTHROPIC_MODEL=claude-sonnet-4-6` for a cheaper model.

---

## `main.py` — router (secondary, runs offline)

The Claude port of the original hand-written router: `classify_intent` → `if/elif`
dispatch to spend / explain / dispute. It uses **Pydantic + `messages.parse`** for
typed structured outputs, and — unlike `main_tools.py` — it **runs with no API
key**, falling back to a deterministic keyword classifier so the CLI always
answers (you just lose the fuzzy language understanding).

```bash
export ANTHROPIC_API_KEY=sk-ant-...     # optional: runs offline without it
python main.py
```

Kept as a reference for the router architecture; the parent
[`../README.md`](../README.md) explains how it compares to tool calling.

---

## Try it

Pick a customer id (e.g. `CUST031`) and ask:

- **spend** — "how much did I spend on groceries in 2025 except January and May?"
- **explain** — "what is TX70196?"
- **dispute** — "I don't recognise TX90009" → follow the prompts

## Delivering these tools over MCP

`main_tools.py` is the *model-orchestrates* philosophy. To let **any** MCP client
(not just this script) use these same tools, you serve them over an **MCP server**
— see [`../MCP_BOT/`](../MCP_BOT/). The tool bodies don't change; only how they're
exposed does.