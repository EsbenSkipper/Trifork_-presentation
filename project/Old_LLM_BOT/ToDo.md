# ToDo — Old_LLM_BOT

Running notes for the OpenAI-backed version of the old LLM-Bot. Pick these up later.

## Context
- This bot was converted from **Azure OpenAI** to the **standard OpenAI API**.
  - `main.py` now reads the key from `OPENAI_API_KEY` (env var) and uses `OPENAI_MODEL`.
  - `_azure_post()` now calls `https://api.openai.com/v1/chat/completions`
    (Bearer auth, model in the body). It kept its old name — see "nice-to-have".
- To run:  `export OPENAI_API_KEY=sk-...`  then  `python main.py`
  (run from inside this folder — the CSV path is relative; see item 1).

## Already done
- [x] Azure → OpenAI transport swap (config + `_azure_post`).
- [x] Spend schema set to `"strict": False` — OpenAI strict mode rejects the
      nested spend schema (it requires every nested object to list `required`
      for all properties). Non-strict adherence works. The intent schema stays
      strict (it's flat, so it's fine).

## Bugs found in the head-to-head vs the Claude bot
- **[FIXED 2026-07-05] Disputes were impossible on this dataset.**
  `_is_nondisputable` used the real wall clock for the 90-day rule while the data
  is from 2025, so every transaction looked too old. Now anchored to the
  dataset's latest transaction via a new `_data_now()` helper (also used by
  `last_n_days`). Verified: recent transactions dispute; genuinely old ones still
  refuse.
- **[FIXED 2026-07-05] Dispute continuation could drop mid-flow.**
  `handle_user_query` now routes straight to `handle_dispute` when a dispute
  session is active for the customer, so replies like "someone used my card" no
  longer get re-classified and dropped. Verified: full flow reaches a ticket.
- **[FIXED 2026-07-05] Isolation trusted the LLM.** `execute_spend_query` now
  takes the authenticated `customer_id` as a parameter and filters on that,
  ignoring any `customer_id` the model wrote into the query. Verified: a
  cross-customer question returns the current customer's own data.

## To do (agreed improvements)
1. **[FIXED 2026-07-05] Absolute CSV path.** `CSV_PATH` is now built from
   `os.path.dirname(os.path.abspath(__file__))`, so the bot runs from any
   working directory. Verified by importing from the repo root.

2. **[FIXED 2026-07-05] Friendly missing-key error.** `OPENAI_API_KEY` is read
   with `os.environ.get(..., "")` and `run_cli()` exits with a clear message
   ("Set your OpenAI key first: export OPENAI_API_KEY=sk-...") instead of a raw
   `KeyError`. Importing the module no longer requires a key (deterministic
   paths work offline).

3. **[FIXED 2026-07-05] Tighten the intent prompt.** Added ranking/breakdown
   cues ("top / biggest / highest / most expensive", "by category",
   "when did I first / last") to `azure_classify_intent`, so "top 3
   transactions" etc. now route to *spend* instead of *explain*.
   - **Bonus bug this exposed & fixed:** `build_spend_explanation` reads
     `transaction_count` and treats a missing value as "empty", but the `top_n`
     and `group_by_category` result dicts didn't set it — so those answers
     wrongly said "none matched". `execute_spend_query` now includes
     `transaction_count` in both. (Was hidden before because these queries were
     misrouted to *explain* and never reached the spend path.)

4. **[FIXED 2026-07-05] Explain fallback for unknown merchants.**
   `_build_grounded_explanation_from_tx` no longer dead-ends on merchants outside
   `MERCHANT_MAP`; it falls back to the raw transaction facts (merchant code,
   amount, date, category). A genuine not-found now says "I couldn't find that
   transaction on your account…" instead of the old "Unknown merchant" message.

All agreed improvements are now done. Remaining items are the optional
nice-to-haves below.

## Nice-to-have / notes
- **Rotate the API key.** The original key was hard-coded and is now exposed;
  it must be revoked and replaced (the code already reads from the env var, so
  no file change needed — just export the new key).
- **Rename `_azure_post` -> `_openai_post`** for clarity (touches ~4 call sites).
- **Stricter schema option.** If we want `"strict": True` back on the spend
  schema, every nested object must list `required` for all its properties and
  set `additionalProperties: false`. More correct, but verbose and makes
  optional fields explicit-null.

## How to trigger each path (quick reference)
- **spend**   : "how much did I spend on groceries?", "how many transactions?"
- **explain** : "what is TX70196?" as the owning customer (CUST031 = Netflix).
- **dispute** : "I don't recognise TX70196" (then follow the 4-reason prompts).