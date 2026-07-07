# LLM-Bot

This project is a tiny, end-to-end assistant that can:

1. **Answer natural-language spending questions** over a CSV of transactions  
2. **Explain a transaction** in human-readable language (but only for the customer who actually made it)  
3. **Run a dispute flow** that follows assignment rules (4 reasons, no fees/ATM, 90-day limit, structured JSON)

It does this by **splitting the problem in two**:

- things that are **semantic / human** → we let the **LLM** handle it
- things that are **relational / deterministic / business rules** → we keep in **Python** (pandas)

That split is what makes the whole thing work on new CSVs and still stay safe.

---
**What GPT handles:**

- **Intent** (e.g., “how much did I…”, “what is…”, “I don’t recognize…”)  
  → `azure_classify_intent(...)`
- **Fuzzy merchant names** (e.g., “McDonalds”, “MCDON”, “mc don”)  
  → `azure_resolve_merchant(...)`
- **Fuzzy time phrases** (e.g., “in October”, “last 2 days”, “January and May”)  
  → `azure_chat_structured(...)` *(returns a structured shape we can reason about)*
- **Free-form dispute text** (e.g., “I was charged twice”, “I didn’t get the goods”)  
  → `_parse_dispute_reason(...)` *(can be LLM-ified)*

---
**What Python/pandas enforces:**

- **Only this customer’s transactions**  
  → `df[df["customerId"] == customer_id]`
- **Exact date filters**  
  → date ranges, `last_n_days` *(with data-relative fallback)*
- **Only real categories**  
  → no hallucinated categories
- **Exact aggregations**  
  → `sum` / `count` / `top_n` / `group_by_category` / `min_date` / `max_date`
- **Business / assignment rules**  
  → no disputes on ATM/fees, not older than 90 days, **4 reasons only**
---

## How it works (Overview)

```
User (CLI)
  │
  ├─▶ (1) read current_customer from CLI
  │
  ├─▶ (2) user types question
  │
  └─▶ handle_user_query(customer_id, question)
        │
        ├─▶ try: azure_classify_intent(...)
        │        │
        │        └─▶ returns { intent: "spend" | "explain" | "dispute",
        │                      normalized_question: "..."}
        │
        ├─▶ if LLM failed → detect_intent_legacy(...)
        │
        ├─▶ branch on intent
        │     ├─ "spend"   → spend path
        │     ├─ "explain" → explain path
        │     └─ "dispute" → dispute path
        │
        └─▶ return JSON to CLI (printed)
```
### Why this design?

**We only ask the LLM what it’s good at.**  
The LLM decides what the user wants (**spend / explain / dispute**) and normalizes the question. Everything that must be strict (filtering, summing, 90-day rule) stays in **Python**.

**Each path can be developed separately.**

- **Spend path** → LLM → structured query → pandas  
- **Explain path** → customer-only lookup → grounded explanation  
- **Dispute path** → interactive, 4 fixed reasons, JSON output  

Because they’re separate, they don’t step on each other.

**Sticky customer = realistic banking flow.**  
We read the `customerId` once and keep it. That makes follow-up turns like “I would like to dispute” possible **without** repeating the ID.

**Graceful degradation.**  
If `azure_classify_intent(...)` fails (bad response, schema mismatch, rate limit), we don’t crash — we fall back to `detect_intent_legacy(...)` (a simple Python classifier). So the CLI always returns *something*.

**Easy to log and demo.**  
At the router we could log:

- what the user asked  
- what the LLM thought the intent was  
- which path was executed  

That makes it very clear to an examiner/reviewer where mistakes come from.

**Security / isolation stays central.**  
Because routing happens in one place, we can enforce “only look at **this** customer’s transactions” in each path, and never let the LLM “guess” other people’s data.

---
### A short note on this demo and missing stuff

This version works  **as long as the user only has one intent per turn** (e.g. “How much did I spend in 2025?” or “What is MC5678 AAR ICA?”). Everything had to happen Friday because of a full weekend, but a few things could be improved if given time:

1. **Multi-intent support**  
   Right now `azure_classify_intent()` returns a single intent. A nicer version would let the model return **multiple** intents (e.g. `["spend", "explain"]`) and then we would run both paths and **merge the JSON** before returning it.

2. **Dynamic categories**  
   Categories are currently **hardcoded** in the code. Ideally they should live in a separate JSON file that is **rebuilt/updated** when a new CSV/dataset is loaded, so the model can work with new merchant categories without code changes.

3. **Fewer / cheaper LLM calls**  
   The demo uses **multiple Azure calls** (intent → structured query → merchant resolve) because it’s easier to understand and debug. In a production setup we could:
   - collapse this into **one** bigger call
   - or use **smaller/cheaper models** for the simple classifier steps.
     
4. **Logging and observability**  
   Because the logic is split into paths (spend / explain / dispute), it’s very easy to **log each step**:
   - what the LLM thought the intent was
   - what structured query it produced
   - how many rows the relational filter actually returned  
   This is great for demos and assignment hand-ins.  

    **But**: when you have several paths, **mistakes can be hidden more easily** — e.g. the LLM misclassifies a question as “spend”, the spend path runs, and you get `0.0 DKK` instead of a nice explanation. Good logs would make that obvious.
---
## Spend path

```text
SPEND path
  │
  ├─▶ azure_chat_structured(normalized_question, customer_id)
  │      │
  │      └─▶ returns structured query:
  │             {
  │               "intent": "spend_query",
  │               "customer_id": "...",
  │               "relational_filters": {...},
  │               "semantic_filters": {...},
  │               "aggregation": {
  │                  "type": "sum" | "count" | "top_n" |
  │                          "group_by_category" | "min_date" | "max_date"
  │               },
  │               "raw_question": "..."
  │             }
  │
  ├─▶ execute_spend_query(query)
  │      │
  │      ├─▶ (1) load df = get_transactions_df()
  │      ├─▶ (2) filter to this customerId
  │      ├─▶ (3) apply **RELATIONAL** filters
  │      │       • date_ranges → OR over ranges
  │      │       • last_n_days → now - N (or data-relative)
  │      │       • date_from / date_to
  │      │       • category (lowercased)
  │      │       • merchantCode_exact
  │      ├─▶ (4) apply **SEMANTIC** merchant
  │      │       • if semantic_filters.merchant_text:
  │      │             – collect candidate merchantCodes
  │      │             – azure_resolve_merchant(user_text, candidates)
  │      │             – else fallback to substring
  │      └─▶ (5) aggregation switch
  │              • sum / count / top_n / group_by_category
  │              • min_date → earliest tx
  │              • max_date → latest tx
  │
  ├─▶ build_spend_explanation(...)
  │
  └─▶ return {
         "intent": "spend",
         "interpreted_query": ...,
         "result": ...,
         "explanation": "..."
       }
```
## Explain path
```text
EXPLAIN path
  │
  ├─▶ handle_explain(customer_id, question)
  │      │
  │      ├─▶ find_transactions_for_user(...)
  │      │       • strip "what is", "explain", "hvad er", ...
  │      │       • if TX... → exact id match
  │      │       • else → substring on merchantCode (for THIS customer only)
  │      │
  │      ├─▶ if no match → "Unknown merchant, please contact support."
  │      │
  │      ├─▶ else → pick first, save in LAST_TX_BY_CUSTOMER[customerId]
  │      │
  │      └─▶ _build_grounded_explanation_from_tx(tx)
  │             • if merchantCode in MERCHANT_MAP → friendly text
  │             • else → "Unknown merchant, please contact support."
  │
  └─▶ return JSON
```
---
## Dispute path
```text
DISPUTE path
  │
  ├─▶ handle_dispute(customer_id, question)
  │      │
  │      ├─▶ check DISPUTE_SESSIONS[customer_id]
  │      │      ├─ if status == "awaiting_reason":
  │      │      │      • _parse_dispute_reason(...)
  │      │      │      • if no reason → ask again (list 4)
  │      │      │      • else → status = "awaiting_details"
  │      │      │
  │      │      └─ if status == "awaiting_details":
  │      │             • take user description
  │      │             • output FINAL JSON:
  │      │                  { TransactionId, Reason,
  │      │                    Collected user inputs, Submission timestamp }
  │      │
  │      └─▶ (no session) → start new
  │             • find tx (from text OR LAST_TX_BY_CUSTOMER)
  │             • check non-disputable (ATM, cash, fees, >90 days)
  │             • create session = awaiting_reason
  │             • return list of 4 dispute reasons
  │
  └─▶ CLI prints the JSON
```

## What can (and can’t) be LLM-ified

Not every function in this project should be handed to the LLM. Some parts are **meant** to be fuzzy and language-driven, and some parts **must** stay deterministic because they touch money, rules, or user isolation.

**Note:** you could also design this so that there is **just a single Azure call** per user message that does *everything* (intent + structured spend query + resolve merchant ). 
In **this demo**, we’ve **split it into several smaller calls** because it’s easier to debug, easier to test each step, and it makes the “semantic vs relational” split very clear.


Here’s how the current code breaks down:

| Function / area                       | Can we LLM it?            | Pros                               | Cons                                   |
| ------------------------------------- | ------------------------- | ---------------------------------- | -------------------------------------- |
| `azure_classify_intent`               | Already LLM               | flexible intents                   | depends on schema/version              |
| `azure_chat_structured`               | Already LLM               | rich queries, date ranges          | can fail → fallback                    |
| `azure_resolve_merchant`              | Already LLM               | fixes misspellings                 | extra call                             |
| `find_transactions_for_user`          | ✅ yes                     | fuzzier matching, nicer “what is…” | must still filter to user’s own txs    |
| `_build_grounded_explanation_from_tx` | ✅ yes (with map)         | nicer, localized text              | must forbid hallucination, pass map    |
| `_parse_dispute_reason`               | ✅ yes                     | handles free-form user text        | still need to whitelist 4 reasons      |
| `build_spend_explanation`             | ✅ optional                | nicer wording                      | costs more, risk mismatch with numbers |
| `execute_spend_query`                 | ❌ no                      | —                                  | must be deterministic                  |
| `_is_nondisputable`                   | ❌ no                      | —                                  | must enforce assignment rules          |
| `start_dispute_submission`            | ❌ no                      | —                                  | must output fixed JSON                 |
| `detect_intent_legacy`                | could drop if LLM is 100% | smaller code                       | but legacy saves you when LLM fails    |

### How to read this table

- **“Already LLM”** → we’re *already* calling GPT here, because it’s the best tool (intent detection, fuzzy merchant matching).
- **“✅ yes”** → we *can* push this to GPT to make it smarter/nicer, **but** we should still validate the answer in Python.
- **“❌ no”** → must stay in Python because it’s either money, rules, or user isolation.


