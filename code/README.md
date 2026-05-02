# Support Triage Agent — `code/`

## Quick Start

```bash
# No install needed — pure Python 3.10+ stdlib only
cd code/
python main.py                  # process all tickets → output.csv
python main.py --verbose        # also print each result to terminal
python main.py --input  ../support_tickets/support_tickets.csv \
               --output ../support_tickets/output.csv
```

No API key. No network. No pip installs.

## Architecture

```
main.py        CLI entry point — reads CSV, runs pipeline, writes output.csv
classifier.py  Domain inference, escalation rules, product_area mapping, query builder
retriever.py   BM25 corpus search over 774 markdown files in data/
agent.py       Offline decision tree — produces Status, product_area, response, justification
```

## Pipeline (per ticket)

```
Input row (Issue, Subject, Company)
  │
  ├─ 1. infer_domain()          → "hackerrank" / "claude" / "visa" / None
  ├─ 2. infer_product_area()    → short label (e.g. "assessments", "fraud")
  ├─ 3. classify_request_type() → bug / feature_request / product_issue / invalid
  ├─ 4. should_escalate()       → True/False + reason (25+ regex rules)
  ├─ 5. build_retrieval_query() → targeted query (keyword overrides)
  ├─ 6. search()                → top-5 BM25 docs from data/, domain-filtered
  └─ 7. triage()                → Status, Product Area, Response, Justification, Request Type
```

## Output Schema

| Column       | Values                                               |
|--------------|------------------------------------------------------|
| Status       | `Replied` or `Escalated`                            |
| Product Area | Short label: `assessments`, `fraud`, `account_access`, etc. |
| Request Type | `product_issue`, `bug`, `feature_request`, `invalid` |

## Key Design Decisions

- **100% offline** — zero API calls, zero network dependency
- **Deterministic** — same input always produces same output
- **Layered escalation** — 25+ regex rules fire before retrieval
- **Grounded responses** — built from top corpus document only
- **Proper Status casing** — `Replied` / `Escalated` per spec
- **Short product_area labels** — `assessments` not `hackerrank/screen/managing-tests`
