#!/usr/bin/env python3
"""
main.py — Terminal entry point for the Multi-Domain Support Triage Agent.

100% OFFLINE. No API key required. No network calls. Pure Python 3.10+ stdlib.

Usage:
    python main.py                           # process support_tickets.csv → output.csv
    python main.py --input  path/to/in.csv
    python main.py --output path/to/out.csv
    python main.py --verbose                 # print each result to terminal

Output schema (matches sample_support_tickets.csv exactly):
    Issue, Subject, Company, Response, Product Area, Status, Request Type, Justification
"""

import argparse
import csv
import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).parent.resolve()
REPO_DIR = CODE_DIR.parent
DEFAULT_INPUT  = REPO_DIR / "support_tickets" / "support_tickets.csv"
DEFAULT_OUTPUT = REPO_DIR / "support_tickets" / "output.csv"

sys.path.insert(0, str(CODE_DIR))

from retriever  import search
from classifier import (
    infer_domain, infer_product_area, should_escalate,
    classify_request_type, build_retrieval_query,
)
from agent import triage

# Output columns — match sample_support_tickets.csv header exactly
OUTPUT_COLUMNS = [
    "Issue", "Subject", "Company",
    "Response", "Product Area", "Status", "Request Type", "Justification",
]


def process_ticket(row: dict, verbose: bool = False) -> dict:
    issue   = (row.get("Issue")   or row.get("issue")   or "").strip()
    subject = (row.get("Subject") or row.get("subject") or "").strip()
    company = (row.get("Company") or row.get("company") or "None").strip()

    # 1. Domain
    domain = infer_domain(company, issue, subject)
    # 2. Product area (pre-retrieval heuristic)
    product_area = infer_product_area(domain, issue, subject)
    # 3. Request type
    request_type = classify_request_type(issue, subject)
    # 4. Escalation rules
    escalate, reason = should_escalate(issue, subject, domain)
    # 5. Corpus retrieval
    query = build_retrieval_query(issue, subject)
    docs  = search(query, domain_filter=domain, top_k=5)
    # 6. Triage decision
    result = triage(
        issue=issue, subject=subject, company=company, domain=domain,
        retrieved_docs=docs, pre_escalate=escalate,
        pre_escalate_reason=reason, pre_request_type=request_type,
        product_area=product_area,
    )

    output_row = {
        "Issue":        issue,
        "Subject":      subject,
        "Company":      company,
        "Response":     result.get("response", "").strip(),
        "Product Area": result.get("product_area", product_area),
        "Status":       result.get("Status", "Escalated"),
        "Request Type": result.get("Request Type", request_type),
        "Justification":result.get("justification", "").strip(),
    }

    if verbose:
        print(f"\n  Status       : {output_row['Status']}")
        print(f"  Product Area : {output_row['Product Area']}")
        print(f"  Request Type : {output_row['Request Type']}")
        print(f"  Justification: {output_row['Justification'][:110]}")
        print(f"  Response     : {output_row['Response'][:160]}")

    return output_row


def main():
    parser = argparse.ArgumentParser(description="Multi-Domain Support Triage Agent (fully offline)")
    parser.add_argument("--input",   default=str(DEFAULT_INPUT))
    parser.add_argument("--output",  default=str(DEFAULT_OUTPUT))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} tickets from {args.input}")
    print("Running offline triage pipeline...\n")

    results = []
    for i, row in enumerate(rows, 1):
        preview = (row.get("Issue") or row.get("issue") or "")[:65].strip()
        print(f"[{i:02d}/{len(rows)}] {preview!r}", end="")
        try:
            result = process_ticket(row, verbose=args.verbose)
            print(f"  → {result['Status']} | {result['Product Area']} | {result['Request Type']}")
        except Exception as exc:
            print(f"  !! Error: {exc}")
            result = {
                "Issue":        (row.get("Issue") or row.get("issue") or "").strip(),
                "Subject":      (row.get("Subject") or row.get("subject") or "").strip(),
                "Company":      (row.get("Company") or row.get("company") or "").strip(),
                "Response":     "This issue requires human support escalation.",
                "Product Area": "general_support",
                "Status":       "Escalated",
                "Request Type": "product_issue",
                "Justification":f"Pipeline error: {exc}",
            }
        results.append(result)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(results)

    replied   = sum(1 for r in results if r["Status"] == "Replied")
    escalated = sum(1 for r in results if r["Status"] == "Escalated")
    print(f"\nDone. {len(results)} rows → {args.output}")
    print(f"  Replied={replied}  Escalated={escalated}")


if __name__ == "__main__":
    main()
