"""
agent.py — 100% offline, rule-based triage agent.

NO API. NO network. Pure Python stdlib only.
Produces Status: "Replied" / "Escalated" (capital R/E per spec).
Product area is derived from the retrieved doc's corpus folder path.
"""

import re
from typing import Optional


# ── Standard response messages ─────────────────────────────────────────────────

ESCALATION_MESSAGE = (
    "This issue requires human support escalation due to security, financial, "
    "or policy constraints. A support agent will review your request and follow up shortly."
)

OUT_OF_SCOPE_MESSAGE = (
    "I am sorry, this is out of scope from my capabilities. "
    "Please contact the relevant support team for assistance."
)

NO_CORPUS_MESSAGE = (
    "We were unable to find relevant documentation for your request. "
    "A support agent will follow up to assist you."
)


# ── Corpus path → short product_area label ─────────────────────────────────────
# Mirrors the _AREA_MAP in classifier.py but operates on the retrieved doc
from classifier import _corpus_path_to_area


# ── Content cleaner — strips YAML frontmatter and metadata ───────────────────

def _clean_content(raw: str, max_lines: int = 20) -> str:
    lines = raw.split("\n")
    result = []
    in_front = False
    past_front = False
    skip_prefixes = (
        "title:", "title_slug:", "source_url:", "final_url:", "last_updated",
        "last_modified", "article_id:", "article_slug:", "breadcrumbs:", "- \"", "- '",
    )
    skip_re = re.compile(r"^_last (updated|modified).*_$", re.IGNORECASE)

    for line in lines:
        s = line.strip()
        if s == "---":
            if not past_front:
                in_front = not in_front
                if not in_front:
                    past_front = True
            continue
        if in_front:
            continue
        if any(s.lower().startswith(p) for p in skip_prefixes):
            continue
        if skip_re.match(s.lower()):
            continue
        if s.startswith("_") and s.endswith("_") and len(s) < 100:
            continue
        if not result and not s:
            continue
        if s.startswith("#"):
            clean = re.sub(r"^#+\s*", "", s)
            if clean:
                result.append(clean)
        else:
            result.append(s)
        if len(result) >= max_lines:
            break

    return "\n".join(result).strip()


# ── Escalation justification builder ─────────────────────────────────────────

def _escalation_justification(reason: str) -> str:
    r = reason.lower()
    if any(k in r for k in ["refund", "chargeback", "financial", "payment", "billing", "order id"]):
        return f"Financial/billing issue requires human handling. {reason}"
    if any(k in r for k in ["identity", "theft", "stolen"]):
        return f"Identity theft report requires immediate human handling. {reason}"
    if any(k in r for k in ["score", "result", "grade", "next round", "review my answers"]):
        return f"Score/result manipulation request is outside support scope. {reason}"
    if any(k in r for k in ["affiche", "internal", "injection", "bypass", "règles", "system prompt", "delete all"]):
        return f"Adversarial/injection request rejected for safety. {reason}"
    if any(k in r for k in ["security", "vulnerab", "bug bounty", "exploit"]):
        return f"Security vulnerability report routed to security team. {reason}"
    if any(k in r for k in ["company", "recruiter", "force", "tell the", "ban"]):
        return f"Third-party coercion request cannot be processed. {reason}"
    if any(k in r for k in ["outage", "all requests", "stopped working", "none of the"]):
        return f"Potential sitewide outage requires engineering escalation. {reason}"
    if any(k in r for k in ["owner", "admin", "restore", "seat"]):
        return f"Account access change by non-admin requires human review. {reason}"
    if "vague" in r or "unknown" in r:
        return f"Ticket lacks sufficient information to process. {reason}"
    return f"High-risk or sensitive content detected; escalated for human review. {reason}"


# ── Response builder from corpus ──────────────────────────────────────────────

def _build_grounded_response(docs: list, domain: str) -> tuple[str, str, str]:
    """Returns (response, justification, product_area)."""
    if not docs:
        return NO_CORPUS_MESSAGE, "No corpus documents found; escalated for human review.", "general_support"

    top_doc, top_score = docs[0]

    title   = top_doc.get("title", "Support Article").strip()
    pa_path = top_doc.get("product_area", "")
    area    = _corpus_path_to_area(pa_path)
    content = top_doc.get("content", "")

    body = _clean_content(content, max_lines=20)

    # Remove leading line if it's identical to the doc title
    body_lines = body.split("\n")
    title_norm = re.sub(r"[^a-z0-9 ]", "", title.lower())
    while body_lines and re.sub(r"[^a-z0-9 ]", "", body_lines[0].lower()) == title_norm:
        body_lines.pop(0)
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    body = "\n".join(body_lines).strip()

    if body:
        response = f"Hi,\n\n{body}"
    else:
        response = f"Hi,\n\nPlease refer to our support documentation on '{title}' for guidance on this issue."

    if len(response) > 1000:
        response = response[:1000].rsplit("\n", 1)[0] + "\n\nFor more details, please visit our support center."

    justification = (
        f"Classified as {area}; retrieved '{title}' from corpus "
        f"(score={top_score:.2f}); response grounded in support documentation."
    )

    return response, justification, area


# ── Main triage function ──────────────────────────────────────────────────────

def triage(
    issue: str,
    subject: str,
    company: str,
    domain: Optional[str],
    retrieved_docs: list,
    pre_escalate: bool,
    pre_escalate_reason: str,
    pre_request_type: str,
    product_area: str,
    **kwargs,
) -> dict:
    """
    Fully offline deterministic triage.
    Returns: status ("Replied"/"Escalated"), product_area, response, justification, request_type.
    """

    # Step 1 — Hard escalation (safety/financial/coercion rules fired before retrieval)
    if pre_escalate:
        pa = product_area
        if retrieved_docs:
            pa = _corpus_path_to_area(retrieved_docs[0][0].get("product_area", ""))
        return {
            "Status":        "Escalated",
            "product_area":  pa,
            "response":      ESCALATION_MESSAGE,
            "justification": _escalation_justification(pre_escalate_reason),
            "Request Type":  "invalid" if pre_request_type == "invalid" else pre_request_type,
        }

    # Step 2 — Invalid / out-of-scope requests (replied, not escalated, per spec example)
    if pre_request_type == "invalid":
        return {
            "Status":        "Replied",
            "product_area":  product_area,
            "response":      OUT_OF_SCOPE_MESSAGE,
            "justification": "Request classified as invalid/out-of-scope based on content analysis.",
            "Request Type":  "invalid",
        }

    # Step 3 — No corpus docs found → escalate
    if not retrieved_docs:
        return {
            "Status":        "Escalated",
            "product_area":  product_area,
            "response":      NO_CORPUS_MESSAGE,
            "justification": "No relevant corpus documents found; escalated for human review.",
            "Request Type":  pre_request_type,
        }

    # Step 4 — Grounded reply from corpus
    response, justification, area = _build_grounded_response(retrieved_docs, domain)

    return {
        "Status":        "Replied",
        "product_area":  area,
        "response":      response,
        "justification": justification,
        "Request Type":  pre_request_type,
    }
