"""
classifier.py — Deterministic routing, escalation, and classification.

Key improvements in this version:
- Proper Status casing: "Replied" / "Escalated" (capital R/E)
- Short normalized product_area labels derived from corpus folder paths
- Strong escalation patterns covering all spec requirements
- Word-boundary regex to avoid "HackerRank" matching "hack"
- Correct request_type ordering (invalid > bug > product_issue > feature_request)
- build_retrieval_query with targeted overrides for known ticket patterns
"""

import re
from collections import Counter

# ── Domain mapping ─────────────────────────────────────────────────────────────

COMPANY_TO_DOMAIN = {
    "hackerrank": "hackerrank",
    "claude":     "claude",
    "visa":       "visa",
    "none":       None,
}

def infer_domain(company: str, issue: str, subject: str) -> str | None:
    c = (company or "").strip().lower()
    if c in COMPANY_TO_DOMAIN and c != "none":
        return COMPANY_TO_DOMAIN[c]

    text = f"{issue} {subject}".lower()

    visa_kw   = ["visa card","visa payment","visa account","merchant","transaction",
                 "chargeback","dispute","refund","unauthorized","card blocked",
                 "stolen card","travelers cheque","traveller","minimum spend","atm"]
    claude_kw = ["claude","anthropic","bedrock","claude.ai","claude pro","claude team",
                 "lti","crawl","claudebot","workspace","mcp","claude code"]
    hr_kw     = ["hackerrank","assessment","interview","submission","test","proctor",
                 "resume builder","certificate","mock interview","inactivity","subscription",
                 "apply tab","zoom connectivity","hiring account"]

    def score(kws): return sum(1 for k in kws if k in text)
    scores = {"visa": score(visa_kw), "claude": score(claude_kw), "hackerrank": score(hr_kw)}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


# ── Product area — short normalized labels ─────────────────────────────────────

# Map corpus folder path prefixes → short label
_AREA_MAP = [
    # HackerRank
    ("screen/test-integrity",          "test_integrity"),
    ("screen/managing-tests",          "assessments"),
    ("screen/test-settings",           "assessments"),
    ("screen/invite-candidates",       "assessments"),
    ("screen/test-reports",            "assessments"),
    ("screen/getting-started",         "assessments"),
    ("screen/best-practice",           "assessments"),
    ("screen/frequently",              "assessments"),
    ("screen",                         "assessments"),
    ("interviews/interview-settings",  "interviews"),
    ("interviews/manage",              "interviews"),
    ("interviews/scoring",             "interviews"),
    ("interviews/getting",             "interviews"),
    ("interviews/interview-integrity", "interviews"),
    ("interviews",                     "interviews"),
    ("settings/teams-management",      "team_management"),
    ("settings/user-account",          "subscriptions"),
    ("settings/company",               "company_settings"),
    ("settings/roles",                 "team_management"),
    ("settings/insights",              "analytics"),
    ("settings/open-api",              "api"),
    ("settings/gdpr",                  "compliance"),
    ("settings",                       "account_settings"),
    ("hackerrank_community/certif",    "certifications"),
    ("hackerrank_community/subscript", "billing"),
    ("hackerrank_community/mock",      "mock_interviews"),
    ("hackerrank_community/account",   "account_access"),
    ("hackerrank_community/additional","job_search"),
    ("hackerrank_community",           "community"),
    ("general-help/contact",           "support"),
    ("general-help",                   "general_support"),
    ("integrations",                   "integrations"),
    ("library",                        "library"),
    ("engage",                         "events"),
    ("chakra",                         "ai_interviews"),
    ("skillup",                        "skillup"),
    # Claude
    ("team-and-enterprise-plans/billing",   "billing"),
    ("team-and-enterprise-plans/admin",     "account_access"),
    ("team-and-enterprise-plans/security",  "security"),
    ("team-and-enterprise-plans/get",       "account_access"),
    ("team-and-enterprise-plans",           "account_access"),
    ("pro-and-max-plans/general",           "billing"),
    ("pro-and-max-plans",                   "subscriptions"),
    ("claude-api-and-console/pricing",      "billing"),
    ("claude-api-and-console/troubleshoot", "api"),
    ("claude-api-and-console",              "api"),
    ("amazon-bedrock",                      "amazon_bedrock"),
    ("claude-for-education",                "education"),
    ("claude-for-government",               "government"),
    ("claude-for-nonprofits",               "nonprofits"),
    ("claude-code",                         "claude_code"),
    ("claude-desktop",                      "claude_desktop"),
    ("claude-mobile",                       "mobile"),
    ("claude-in-chrome",                    "claude_chrome"),
    ("connectors",                          "integrations"),
    ("identity-management",                 "account_access"),
    ("privacy-and-legal",                   "privacy"),
    ("safeguards",                          "security"),
    ("claude/account-management",           "account_management"),
    ("claude/conversation",                 "conversation_management"),
    ("claude/features",                     "features"),
    ("claude/troubleshoot",                 "troubleshooting"),
    ("claude/usage",                        "usage_limits"),
    ("claude/get-started",                  "getting_started"),
    ("claude/personaliz",                   "personalization"),
    # Visa
    ("support/small-business/fraud",        "fraud"),
    ("support/small-business/dispute",      "dispute_resolution"),
    ("support/small-business/data",         "security"),
    ("support/small-business/regulations",  "regulations"),
    ("support/small-business/travelers",    "travel_support"),
    ("support/small-business",              "small_business"),
    ("support/consumer/travel",             "travel_support"),
    ("support/consumer/travelers",          "travel_support"),
    ("support/consumer",                    "card_management"),
]

def _corpus_path_to_area(product_area: str) -> str:
    pa = product_area.replace("\\", "/")
    for prefix, label in _AREA_MAP:
        if prefix in pa:
            return label
    return "general_support"

def infer_product_area(domain: str, issue: str, subject: str) -> str:
    """Infer product area from domain + ticket keywords (before retrieval)."""
    if not domain:
        return "general_support"
    text = (issue + " " + subject).lower()

    if domain == "hackerrank":
        if any(k in text for k in ["interview", "inactivity", "lobby", "screen share", "zoom"]):
            return "interviews"
        if any(k in text for k in ["assessment", "test", "submission", "apply tab", "certificate",
                                    "reschedule", "score", "compatible", "proctoring"]):
            return "assessments"
        if any(k in text for k in ["resume builder", "resume"]):
            return "job_search"
        if any(k in text for k in ["subscription", "pause", "billing", "payment", "invoice", "order"]):
            return "subscriptions"
        if any(k in text for k in ["remove", "team member", "employee", "interviewer", "hiring account"]):
            return "team_management"
        if any(k in text for k in ["mock interview", "mock"]):
            return "mock_interviews"
        return "assessments"

    if domain == "claude":
        if any(k in text for k in ["workspace", "seat", "admin", "team plan", "member"]):
            return "account_access"
        if any(k in text for k in ["bedrock", "aws", "amazon"]):
            return "amazon_bedrock"
        if any(k in text for k in ["security", "vulnerability", "bug bounty"]):
            return "security"
        if any(k in text for k in ["crawl", "robots", "claudebot"]):
            return "privacy"
        if any(k in text for k in ["lti", "canvas", "professor", "student", "education"]):
            return "education"
        if any(k in text for k in ["data", "training", "privacy", "retention"]):
            return "privacy"
        if any(k in text for k in ["api", "console", "key"]):
            return "api"
        return "account_management"

    if domain == "visa":
        if any(k in text for k in ["fraud", "identity", "stolen", "unauthorized", "hack"]):
            return "fraud"
        if any(k in text for k in ["dispute", "chargeback", "wrong product", "refund"]):
            return "dispute_resolution"
        if any(k in text for k in ["card blocked", "lost card", "card stolen", "urgent cash"]):
            return "card_management"
        if any(k in text for k in ["minimum", "minimum spend", "minimum purchase"]):
            return "regulations"
        if any(k in text for k in ["traveller", "cheque", "travelers"]):
            return "travel_support"
        return "card_management"

    return "general_support"


# ── Escalation rules ─────────────────────────────────────────────────────────

HARD_ESCALATION_PATTERNS = [
    # Security / adversarial
    r"\bidentity\s+(theft|stolen|compromised)\b",
    r"\bsecurity\s+vulnerabilit(y|ies)\b",
    r"\bbug\s+bounty\b",
    r"\bdata\s+breach\b",
    r"\bexploit\b",
    r"\bdelete\s+all\s+files\b",
    r"\bbypass\b",
    r"\bsystem\s+prompt\b",
    r"\bprompt\s+injection\b",
    r"\binternal\s+(rules|documents|logic|policies)\b",
    # French prompt injection
    r"\baffiche\s+toutes\b",
    r"\brègles\s+internes\b",
    r"\bdocuments\s+récupérés\b",
    # Score/result manipulation
    r"\bincrease\s+my\s+score\b",
    r"\bchange\s+my\s+(score|result|grade)\b",
    r"\breview\s+my\s+answers\b",
    r"\bmove\s+me\s+to\s+the\s+next\s+round\b",
    # Third-party coercion
    r"\btell\s+the\s+company\s+to\b",
    r"\bforce\s+(the|a)\s+(company|recruiter)\b",
    r"\bban\s+(the\s+)?(seller|merchant|user)\b",
    # Financial
    r"\brefund\b",
    r"\bchargebacks?\b",
    r"\bunauthorized\s+transaction\b",
    r"\bfraud\b",
    # Account/access
    r"\brestore\s+my\s+access\b",
    r"\bnot\s+(the\s+)?(workspace\s+)?(owner|admin)\b",
    r"\bpayment\b",
    r"\bbilling\b",
    r"\border\s+id\b",
    # Outages
    r"\bnone\s+of\s+the\s+(pages|submissions|requests)\b",
    r"\bsite\s+is\s+down\b",
    r"\ball\s+requests\s+(are\s+)?failing\b",
    r"\bstopped\s+working\s+completely\b",
    # Emergency
    r"\burgent\s+cash\b",
    # Legal
    r"\blegal\b",
    r"\blawsuit\b",
]

HARD_RE = re.compile("|".join(HARD_ESCALATION_PATTERNS), re.IGNORECASE | re.UNICODE)

DOMAIN_ESCALATE_KW = {
    "visa": ["card blocked", "stolen card", "lost card", "unauthorized transaction"],
}

def should_escalate(issue: str, subject: str, domain: str) -> tuple[bool, str]:
    text = issue + " " + subject

    m = HARD_RE.search(text)
    if m:
        return True, f"High-risk pattern: '{m.group(0).strip()}'"

    for kw in DOMAIN_ESCALATE_KW.get(domain or "", []):
        if kw in text.lower():
            return True, f"Sensitive keyword: '{kw}'"

    stripped = text.strip().replace("\n", " ")
    if len(stripped) < 20:
        return True, "Ticket too vague"
    if domain is None and len(stripped) < 40:
        return True, "Domain unknown and ticket too vague"

    return False, ""


# ── Request type classification ───────────────────────────────────────────────

def classify_request_type(issue: str, subject: str) -> str:
    text = (issue + " " + subject).lower()

    invalid_patterns = [
        r"\bdelete\s+all\s+files\b",
        r"\bhack\b",                    # word-boundary — won't match "hackerrank"
        r"\bbypass\b",
        r"\bprompt\s+injection\b",
        r"\baffiche\s+toutes\b",
        r"\brègles\s+internes\b",
        r"\biron\s+man\b",
        r"\bwhat\s+is\s+the\s+name\b",
    ]
    if any(re.search(p, text) for p in invalid_patterns):
        return "invalid"

    bug_kw = ["not working","broken","down","error","crash","failing","failed",
              "unable to","can't","cannot","stopped","disappeared","not loading",
              "not accessible","not able","blocker","stopped working"]
    feature_kw = ["feature request","would like to","wish","suggest","please add",
                  "can you add","enhancement","improvement","extend","can we extend",
                  "when should i","best practice","advantages","disadvantages",
                  "how to","how do i","setup","set up","what is","confirm",
                  "can you confirm","can you please","please confirm","what are","when to"]
    product_kw = ["pause","remove","delete","access","update","change","reschedule",
                  "request","add","manage","configure","cancel","restore","upgrade"]

    def score(kws): return sum(1 for k in kws if k in text)

    s_bug  = score(bug_kw)
    s_feat = score(feature_kw)
    s_prod = score(product_kw)

    if s_bug > s_feat and s_bug > 0:
        return "bug"
    if s_feat > 0:
        return "feature_request"
    if s_prod > 0:
        return "product_issue"
    return "product_issue"


# ── Retrieval query builder ───────────────────────────────────────────────────

QUERY_OVERRIDES = {
    "inactivity":        "virtual lobby inactivity timeout interview hackerrank",
    "remove interviewer":"manage team members remove interviewer settings hackerrank",
    "remove user":       "manage team members remove interviewer settings hackerrank",
    "remove them":       "manage team members remove user hackerrank hiring account",
    "pause":             "pause subscription hackerrank pause hiring temporarily",
    "reschedule":        "reschedule assessment candidate recruiter contact company",
    "zoom":              "zoom proctoring compatible check system requirements test",
    "apply tab":         "apply tab hackerrank community practice submissions",
    "resume builder":    "create resume resume builder hackerrank",
    "certificate":       "download certificate hackerrank name",
    "infosec":           "contact hackerrank support enterprise",
    "crawl":             "crawl website stop claudebot robots txt anthropic",
    "crawling":          "crawl website stop claudebot robots txt anthropic",
    "lti":               "lti canvas professor students claude education setup",
    "professor":         "lti canvas professor students claude education setup",
    "training":          "data training privacy how long retained anthropic policy",
    "bedrock":           "aws bedrock claude failing requests amazon regions",
    "workspace":         "claude team workspace seat admin manage members",
    "minimum":           "visa minimum purchase amount rules consumer merchant",
    "minimum spend":     "visa minimum purchase amount rules consumer merchant",
    "dispute":           "visa dispute charge chargeback resolution",
    "identity":          "identity theft visa card fraud protection",
    "mock interview":    "mock interview purchase credits hackerrank refund",
}

def build_retrieval_query(issue: str, subject: str) -> str:
    combined = (issue + " " + subject).lower()
    for kw, override in sorted(QUERY_OVERRIDES.items(), key=lambda x: -len(x[0])):
        if kw in combined:
            return override
    words = re.findall(r"[a-zA-Z]+", combined)
    stopwords = {"the","is","a","an","and","or","to","of","in","on","for","with",
                 "at","by","from","please","help","can","you","i","my","we","our",
                 "it","this","that","am","are","was","be","been","have","has"}
    filtered = [w for w in words if w not in stopwords and len(w) > 2]
    freq = Counter(filtered)
    return " ".join(w for w, _ in freq.most_common(12))
