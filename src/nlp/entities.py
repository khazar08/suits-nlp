"""
Named entity extraction and legal term detection for Suits dialogue.

Uses spaCy en_core_web_sm for PERSON + ORG entities, extended with
Suits-specific firm names and legal terms that the base model may miss.

Outputs per line:
  entities_people    — pipe-separated person names found
  entities_orgs      — pipe-separated organization names found
  legal_term_count   — number of distinct legal terms detected
  legal_terms        — pipe-separated legal terms found
  has_legal_language — bool convenience flag
"""

import re
import spacy
from functools import lru_cache

# ── Legal term lexicon ────────────────────────────────────────────────────────

LEGAL_TERMS: set[str] = {
    # Court & procedure
    "subpoena", "deposition", "injunction", "affidavit", "habeas corpus",
    "prima facie", "voir dire", "contempt", "contempt of court", "perjury",
    "testimony", "verdict", "appeal", "bench", "plaintiff", "defendant",
    "motion", "brief", "discovery", "settlement", "damages", "liability",
    "negligence", "breach", "breach of contract", "indictment", "arraignment",
    "plea", "plea bargain", "probation", "parole", "acquittal", "conviction",
    "due process", "burden of proof", "statute of limitations", "writ",
    "restraining order", "cease and desist", "injunctive relief",
    "summary judgment", "deposition", "cross-examination", "voir dire",
    "opening statement", "closing argument", "sidebar",
    # Law firm / business
    "partner", "associate", "equity partner", "non-equity", "managing partner",
    "senior partner", "partnership track", "pro bono", "retainer",
    "billable hours", "contingency", "merger", "acquisition", "hostile takeover",
    "due diligence", "non-disclosure", "NDA", "non-compete", "fiduciary",
    "fiduciary duty", "conflict of interest", "legal malpractice",
    "bar exam", "bar association", "disbarment", "disciplinary", "ethics",
    # Securities / regulatory
    "SEC", "RICO", "securities fraud", "insider trading", "whistleblower",
    "IPO", "shareholder", "board of directors", "proxy", "arbitration",
    "class action", "tort", "antitrust", "bankruptcy", "Chapter 11",
    # Suits-specific firms / concepts
    "Pearson Specter", "Pearson Hardman", "Pearson Specter Litt",
    "Zane Specter", "Specter Litt", "Wheeler Williams", "Zane Specter Litt",
    "Harvard law", "Harvard", "legal tender", "power of attorney",
    "probate", "estate", "trust fund", "wire transfer", "escrow",
}

# Build a single fast-match regex (longest term first to avoid partial matches)
_sorted_terms = sorted(LEGAL_TERMS, key=len, reverse=True)
_LEGAL_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _sorted_terms) + r")\b",
    re.IGNORECASE,
)

# Suits-specific org patterns that spaCy may miss
_SUITS_ORGS = re.compile(
    r"\b(Pearson\s+(Specter|Hardman|Darby|Specter\s+Litt)|"
    r"Specter\s+Litt(\s+Wheeler\s+Williams)?|"
    r"Zane\s+Specter(\s+Litt)?|"
    r"Wheeler\s+Williams|"
    r"Rand\s+Calder\s+Isaacs|"
    r"Bratton\s+Gould|"
    r"Harvard(\s+Law(\s+School)?)?|"
    r"the\s+SEC|the\s+RICO|the\s+DA['']?s?\s+office)\b",
    re.IGNORECASE,
)

# ── spaCy init ────────────────────────────────────────────────────────────────

_nlp: spacy.Language | None = None


def _get_nlp() -> spacy.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["lemmatizer", "textcat"])
    return _nlp


def score(text: str) -> dict:
    nlp = _get_nlp()
    doc = nlp(text[:512])  # truncate for speed

    # spaCy entities
    people = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]
    orgs   = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]

    # Supplement orgs with Suits-specific patterns
    extra_orgs = _SUITS_ORGS.findall(text)
    orgs = list(dict.fromkeys(orgs + [o[0] if isinstance(o, tuple) else o for o in extra_orgs]))

    # Legal terms
    found_legal = list(dict.fromkeys(
        m.lower() for m in _LEGAL_RE.findall(text)
    ))

    return {
        "entities_people":    "|".join(people) if people else "",
        "entities_orgs":      "|".join(orgs)   if orgs   else "",
        "legal_term_count":   len(found_legal),
        "legal_terms":        "|".join(found_legal) if found_legal else "",
        "has_legal_language": len(found_legal) > 0,
    }


def score_batch(texts: list[str]) -> list[dict]:
    nlp = _get_nlp()
    results = []

    # Use nlp.pipe for batched spaCy processing (much faster)
    truncated = [t[:512] for t in texts]
    for doc, text in zip(nlp.pipe(truncated, batch_size=64), texts):
        people = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]
        orgs   = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]

        extra_orgs = _SUITS_ORGS.findall(text)
        orgs = list(dict.fromkeys(orgs + [o[0] if isinstance(o, tuple) else o for o in extra_orgs]))

        found_legal = list(dict.fromkeys(m.lower() for m in _LEGAL_RE.findall(text)))

        results.append({
            "entities_people":    "|".join(people) if people else "",
            "entities_orgs":      "|".join(orgs)   if orgs   else "",
            "legal_term_count":   len(found_legal),
            "legal_terms":        "|".join(found_legal) if found_legal else "",
            "has_legal_language": len(found_legal) > 0,
        })

    return results
