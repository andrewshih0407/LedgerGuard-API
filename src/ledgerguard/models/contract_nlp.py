"""Module 2 — Contract/Invoice NLP Risk Analyzer.

Approach: zero-shot classification with a small transformer (cross-encoder
style) + regex anchors for high-confidence clause types. This avoids the
need for labelled contract data while remaining robust enough for a POC.

Risk categories detected:
  AUTO_RENEWAL       — auto-renewal clauses without price cap
  PRICE_ESCALATION   — uncapped escalation (CPI, % increases)
  VAGUE_SCOPE        — undefined deliverables / effort-based billing
  TERMINATION_FEE    — early termination penalties
  UNILATERAL_CHANGE  — vendor can unilaterally change terms/pricing
  INDEMNIFICATION    — broad indemnification obligations
  ARBITRATION        — forced arbitration / class-action waiver

Usage
-----
    from ledgerguard.models.contract_nlp import ContractAnalyzer
    analyzer = ContractAnalyzer()
    results = analyzer.analyze("path/to/contract.txt")
    for r in results:
        print(r)
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

RISK_PATTERNS: dict[str, list[str]] = {
    "AUTO_RENEWAL": [
        r"auto(?:matically)?\s+renew",
        r"shall\s+renew\s+(?:automatically|unless)",
        r"evergreen\s+clause",
        r"rolling\s+renewal",
        r"unless\s+(?:written\s+)?notice\s+(?:is\s+)?(?:given|provided|received)",
    ],
    "PRICE_ESCALATION": [
        r"(?:annual|yearly|periodic)\s+(?:price\s+)?(?:increase|escalation|adjustment)",
        r"CPI\s*[\-–]\s*(?:linked|based|adjusted)",
        r"consumer\s+price\s+index",
        r"increase[sd]?\s+by\s+(?:up\s+to\s+)?[\d\.]+\s*%",
        r"may\s+(?:increase|adjust|modify)\s+(?:prices?|fees?|rates?)",
        r"without\s+(?:any\s+)?(?:cap|limit|ceiling)",
    ],
    "VAGUE_SCOPE": [
        r"as\s+(?:deemed\s+)?(?:necessary|appropriate|required)\s+by\s+(?:vendor|supplier|contractor)",
        r"reasonable\s+(?:efforts?|endeavors?)\s+only",
        r"best\s+efforts?\s+(?:basis|only|standard)",
        r"time\s+and\s+materials",
        r"additional\s+work\s+(?:may\s+be\s+)?(?:billed|charged)\s+separately",
        r"scope\s+(?:of\s+work\s+)?(?:may|can|shall)\s+(?:change|vary|expand)",
    ],
    "TERMINATION_FEE": [
        r"early\s+termination\s+(?:fee|penalty|charge)",
        r"termination\s+for\s+convenience.*?(?:fee|penalty|\d+%)",
        r"cancellation\s+(?:fee|charge|penalty)",
        r"liquidated\s+damages",
        r"breakup\s+fee",
    ],
    "UNILATERAL_CHANGE": [
        r"(?:vendor|supplier|company)\s+(?:may|can|reserves?\s+the\s+right\s+to)\s+"
        r"(?:unilaterally\s+)?(?:modify|change|update|amend)\s+"
        r"(?:the\s+)?(?:terms|pricing|fees|agreement|contract)",
        r"(?:thirty|30|sixty|60|ninety|90)\s+days?\s+(?:written\s+)?notice\s+to\s+(?:change|modify|update)",
        r"pricing\s+(?:subject\s+to\s+)?change\s+without\s+(?:prior\s+)?notice",
    ],
    "INDEMNIFICATION": [
        r"indemnif(?:y|ication|ies)\s+(?:and\s+hold\s+harmless)?",
        r"hold\s+harmless",
        r"defend\s+(?:and\s+)?indemnif",
        r"customer\s+shall\s+(?:bear|assume|be\s+responsible\s+for)\s+(?:all\s+)?(?:liability|losses?|damages?|costs?)",
    ],
    "ARBITRATION": [
        r"binding\s+arbitration",
        r"class\s+action\s+waiver",
        r"waive[sd]?\s+(?:the\s+)?right\s+to\s+(?:a\s+)?(?:jury|class)",
        r"disputes?\s+(?:shall|must|will)\s+be\s+(?:resolved|settled)\s+(?:by|through|via)\s+arbitration",
    ],
}

RISK_LEVELS: dict[str, str] = {
    "AUTO_RENEWAL": "HIGH",
    "PRICE_ESCALATION": "HIGH",
    "VAGUE_SCOPE": "MEDIUM",
    "TERMINATION_FEE": "HIGH",
    "UNILATERAL_CHANGE": "HIGH",
    "INDEMNIFICATION": "MEDIUM",
    "ARBITRATION": "MEDIUM",
}

RISK_DESCRIPTIONS: dict[str, str] = {
    "AUTO_RENEWAL": (
        "Contract auto-renews, potentially locking you in without explicit consent. "
        "Ensure a price cap and cancellation notice window are specified."
    ),
    "PRICE_ESCALATION": (
        "Pricing can increase — verify whether a maximum annual cap is stated. "
        "Uncapped escalation clauses create unpredictable future costs."
    ),
    "VAGUE_SCOPE": (
        "Deliverables or effort are not clearly defined. Vague scope enables vendors "
        "to expand billable work without clear authorisation."
    ),
    "TERMINATION_FEE": (
        "Early termination carries a financial penalty. Model the total cost "
        "if you need to exit the contract early."
    ),
    "UNILATERAL_CHANGE": (
        "Vendor can change terms or pricing with limited notice. "
        "Negotiate a mutual-consent amendment clause."
    ),
    "INDEMNIFICATION": (
        "Broad indemnification may expose your organisation to vendor liabilities. "
        "Have legal counsel review scope limits."
    ),
    "ARBITRATION": (
        "Disputes are routed to binding arbitration, waiving your right to a jury trial "
        "or class-action participation."
    ),
}


@dataclass
class ClauseRisk:
    clause_type: str
    risk_level: str          # HIGH / MEDIUM / LOW
    matched_text: str        # the offending sentence/span
    start_char: int
    end_char: int
    description: str         # plain-English explanation
    confidence: float        # 0–1, based on regex confidence vs zero-shot


def _extract_sentence(text: str, match: re.Match) -> tuple[str, int, int]:
    """Return the sentence containing the match."""
    start = text.rfind(".", 0, match.start())
    start = 0 if start < 0 else start + 1
    end = text.find(".", match.end())
    end = len(text) if end < 0 else end + 1
    return text[start:end].strip(), start, end


class ContractAnalyzer:
    """Regex + optional zero-shot transformer contract risk analyzer."""

    def __init__(self, use_transformer: bool = False, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        self.use_transformer = use_transformer
        self._classifier = None
        if use_transformer:
            self._load_transformer(model_name)

    def _load_transformer(self, model_name: str):
        try:
            from transformers import pipeline
            self._classifier = pipeline("zero-shot-classification", model=model_name, device=-1)
            logger.info("Zero-shot classifier loaded: %s", model_name)
        except Exception as e:
            logger.warning("Could not load transformer (%s); falling back to regex only.", e)
            self.use_transformer = False

    def analyze_text(self, text: str) -> list[ClauseRisk]:
        """Detect risky clauses in contract text. Returns list of ClauseRisk."""
        text_lower = text  # keep original case for span extraction
        findings: list[ClauseRisk] = []
        seen_spans: set[tuple[int, int]] = set()

        for clause_type, patterns in RISK_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text_lower, re.IGNORECASE | re.DOTALL):
                    sentence, s, e = _extract_sentence(text, match)
                    # Dedupe overlapping spans
                    span_key = (s, e)
                    if span_key in seen_spans:
                        continue
                    seen_spans.add(span_key)

                    confidence = 0.85  # regex matches are high-confidence
                    if self.use_transformer and self._classifier:
                        confidence = self._transformer_confidence(sentence, clause_type)
                        if confidence < 0.4:
                            continue  # transformer disagrees; skip

                    findings.append(ClauseRisk(
                        clause_type=clause_type,
                        risk_level=RISK_LEVELS[clause_type],
                        matched_text=sentence[:500],
                        start_char=s,
                        end_char=e,
                        description=RISK_DESCRIPTIONS[clause_type],
                        confidence=round(confidence, 3),
                    ))

        # Sort: HIGH first, then by position
        findings.sort(key=lambda f: (0 if f.risk_level == "HIGH" else 1, f.start_char))
        return findings

    def _transformer_confidence(self, text: str, clause_type: str) -> float:
        label_map = {
            "AUTO_RENEWAL": "automatic contract renewal",
            "PRICE_ESCALATION": "price increase or escalation",
            "VAGUE_SCOPE": "vague or undefined scope of work",
            "TERMINATION_FEE": "early termination fee or penalty",
            "UNILATERAL_CHANGE": "vendor right to change terms unilaterally",
            "INDEMNIFICATION": "indemnification or hold harmless",
            "ARBITRATION": "binding arbitration or class action waiver",
        }
        result = self._classifier(text[:512], candidate_labels=[label_map.get(clause_type, clause_type)])
        return result["scores"][0]

    def analyze_file(self, path: Path) -> list[ClauseRisk]:
        text = path.read_text(encoding="utf-8", errors="replace")
        return self.analyze_text(text)

    def report(self, findings: list[ClauseRisk]) -> str:
        """Return a human-readable plain-text report."""
        if not findings:
            return "No high-risk clauses detected."
        lines = [f"Contract Risk Report — {len(findings)} clause(s) flagged\n" + "=" * 55]
        for f in findings:
            lines.append(
                f"\n[{f.risk_level}] {f.clause_type.replace('_', ' ').title()}"
                f" (confidence={f.confidence:.0%})"
            )
            lines.append(f"  {f.description}")
            lines.append(f"  Text: «{f.matched_text[:200]}»")
        return "\n".join(lines)
