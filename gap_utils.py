import re
from collections import Counter, defaultdict
from typing import List, Dict, Any


# ---------------------------------------------------------------------
# Gap pattern taxonomy
# ---------------------------------------------------------------------

def get_gap_taxonomy():
    """
    Returns gap patterns grouped by category.
    Each category maps to a list of regex patterns.
    """
    return {
        "performance": [
            r"(?:remains|is) (?:a challenge|an open problem|unsolved)",
            r"(?:fails|struggles) to (?:generalize|scale|converge)",
            r"(?:performance|accuracy) (?:degrades|drops|plateaus)",
            r"unable to (?:reach|achieve|match) (?:human-level|SOTA|state-of-the-art)",
        ],
        "data": [
            r"lack of (?:data|annotations|ground truth)",
            r"dataset is (?:small|limited|biased|imbalanced)",
            r"insufficient training data",
            r"data scarcity",
        ],
        "efficiency": [
            r"computationally (?:expensive|prohibitive|demanding)",
            r"limited by (?:hardware|gpu|memory|vram)",
            r"high (?:latency|energy consumption|memory footprint)",
        ],
        "evaluation": [
            r"lack of (?:standardized|robust) evaluation",
            r"evaluation protocol is unclear",
            r"no fair comparison",
        ],
        "theory": [
            r"lack of theoretical guarantees",
            r"no formal proof",
            r"theoretical understanding is limited",
        ],
        "deployment": [
            r"not suitable for real[- ]world deployment",
            r"deployment remains challenging",
            r"scalability in production is unclear",
        ],
        "ethics_safety": [
            r"ethical concerns",
            r"privacy issues",
            r"bias remains",
            r"safety is not guaranteed",
        ],
    }


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def split_into_sentences(text: str) -> List[str]:
    if not text:
        return []
    return re.split(r'(?<=[.!?])\s+', text)


def compile_patterns(taxonomy: Dict[str, List[str]]):
    compiled = {}
    for category, patterns in taxonomy.items():
        compiled[category] = [re.compile(p, re.IGNORECASE) for p in patterns]
    return compiled


def severity_score(sentence: str) -> int:
    """
    Heuristic severity scoring: 1 (weak) → 5 (critical)
    """
    sentence = sentence.lower()
    score = 1

    strong_terms = [
        "critical", "fundamental", "severe", "unsolved",
        "open problem", "major limitation", "bottleneck"
    ]

    medium_terms = [
        "challenge", "difficult", "limited", "problematic"
    ]

    for t in strong_terms:
        if t in sentence:
            score = max(score, 5)

    for t in medium_terms:
        if t in sentence:
            score = max(score, 3)

    return score


# ---------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------

def analyze_research_gaps(papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Expected paper format:
    {
        "title": "...",
        "abstract": "...",
        "year": 2023,
        "keywords": "a, b, c"
    }
    """

    taxonomy = get_gap_taxonomy()
    compiled = compile_patterns(taxonomy)

    found_gaps = []
    category_counter = Counter()
    yearly_trends = defaultdict(int)
    keyword_counter = Counter()
    co_occurrence = defaultdict(Counter)

    for paper in papers:
        text = f"{paper.get('title', '')}. {paper.get('abstract', '')}"
        sentences = split_into_sentences(text)

        keywords = []
        if paper.get("keywords"):
            keywords = [k.strip().lower() for k in paper["keywords"].split(",")]
            keyword_counter.update(keywords)

        for sent in sentences:
            for category, patterns in compiled.items():
                for pat in patterns:
                    if pat.search(sent):
                        score = severity_score(sent)

                        gap_record = {
                            "title": paper.get("title", ""),
                            "year": paper.get("year", "N/A"),
                            "category": category,
                            "sentence": sent.strip(),
                            "severity": score,
                        }

                        found_gaps.append(gap_record)
                        category_counter[category] += 1

                        if paper.get("year"):
                            yearly_trends[paper["year"]] += 1

                        for kw in keywords:
                            co_occurrence[category][kw] += 1

                        break  # avoid double counting same sentence per category

    return {
        "total_gaps_found": len(found_gaps),
        "gap_list": found_gaps,
        "gap_categories": dict(category_counter),
        "yearly_trends": dict(sorted(yearly_trends.items())),
        "top_keywords": keyword_counter.most_common(15),
        "gap_keyword_cooccurrence": {
            cat: co_occurrence[cat].most_common(10)
            for cat in co_occurrence
        }
    }


# ---------------------------------------------------------------------
# Optional: simple summarizer
# ---------------------------------------------------------------------

def summarize_gaps(analysis_result: Dict[str, Any]) -> str:
    """
    Produces a short human-readable summary.
    """
    total = analysis_result.get("total_gaps_found", 0)
    categories = analysis_result.get("gap_categories", {})
    years = analysis_result.get("yearly_trends", {})

    lines = [
        f"Total gaps identified: {total}",
        "Top gap categories:"
    ]

    for cat, cnt in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  - {cat}: {cnt}")

    if years:
        lines.append("Trend by year:")
        for y, c in years.items():
            lines.append(f"  {y}: {c}")

    return "\n".join(lines)