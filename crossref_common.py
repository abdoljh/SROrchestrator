# ==============================================
# crossref_common.py
# Shared Crossref parsing for the engines built on api.crossref.org
# (Crossref/DOI, SSRN, Wiley, Taylor & Francis, ACM, SAGE).
# ==============================================

import re
from typing import Dict, List, Optional, Tuple

from query_utils import clean_abstract

# Fields requested from Crossref (`select=`)
CROSSREF_SELECT = (
    "DOI,title,author,editor,container-title,published-print,published-online,"
    "published,issued,URL,type,abstract,is-referenced-by-count"
)

# Crossref registers DOIs for things that are not papers: peer-review reports,
# decision letters, supplementary files ("component"), etc. They have no
# authors/venue and only clutter the results.
NON_ARTICLE_TYPES = {
    "peer-review", "component", "grant", "dataset", "database",
    "journal", "journal-issue", "journal-volume", "book-series",
    "book-set", "proceedings", "proceedings-series", "report-series", "standard-series",
}
# Encyclopedia / dictionary entries ("Catalysis from A to Z: fluid catalytic
# cracking"): short definitions with no abstract and usually no author.
# Excluded unless the orchestrator config sets include_reference_entries.
INCLUDE_REFERENCE_ENTRIES = False
REFERENCE_TYPES = {"reference-entry"}

_NON_ARTICLE_DOI_RE = re.compile(r"(/v\d+/(review|decision|response)\d*$)|(\.s\d{3,}$)", re.I)
_NON_ARTICLE_TITLE_RE = re.compile(r"^(review for|decision letter for|author response for|reviewer report)\b", re.I)


def is_non_article(entry: Dict) -> bool:
    etype = (entry.get("type") or "").lower()
    if etype in NON_ARTICLE_TYPES:
        return True
    if etype in REFERENCE_TYPES and not INCLUDE_REFERENCE_ENTRIES:
        return True
    if _NON_ARTICLE_DOI_RE.search(entry.get("DOI") or ""):
        return True
    title = first(entry.get("title"))
    return bool(title and _NON_ARTICLE_TITLE_RE.match(title))


def first(value, default: str = "") -> str:
    if isinstance(value, list):
        return str(value[0]).strip() if value else default
    return str(value).strip() if value else default


def crossref_year(entry: Dict) -> str:
    """Crossref always fills `issued`; print/online dates are often missing."""
    for key in ("issued", "published", "published-print", "published-online"):
        parts = (entry.get(key) or {}).get("date-parts") or [[None]]
        if parts and parts[0] and parts[0][0]:
            return str(parts[0][0])
    return "n.d."


def _person_names(people: Optional[List[Dict]]) -> Tuple[List[str], List[str]]:
    """IEEE-style names ('J. Smith') + sort names, tolerating partial records."""
    formatted, sort_names = [], []
    for p in people or []:
        family = (p.get("family") or "").strip()
        given = (p.get("given") or "").strip()
        name = (p.get("name") or "").strip()          # organisations / consortia
        if family and given:
            formatted.append(f"{given[0]}. {family}")
            sort_names.append(f"{family}, {given}")
        elif family:
            formatted.append(family)
            sort_names.append(family)
        elif name:
            formatted.append(name)
            sort_names.append(name)
        elif given:
            formatted.append(given)
            sort_names.append(given)
    return formatted, sort_names


def crossref_authors(entry: Dict) -> Tuple[str, str]:
    """Return (display_authors, sort_key). Falls back to editors for books/chapters."""
    formatted, sort_names = _person_names(entry.get("author"))
    suffix = ""
    if not formatted:
        formatted, sort_names = _person_names(entry.get("editor"))
        if formatted:
            suffix = " (Ed.)" if len(formatted) == 1 else " (Eds.)"
    if not formatted:
        return "Unknown Author", "Unknown"
    if len(formatted) >= 3:
        display = f"{formatted[0]} et al."
    elif len(formatted) == 2:
        display = f"{formatted[0]} and {formatted[1]}"
    else:
        display = formatted[0]
    return display + suffix, sort_names[0]


def parse_crossref_item(entry: Dict, default_venue: str) -> Optional[Dict]:
    """Convert one Crossref work into the SROrch paper schema (None = skip)."""
    if is_non_article(entry):
        return None
    doi = entry.get("DOI") or ""
    authors, sort_key = crossref_authors(entry)
    try:
        cites = int(entry.get("is-referenced-by-count") or 0)
    except (TypeError, ValueError):
        cites = 0
    return {
        "sort_name": sort_key,
        "ieee_authors": authors,
        "title": first(entry.get("title"), "No Title"),
        "venue": first(entry.get("container-title"), default_venue),
        "year": crossref_year(entry),
        "citations": cites,
        "doi": doi or "N/A",
        "url": entry.get("URL") or (f"https://doi.org/{doi}" if doi else ""),
        "abstract": clean_abstract(entry.get("abstract")),
    }
