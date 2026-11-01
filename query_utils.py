# ==============================================
# query_utils.py
# Shared helpers for SROrch search engines:
#   * filesystem-safe slugs for (arbitrarily long) queries
#   * boolean-query parsing + per-engine query dialects
#   * abstract extraction / cleaning
# ==============================================

import hashlib
import html
import re
from typing import Dict, List, Optional

# ----------------------------------------------
# 1. Filesystem-safe names
# ----------------------------------------------

# Most filesystems cap a single path component at 255 *bytes*. We stay far
# below that so engine prefixes and timestamps always fit.
MAX_SLUG_LEN = 60


def safe_query_slug(query: str, max_len: int = MAX_SLUG_LEN) -> str:
    """
    Turn any query (including long boolean expressions) into a short,
    filesystem-safe slug.

    Long queries are truncated on a word boundary and suffixed with a short
    hash of the full query, so different long queries never collide.

    >>> safe_query_slug('machine learning')
    'machine_learning'
    """
    query = query or ""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", query).strip("_") or "query"
    if len(slug) <= max_len:
        return slug

    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:8]
    room = max_len - len(digest) - 1
    head = slug[:room]
    # Prefer cutting on a word boundary
    if "_" in head[room // 2:]:
        head = head[:head.rfind("_")]
    return f"{head.strip('_')}_{digest}"


# ----------------------------------------------
# 2. Boolean query handling
# ----------------------------------------------

_BOOL_RE = re.compile(r'\b(AND|OR|NOT)\b|["()*]')
_TOKEN_RE = re.compile(r'"[^"]*"|\(|\)|[^\s()"]+')


def is_boolean_query(query: str) -> bool:
    """True if the query uses boolean operators, phrases, grouping or wildcards."""
    return bool(_BOOL_RE.search(query or ""))


def tokenize_query(query: str) -> List[str]:
    """Split a query into phrases ("..."), parentheses, operators and words."""
    return _TOKEN_RE.findall(query or "")


def to_plain_keywords(query: str) -> str:
    """
    Strip boolean syntax and return de-duplicated keywords, for engines that
    only accept free text (e.g. Crossref's relevance search).
    Wildcards are dropped (``co-pyroly*`` -> ``co-pyroly``).
    """
    words, seen = [], set()
    for tok in tokenize_query(query):
        if tok in ("(", ")") or tok in ("AND", "OR", "NOT"):
            continue
        for w in tok.strip('"').replace("*", "").split():
            w = w.strip(",;")
            if w and w.lower() not in seen:
                seen.add(w.lower())
                words.append(w)
    return " ".join(words)


def to_s2_bulk_query(query: str) -> str:
    """
    Convert a standard boolean query to Semantic Scholar /paper/search/bulk
    syntax: ``+`` AND, ``|`` OR, ``-`` NOT, ``"phrase"``, ``prefix*``, ``( )``.

    A literal word like "AND" would otherwise be searched as a term, which
    makes a long boolean query return nothing.
    """
    if not is_boolean_query(query):
        return query
    out = []
    for tok in tokenize_query(query):
        if tok == "AND":
            out.append("+")
        elif tok == "OR":
            out.append("|")
        elif tok == "NOT":
            out.append("-")
        elif not tok.startswith('"') and "-" in tok.strip("-"):
            # Hyphenated word (co-pyroly*): a mid-word '-' could be read as NOT,
            # so split it into an explicit AND group.
            parts = [p for p in tok.split("-") if p]
            out.append("(" + " + ".join(parts) + ")")
        else:
            out.append(tok)
    return re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", " ".join(out)))


def split_and_groups(query: str) -> List[List[str]]:
    """
    Best-effort parse of ``(a OR b) AND (c OR d) ...`` into ``[[a, b], [c, d]]``.
    Terms outside any parenthesis form their own group.
    """
    groups: List[List[str]] = []
    current: List[str] = []
    depth = 0
    for tok in tokenize_query(query):
        if tok == "(":
            depth += 1
            continue
        if tok == ")":
            depth = max(depth - 1, 0)
            if depth == 0 and current:
                groups.append(current)
                current = []
            continue
        if tok in ("AND", "OR", "NOT"):
            if tok == "AND" and depth == 0 and current:
                groups.append(current)
                current = []
            continue
        current.append(tok.strip('"'))
    if current:
        groups.append(current)
    return groups


def to_dblp_query(query: str) -> str:
    """
    DBLP: whitespace = AND, ``|`` = OR, prefix matching by default, no phrases.
    Each AND group becomes ``word1|word2|...``.
    """
    if not is_boolean_query(query):
        return query
    parts = []
    for group in split_and_groups(query):
        words, seen = [], set()
        for term in group:
            for w in term.replace("*", "").split():
                if len(w) > 2 and w.lower() not in seen:
                    seen.add(w.lower())
                    words.append(w)
        if words:
            parts.append("|".join(words))
    return " ".join(parts) or to_plain_keywords(query)


# ----------------------------------------------
# Wildcard expansion (for engines that ignore trailing '*')
# ----------------------------------------------

_WILDCARD_RE = re.compile(r'(?<!["\w])([A-Za-z][\w-]*?)\*')
# Fallback endings when no word list is reachable; covers common scientific forms
# (pyroly* -> pyrolysis/pyrolytic/pyrolyzed, catal* -> catalyst/catalytic, ...)
_FALLBACK_SUFFIXES = ["", "s", "sis", "tic", "se", "sed", "ze", "zed", "ses", "zing", "ed",
                      "ing", "ion", "ions", "ic", "ical", "er", "ers", "y", "ies"]
_EXPANSION_CACHE: Dict[str, List[str]] = {}


def wildcard_terms(query: str) -> List[str]:
    """Return the wildcard stems in a query, e.g. ['co-pyroly', 'copyroly']."""
    return list(dict.fromkeys(_WILDCARD_RE.findall(query or "")))


def _lookup_words(stem: str, max_words: int) -> List[str]:
    """Real English/scientific words starting with `stem` (Datamuse, no key needed)."""
    try:
        import requests
        r = requests.get("https://api.datamuse.com/words",
                         params={"sp": f"{stem.lower()}*", "max": 40}, timeout=6)
        if r.status_code == 200:
            words = [w["word"] for w in r.json() if " " not in w.get("word", "")]
            return words[:max_words]
    except Exception:
        pass
    return []


def expand_stem(stem: str, max_words: int = 6) -> List[str]:
    """
    Expand a truncated term into whole words. Hyphenated stems keep their prefix:
    'co-pyroly' -> ['co-pyrolysis', 'co-pyrolytic', ...].
    """
    key = f"{stem.lower()}|{max_words}"
    if key in _EXPANSION_CACHE:
        return _EXPANSION_CACHE[key]
    prefix, _, last = stem.rpartition("-")
    prefix = prefix + "-" if prefix else ""
    words = _lookup_words(last, max_words) if len(last) >= 3 else []
    if not words:
        words = [last + suf for suf in _FALLBACK_SUFFIXES][:max(max_words, 8)]
    out = list(dict.fromkeys(prefix + w for w in words))
    _EXPANSION_CACHE[key] = out
    return out


def expand_wildcards(query: str, max_words: int = 6) -> str:
    """
    Replace every ``stem*`` with ``(word1 OR word2 ...)``. Used to retry
    engines that returned nothing because they ignore truncation
    (OpenAlex, Google Scholar, Europe PMC with hyphenated stems, ...).
    """
    def repl(m):
        words = expand_stem(m.group(1), max_words)
        return "(" + " OR ".join(f'"{w}"' if "-" in w else w for w in words) + ")" if len(words) > 1 else words[0]
    return _WILDCARD_RE.sub(repl, query or "")


def to_springer_query(query: str) -> str:
    """Springer Meta API accepts boolean expressions directly."""
    if is_boolean_query(query):
        return query
    return f'keyword:"{query}"'


def to_plos_query(query: str) -> str:
    """PLOS Solr: search title and abstract with the full boolean expression."""
    if is_boolean_query(query):
        return f"title:({query}) OR abstract:({query})"
    escaped = query.replace('"', '\\"')
    return f'title:"{escaped}" OR abstract:"{escaped}"'


# ----------------------------------------------
# 3. Abstract helpers
# ----------------------------------------------

NO_ABSTRACT = "Abstract not available."

_TAG_RE = re.compile(r"<[^>]+>")
_LEADING_LABEL_RE = re.compile(r"^\s*(abstract|summary)\s*[:.\-–]?\s*", re.I)


def clean_abstract(text) -> str:
    """
    Normalise an abstract coming from any API: strip JATS/HTML tags
    (Crossref returns ``<jats:p>``), unescape entities, drop a leading
    "Abstract" label and collapse whitespace. Returns '' when empty.
    """
    if not text:
        return ""
    if isinstance(text, (list, tuple)):
        text = " ".join(str(t) for t in text if t)
    text = str(text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _LEADING_LABEL_RE.sub("", text, count=1).strip()
    if text.lower() in ("", "n/a", "none", "null", NO_ABSTRACT.lower()):
        return ""
    return text


def reconstruct_inverted_index(inv_index: Optional[Dict[str, List[int]]]) -> str:
    """Rebuild OpenAlex ``abstract_inverted_index`` into plain text."""
    if not inv_index:
        return ""
    positions = []
    for word, idxs in inv_index.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return clean_abstract(" ".join(w for _, w in positions))


def has_abstract(paper: Dict) -> bool:
    return bool(clean_abstract(paper.get("abstract")))
