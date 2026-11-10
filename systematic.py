# ==============================================
# systematic.py
# Systematic-review mode: exhaustive, logged retrieval for a fixed set of
# Boolean search strings (e.g. a review protocol's S1..S6).
# ==============================================
"""
SROrch's normal mode takes the top N results per engine (relevance-ranked), which
is right for exploring a topic but caps recall. For a systematic search this
module pages through *every* hit on the engines that apply a Boolean query
server-side, and reports how many hits each engine said it had:

    exhaustive (paged to the end, or to ``max_records``, which is then reported
    as truncation): OpenAlex (title+abstract), Europe PMC, Semantic Scholar
    (bulk search), PubMed, CORE (API key) and Scopus (API key).

The keyword-only engines (Crossref and the Crossref-backed publisher engines,
DBLP, ...) can still run through ``ResearchOrchestrator.run_search`` with a cap;
they are logged as "capped" so a report never presents them as exhaustive.

Every HTTP request is recorded by ``query_log.RequestRecorder``, so the query
log shows the query each engine actually received.
"""
import csv
import hashlib
import os
import re
import shutil
import time
import zipfile
from datetime import date

import requests

import europe_pmc_utils
import openalex_utils
import pubmed_utils
import s2_utils
import scopus_utils
from query_log import RequestRecorder, engine_context, redact_url
from query_utils import expand_stem, expand_wildcards, to_s2_bulk_query, wildcard_terms, _WILDCARD_RE

# Engine names exactly as ResearchOrchestrator.run_search registers them.
ALL_ENGINES = ("Semantic Scholar", "Google Scholar", "CORE", "SCOPUS", "Springer Nature", "arXiv",
               "PubMed", "Crossref/DOI", "OpenAlex", "Europe PMC", "PLOS", "SSRN", "DeepDyve",
               "Wiley", "Taylor & Francis", "ACM Digital Library", "DBLP", "SAGE Journals")
EXHAUSTIVE_ENGINES = ("OpenAlex", "Europe PMC", "Semantic Scholar", "PubMed", "CORE", "SCOPUS")
# Capped by default: everything else except Google Scholar (each call costs a SerpApi credit).
DEFAULT_CAPPED = tuple(e for e in ALL_ENGINES if e not in EXHAUSTIVE_ENGINES and e != "Google Scholar")

MASTER_KEYS = ['relevance_score', 'source_count', 'ieee_authors', 'title', 'venue', 'year',
               'citations', 'impact_factor', 'h_index', 'influential_citations',
               'doi', 'url', 'abstract', 'abstract_source', 'found_in', 'keywords', 'tldr',
               'recency_boosted']
EXPORT_KEYS = ['string_id', 'record_id'] + MASTER_KEYS
# First six columns match a review protocol's search_log.csv
# (date,database,string_id,query_as_run,hits,notes).
QUERY_LOG_KEYS = ['date', 'database', 'string_id', 'query_as_run', 'hits', 'notes',
                  'mode', 'records_retrieved', 'requests', 'http_status']

OPENALEX_SELECT = ("id,title,publication_year,authorships,primary_location,cited_by_count,doi,"
                   "abstract_inverted_index")
S2_FIELDS = ("paperId,title,year,authors,venue,url,citationCount,influentialCitationCount,"
             "externalIds,abstract")


class EngineResult(object):
    def __init__(self, records, total="", query="", note="", error=""):
        self.records, self.total, self.query, self.note = records, total, query, note
        self.error = error  # paging stopped early: the records so far are kept and it is logged


# ---------------------------------------------------------------- helpers
def http_get(url, params=None, headers=None, tries=4, pause=1.0, timeout=60):
    """GET with backoff on 429/5xx and connection errors. Returns the Response (any status)."""
    for k in range(tries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.exceptions.RequestException:
            if k == tries - 1:
                raise
            time.sleep(pause * 2 ** k)
            continue
        if r.status_code in (429, 500, 502, 503, 504) and k < tries - 1:
            time.sleep(pause * 2 ** k)
            continue
        return r
    return r


def _scrub(text):
    """Error text for the query log: no key or e-mail values (exceptions can quote the URL)."""
    return re.sub(r"(?i)\b(api_key|apikey|key|mailto|email|access_token)=[^&\s'\")]+", r"\1=<redacted>", str(text))


def _fail(engine, r):
    raise RuntimeError("%s HTTP %s: %s" % (engine, r.status_code, (r.text or "")[:160]))


def expand_hyphenated_wildcards(query, max_words=6):
    """Expand only hyphenated stems (co-pyroly*), which Europe PMC cannot truncate;
    plain stems (carboni*, leach*) keep native truncation."""
    def repl(m):
        stem = m.group(1)
        if "-" not in stem:
            return m.group(0)
        words = expand_stem(stem, max_words)
        return "(" + " OR ".join('"%s"' % w for w in words) + ")" if len(words) > 1 else words[0]
    return _WILDCARD_RE.sub(repl, query or "")


def norm_doi(doi):
    s = str(doi or "").strip().lower()
    for p in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/", "doi:"):
        if s.startswith(p):
            s = s[len(p):]
    return "" if s in ("", "n/a", "none", "nan") else s


def record_id(rec):
    """Stable ID from the DOI, else the normalised title: same paper -> same ID across runs."""
    doi = norm_doi(rec.get("doi"))
    key = "doi:" + doi if doi else "t:" + re.sub(r"[^a-z0-9]", "", str(rec.get("title") or "").lower())
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _year(rec):
    m = re.match(r"\s*(\d{4})", str(rec.get("year") or ""))
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------- exhaustive engines
def page_openalex(query, max_records, from_year=None, mailto="", openalex_key="", **_):
    """Cursor-page OpenAlex to the last hit. Paging past page 1 needs the free API key
    (openalex_key, else OPENALEX_API_KEY). If a later page fails, the pages already fetched
    are returned with EngineResult.error set, so the run logs a truncation, not a loss."""
    headers = openalex_utils.auth_headers(openalex_key)
    q = expand_wildcards(query) if wildcard_terms(query) else query  # OpenAlex has no truncation
    params = {"per_page": 200, "cursor": "*", "select": OPENALEX_SELECT}
    filters = []
    note = "title+abstract search"
    if "," in q:  # commas separate OpenAlex filters, so fall back to the general search
        params["search"] = q
        note = "general search (query contains a comma)"
    else:
        filters.append("title_and_abstract.search:" + q)
    if from_year:
        filters.append("from_publication_date:%d-01-01" % from_year)
    if filters:
        params["filter"] = ",".join(filters)
    if mailto:
        params["mailto"] = mailto
    if q != query:
        note += "; wildcards expanded"
    out, total, error = [], "", ""
    while len(out) < max_records:
        try:
            r = http_get("https://api.openalex.org/works", params=params, headers=headers)
            if r.status_code in (401, 403) and headers:
                headers, note = None, note + "; key rejected, no key used"
                r = http_get("https://api.openalex.org/works", params=params)
            if r.status_code != 200:
                _fail("OpenAlex", r)
            data = r.json()
        except Exception as e:
            if not out:
                raise
            error = _scrub(e)[:200]
            if not headers:
                error += " (no working OPENALEX_API_KEY: paging past page 1 needs the free key)"
            break
        total = (data.get("meta") or {}).get("count", total)
        batch = data.get("results") or []
        out += [openalex_utils.process_work(w) for w in batch]
        cursor = (data.get("meta") or {}).get("next_cursor")
        if not batch or not cursor:
            break
        params["cursor"] = cursor
        time.sleep(0.1)
    return EngineResult(out[:max_records], total, q, note, error)


def page_europe_pmc(query, max_records, from_year=None, **_):
    q = expand_hyphenated_wildcards(query)
    if from_year:
        q = "(%s) AND PUB_YEAR:[%d TO 3000]" % (q, from_year)
    params = {"query": q, "format": "json", "pageSize": 1000, "resultType": "core", "cursorMark": "*"}
    out, total, seen = [], "", set()
    while len(out) < max_records:
        r = http_get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params=params)
        if r.status_code != 200:
            _fail("Europe PMC", r)
        data = r.json()
        total = data.get("hitCount", total)
        batch = (data.get("resultList") or {}).get("result") or []
        for e in batch:
            eid = e.get("doi") or e.get("id")
            if eid not in seen:
                seen.add(eid)
                out.append(europe_pmc_utils.process_entry(e))
        nxt = data.get("nextCursorMark")
        if not batch or not nxt or nxt == params["cursorMark"]:
            break
        params["cursorMark"] = nxt
    return EngineResult(out[:max_records], total, q, "all fields (Europe PMC default)")


def page_semantic_scholar(query, max_records, from_year=None, s2_key="", **_):
    params = {"query": to_s2_bulk_query(query), "fields": S2_FIELDS}
    if from_year:
        params["year"] = "%d-" % from_year
    headers = {"x-api-key": s2_key} if s2_key else None
    url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
    out, total, note = [], "", "bulk search (title+abstract)"
    while len(out) < max_records:
        r = http_get(url, params=params, headers=headers)
        if r.status_code in (401, 403) and headers:
            headers, note = None, note + "; key rejected, public pool used"
            r = http_get(url, params=params)
        if r.status_code != 200:
            _fail("Semantic Scholar", r)
        data = r.json()
        total = data.get("total", total)
        batch = data.get("data") or []
        out += [s2_utils.process_paper(p) for p in batch]
        token = data.get("token")
        if not batch or not token:
            break
        params["token"] = token
        time.sleep(1.0)  # the shared pool allows ~1 request/second
    return EngineResult(out[:max_records], total, params["query"], note)


def page_pubmed(query, max_records, from_year=None, **_):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    common = {"tool": "SROrch"}
    if os.getenv("USER_EMAIL"):
        common["email"] = os.getenv("USER_EMAIL")
    p = {**common, "db": "pubmed", "term": query, "retmode": "json", "retmax": 0}
    if from_year:
        p.update({"datetype": "pdat", "mindate": str(from_year), "maxdate": "3000"})
    ids, total, start = [], "", 0
    while len(ids) < max_records:
        p.update({"retstart": start, "retmax": min(5000, max_records - len(ids))})
        r = http_get(base + "/esearch.fcgi", params=p)
        if r.status_code != 200:
            _fail("PubMed", r)
        res = r.json().get("esearchresult", {})
        total = int(res.get("count", 0) or 0)
        batch = res.get("idlist") or []
        ids += batch
        start += len(batch)
        if not batch or start >= total or start >= 9999:  # esearch cannot page past 9,999
            break
        time.sleep(0.4)
    out = []
    for i in range(0, len(ids), 200):
        r = http_get(base + "/efetch.fcgi", params={**common, "db": "pubmed",
                                                    "id": ",".join(ids[i:i + 200]), "retmode": "xml"})
        if r.status_code != 200:
            _fail("PubMed", r)
        out += pubmed_utils.parse_pubmed_xml(r.content)
        time.sleep(0.4)
    return EngineResult(out, total, query, "all fields (PubMed default)")


def page_scopus(query, max_records, from_year=None, **_):
    key = os.getenv("SCOPUS_API_KEY")
    if not key:
        raise RuntimeError("SCOPUS_API_KEY not set")
    headers = {"X-ELS-APIKey": key, "Accept": "application/json"}
    if os.getenv("SCOPUS_INST_TOKEN"):
        headers["X-ELS-Insttoken"] = os.getenv("SCOPUS_INST_TOKEN")
    q = "TITLE-ABS-KEY(%s)" % query + (" AND PUBYEAR > %d" % (from_year - 1) if from_year else "")
    out, total, start = [], "", 0
    while len(out) < max_records and start < 5000:  # the Scopus Search API stops at 5,000
        r = http_get("https://api.elsevier.com/content/search/scopus", headers=headers,
                     params={"query": q, "count": 25, "start": start, "view": "STANDARD"})
        if r.status_code != 200:
            _fail("SCOPUS", r)
        res = r.json().get("search-results", {})
        total = int(res.get("opensearch:totalResults", 0) or 0)
        batch = [e for e in res.get("entry") or [] if "error" not in e]
        out += [scopus_utils.process_entry(e) for e in batch]
        start += 25
        if not batch or start >= total:
            break
    return EngineResult(out[:max_records], total, q, "TITLE-ABS-KEY")


def page_core(query, max_records, core_key="", **_):
    if not core_key:
        raise RuntimeError("CORE_API_KEY not set")
    import core_utils
    recs = core_utils.fetch_and_process_core(core_key, query, max_limit=max_records, save_csv=False)
    return EngineResult(recs or [], "", query, "total hits not reported by this client")


PAGERS = {"OpenAlex": page_openalex, "Europe PMC": page_europe_pmc,
          "Semantic Scholar": page_semantic_scholar, "PubMed": page_pubmed,
          "CORE": page_core, "SCOPUS": page_scopus}


# ---------------------------------------------------------------- one search string
def run_string(orchestrator, string_id, query, exhaustive=EXHAUSTIVE_ENGINES, capped=DEFAULT_CAPPED,
               max_records=2000, capped_limit=25, from_year=None, mailto="", today=None):
    """Run one search string. Returns (records, query_log_rows, request_rows).

    records are the deduplicated, enriched SROrch records with string_id and record_id
    added; query_log_rows has one row per engine (QUERY_LOG_KEYS)."""
    today = today or date.today().isoformat()
    keys = orchestrator.api_keys
    rows, extra = [], []
    with RequestRecorder() as rec:
        for name in exhaustive:
            row = {"date": today, "database": name, "string_id": string_id, "mode": "exhaustive"}
            try:
                with engine_context(name):
                    res = PAGERS[name](query, max_records, from_year=from_year, mailto=mailto,
                                       s2_key=keys.get("s2") or "", core_key=keys.get("core") or "",
                                       openalex_key=keys.get("openalex") or "")
            except Exception as e:
                row.update({"query_as_run": query, "hits": "", "records_retrieved": 0,
                            "notes": "failed: %s" % _scrub(e)[:200]})
                rows.append(row)
                continue
            for p in res.records:
                p["found_in"] = name
            extra += res.records
            n, total = len(res.records), res.total
            note = res.note
            if getattr(res, "error", ""):
                note += "; TRUNCATED: paging stopped after %d of %s records: %s" % (n, total or "?", res.error)
            elif total != "" and n < int(total):
                note += "; TRUNCATED: retrieved %d of %s (raise max_records)" % (n, total)
            row.update({"query_as_run": res.query, "hits": total, "records_retrieved": n, "notes": note})
            rows.append(row)

        orchestrator.config["disabled_engines"] = [e for e in ALL_ENGINES if e not in capped]
        results = orchestrator.run_search(query, limit_per_engine=capped_limit, extra_results=extra)

    sent = {s["engine"]: s for s in rec.summary()}
    failed = [str(f) for f in orchestrator.session_metadata.get("failed_engines", [])]
    for name in capped:
        per = sum(1 for r in results if name in str(r.get("found_in", "")).split("; "))
        s = sent.get(name)
        why = next((f[len(name):].strip(" ()") for f in failed if f.startswith(name + " ")), "")
        rows.append({"date": today, "database": name, "string_id": string_id, "mode": "capped",
                     "query_as_run": " || ".join(s["queries"]) if s else query,
                     "hits": "", "records_retrieved": per,
                     "notes": "relevance-ranked top %d; not exhaustive; count is unique records "
                              "after cross-engine dedup" % capped_limit
                              + ("; failed: " + why[:160] if why else "")
                              + ("" if s else "; no request recorded")})
    for row in rows:
        s = sent.get(row["database"])
        row["requests"] = s["requests"] if s else 0
        row["http_status"] = ";".join(s["statuses"]) if s else ""
        if s and s["queries"] and row["mode"] == "exhaustive":
            row["query_as_run"] = " || ".join(s["queries"])  # what the API really received

    dropped = 0
    if from_year:
        keep = [r for r in results if _year(r) is None or _year(r) >= from_year]
        dropped, results = len(results) - len(keep), keep
    for r in results:
        r["string_id"] = string_id
        r["record_id"] = record_id(r)
    rows.append({"date": today, "database": "SROrch (all engines)", "string_id": string_id,
                 "mode": "merged", "query_as_run": query, "hits": "", "records_retrieved": len(results),
                 "notes": "unique records after dedup" + ("; %d dropped as published before %d"
                                                          % (dropped, from_year) if dropped else ""),
                 "requests": len(rec.rows), "http_status": ""})
    return results, rows, rec.rows


def write_csv(path, rows, keys):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})


def export_string(out_dir, string_id, records, today=None):
    """Write <out>/<SID>_srorch_<date>.csv (one header, one row per unique record)."""
    today = today or date.today().isoformat()
    path = os.path.join(out_dir, "%s_srorch_%s.csv" % (string_id, today))
    write_csv(path, records, EXPORT_KEYS)
    return path


def move_session_dir(orchestrator, out_dir, string_id):
    """Keep run_search's per-query folder (per-engine CSVs) next to the batch outputs."""
    src = getattr(orchestrator, "output_dir", "")
    if src and os.path.isdir(src):
        dst = os.path.join(out_dir, "raw_%s" % string_id)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.move(src, dst)
        return dst
    return ""


REQUEST_LOG_KEYS = ["time", "string_id", "engine", "method", "status", "query_sent", "url"]


def load_strings_text(text):
    """Parse a strings YAML document (``strings: [{id, q}]``, optional ``window: {from_year}``).
    Returns (list of (id, query), from_year or None)."""
    import yaml
    cfg = yaml.safe_load(text) or {}
    if not isinstance(cfg, dict) or not isinstance(cfg.get("strings"), list):
        raise ValueError("expected a YAML mapping with a 'strings:' list of {id, q} items")
    strings = []
    for s in cfg["strings"]:
        if not isinstance(s, dict) or not s.get("id") or not s.get("q"):
            raise ValueError("every item under 'strings:' needs an 'id' and a 'q'")
        strings.append((str(s["id"]), str(s["q"])))
    from_year = (cfg.get("window") or {}).get("from_year")
    return strings, (int(from_year) if from_year else None)


def run_batch(strings, out_dir, exhaustive=EXHAUSTIVE_ENGINES, capped=DEFAULT_CAPPED, max_records=2000,
              capped_limit=25, from_year=None, email="", enrich=True, config=None, keep_raw=True,
              on_string_start=None, on_string_done=None):
    """Run several (id, query) strings into out_dir: <SID>_srorch_<date>.csv per string plus
    QUERY_LOG.csv and REQUESTS_LOG.csv, rewritten after every string so an interrupted batch keeps
    what it has. Shared by srorch_batch.py and the Streamlit 'Systematic' tab.
    Returns (query_log_rows, {string_id: export_path})."""
    from master_orchestrator import ResearchOrchestrator
    overlap = set(exhaustive) & set(capped)
    if overlap:
        raise ValueError("an engine cannot be both exhaustive and capped: %s" % sorted(overlap))
    if email:
        os.environ["USER_EMAIL"] = email  # PubMed and the enrichment chain read it
    os.makedirs(out_dir, exist_ok=True)
    all_rows, all_requests, paths = [], [], {}
    for n, (sid, q) in enumerate(strings):
        if on_string_start:
            on_string_start(n, sid, q)
        orch = ResearchOrchestrator(config=dict(config) if config else None)
        orch.config["enrich_abstracts"] = bool(enrich)
        orch.config["enable_visualization"] = False
        if email:
            orch.api_keys["email"] = email
        records, rows, reqs = run_string(orch, sid, q, exhaustive=exhaustive, capped=capped,
                                         max_records=max_records, capped_limit=capped_limit,
                                         from_year=from_year, mailto=email)
        paths[sid] = export_string(out_dir, sid, records)
        if keep_raw:
            move_session_dir(orch, out_dir, sid)
        elif getattr(orch, "output_dir", "") and os.path.isdir(orch.output_dir):
            shutil.rmtree(orch.output_dir, ignore_errors=True)
        for r in reqs:
            r["string_id"] = sid
        all_rows += rows
        all_requests += reqs
        write_csv(os.path.join(out_dir, "QUERY_LOG.csv"), all_rows, QUERY_LOG_KEYS)
        write_csv(os.path.join(out_dir, "REQUESTS_LOG.csv"), all_requests, REQUEST_LOG_KEYS)
        if on_string_done:
            on_string_done(sid, q, records, rows, paths[sid])
    return all_rows, paths


def zip_run(out_dir, include_raw=False):
    """The run folder as zip bytes (top folder = the run's name). raw_<SID>/ only if asked."""
    import io
    buf = io.BytesIO()
    base = os.path.basename(os.path.normpath(out_dir))
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for folder, _, files in os.walk(out_dir):
            rel = os.path.relpath(folder, out_dir)
            if not include_raw and rel.split(os.sep)[0].startswith("raw_"):
                continue
            for f in sorted(files):
                full = os.path.join(folder, f)
                z.write(full, os.path.join(base, os.path.relpath(full, out_dir)))
    return buf.getvalue()


__all__ = ["run_string", "run_batch", "zip_run", "load_strings_text", "export_string", "write_csv", "record_id", "PAGERS", "ALL_ENGINES",
           "EXHAUSTIVE_ENGINES", "DEFAULT_CAPPED", "EXPORT_KEYS", "QUERY_LOG_KEYS", "redact_url"]
