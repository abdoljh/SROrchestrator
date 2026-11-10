"""Offline tests for systematic mode (query_log.py, systematic.py, srorch_batch.py)."""
import csv
import os
import re
import urllib.parse

import requests

import query_log
import systematic as sm


def qs(request):
    return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(request.url).query))


def oa_work(i, year=2020):
    return {"id": "https://openalex.org/W%d" % i, "title": "Sludge char paper %d" % i,
            "publication_year": year, "doi": "https://doi.org/10.1000/oa.%d" % i,
            "cited_by_count": i, "authorships": [{"author": {"display_name": "Ann Lee"}}],
            "primary_location": {"source": {"display_name": "Fuel"}},
            "abstract_inverted_index": {"Oily": [0], "sludge": [1]}}


def openalex_route(total=3, page_size=2):
    def handler(r):
        cur = qs(r).get("cursor", "*")
        start = 0 if cur == "*" else int(cur)
        works = [oa_work(i) for i in range(start, min(start + page_size, total))]
        nxt = str(start + page_size) if start + page_size < total else None
        return 200, {"meta": {"count": total, "next_cursor": nxt}, "results": works}
    return handler


# ------------------------------------------------------------------ query_log
def test_recorder_attributes_and_redacts(fake_http):
    fake_http.route("example.org", lambda r: (200, {}))
    with query_log.RequestRecorder() as rec:
        with query_log.engine_context("OpenAlex"):
            requests.get("https://example.org/works",
                         params={"search": "oily sludge", "api_key": "SECRET", "mailto": "me@x.org"})
        requests.post("https://example.org/core", json={"q": "spent FCC"})
    assert [r["engine"] for r in rec.rows] == ["OpenAlex", "unattributed"]
    assert rec.rows[0]["query_sent"] == "oily sludge"
    assert rec.rows[1]["query_sent"] == "spent FCC"
    assert "SECRET" not in rec.rows[0]["url"] and "me%40x.org" not in rec.rows[0]["url"]
    assert rec.rows[0]["status"] == 200
    # outside the block nothing is recorded and requests is restored
    requests.get("https://example.org/again")
    assert len(rec.rows) == 2


def test_query_from_known_parameters():
    assert query_log.query_from("https://export.arxiv.org/api/query?search_query=all%3Asludge&start=0") == "all:sludge"
    assert query_log.query_from("https://x.org/esearch?db=pubmed&term=a+AND+b") == "a AND b"


def test_failed_capped_engine_says_so(fake_http, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_http.route("api.crossref.org", lambda r: (503, {"error": "down"}))
    _, rows, _ = sm.run_string(_orchestrator(), "S6", "x", exhaustive=(), capped=("Crossref/DOI",))
    row = next(r for r in rows if r["database"] == "Crossref/DOI")
    assert "failed: no results" in row["notes"] and row["records_retrieved"] == 0


def test_recorder_records_failures(fake_http, monkeypatch):
    def boom(self, request, **kw):
        raise requests.exceptions.ConnectionError("blocked")
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", boom)
    with query_log.RequestRecorder() as rec:
        try:
            requests.get("https://example.org/x", params={"q": "a"})
        except requests.exceptions.ConnectionError:
            pass
    assert rec.rows[0]["status"] == "ConnectionError"


# ------------------------------------------------------------------ pagers
def test_openalex_pages_to_the_end_with_title_abstract_filter(fake_http):
    fake_http.route("api.openalex.org", openalex_route(total=5, page_size=2))
    res = sm.page_openalex('"oily sludge" AND co-pyroly*', 100, from_year=2000, mailto="me@x.org")
    assert len(res.records) == 5 and res.total == 5
    first = qs(fake_http.calls[-3])
    assert first["filter"].startswith("title_and_abstract.search:")
    assert "co-pyroly*" not in first["filter"]  # OpenAlex has no truncation: expanded
    assert "from_publication_date:2000-01-01" in first["filter"]
    assert res.records[0]["doi"] == "10.1000/oa.0" and res.records[0]["abstract"] == "Oily sludge"


def test_openalex_truncation_is_bounded(fake_http):
    fake_http.route("api.openalex.org", openalex_route(total=9, page_size=2))
    res = sm.page_openalex("sludge", 3)
    assert len(res.records) == 3 and res.total == 9


def keyed_openalex_route(total=5, page_size=2, good_key="GOODKEY1"):
    """Like the live API since 2026-11: page 1 is open, later cursors need a valid key."""
    inner = openalex_route(total, page_size)

    def handler(r):
        auth = r.headers.get("Authorization", "")
        if auth and auth != "Bearer " + good_key:
            return 403, {"error": "key_not_activated"}
        if qs(r).get("cursor", "*") != "*" and not auth:
            return 401, {"error": "API key required for cursor paging"}
        return inner(r)
    return handler


def test_openalex_sends_key_as_header_and_pages(fake_http, monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "GOODKEY1")
    fake_http.route("api.openalex.org", keyed_openalex_route())
    with query_log.RequestRecorder() as rec:
        res = sm.page_openalex("sludge", 100, mailto="me@x.org")
    assert len(res.records) == 5 and res.error == ""
    assert all(c.headers["Authorization"] == "Bearer GOODKEY1" for c in fake_http.calls)
    assert not any("GOODKEY1" in c.url for c in fake_http.calls)   # never in a URL
    assert "GOODKEY1" not in str(rec.rows) and "me%40x.org" not in str(rec.rows)
    assert "api_key" in query_log.REDACT


def test_openalex_without_key_keeps_first_page(fake_http, monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    fake_http.route("api.openalex.org", keyed_openalex_route())
    res = sm.page_openalex("sludge", 100)
    assert len(res.records) == 2 and res.total == 5
    assert "HTTP 401" in res.error and "OPENALEX_API_KEY" in res.error


def test_openalex_rejected_key_falls_back_and_keeps_first_page(fake_http, monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    fake_http.route("api.openalex.org", keyed_openalex_route())
    res = sm.page_openalex("sludge", 100, openalex_key="BADKEY99")
    assert len(res.records) == 2 and "key rejected" in res.note and res.error


def test_openalex_first_page_failure_still_raises(fake_http):
    fake_http.route("api.openalex.org", lambda r: (500, {"error": "down"}))
    try:
        sm.page_openalex("sludge", 100)
    except RuntimeError as e:
        assert "OpenAlex HTTP 500" in str(e)
    else:
        raise AssertionError("expected a failure")


def test_paging_failure_is_logged_truncated_with_records(fake_http, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    fake_http.route("api.openalex.org", keyed_openalex_route())
    recs, rows, _ = sm.run_string(_orchestrator(), "S4", "x", exhaustive=("OpenAlex",), capped=())
    assert len(recs) == 2 and rows[0]["records_retrieved"] == 2 and rows[0]["hits"] == 5
    assert "TRUNCATED: paging stopped after 2 of 5 records: OpenAlex HTTP 401" in rows[0]["notes"]
    assert not rows[0]["notes"].startswith("failed")


def test_error_text_is_scrubbed():
    msg = sm._scrub("Max retries: /works?mailto=me%40x.org&api_key=SECRET&per_page=200 (timeout)")
    assert "SECRET" not in msg and "me%40x.org" not in msg and "per_page=200" in msg


def test_europe_pmc_cursor_and_hyphenated_wildcards(fake_http):
    pages = {"*": ("A", [{"id": "1", "doi": "10.1/a", "title": "t1", "authorString": "Lee A"}]),
             "A": ("A", [{"id": "2", "doi": "10.1/b", "title": "t2"}])}

    def h(r):
        nxt, res = pages[qs(r)["cursorMark"]]
        return 200, {"hitCount": 2, "nextCursorMark": nxt, "resultList": {"result": res}}
    fake_http.route("ebi.ac.uk", h)
    res = sm.page_europe_pmc("(co-pyroly* OR carboni*) AND sludge", 100)
    assert [r["doi"] for r in res.records] == ["10.1/a", "10.1/b"] and res.total == 2
    q = qs(fake_http.calls[-1])["query"]
    assert "carboni*" in q and "co-pyroly*" not in q


def test_semantic_scholar_token_paging_and_key_fallback(fake_http):
    def h(r):
        if r.headers.get("x-api-key"):
            return 403, {"message": "Forbidden"}
        tok = qs(r).get("token")
        if not tok:
            return 200, {"total": 3, "token": "T2", "data": [{"title": "a", "externalIds": {"DOI": "10.2/a"}},
                                                               {"title": "b"}]}
        return 200, {"total": 3, "data": [{"title": "c"}]}
    fake_http.route("api.semanticscholar.org", h)
    res = sm.page_semantic_scholar('"oily sludge" AND kinetic*', 100, s2_key="BADKEY", from_year=2000)
    assert [r["title"] for r in res.records] == ["a", "b", "c"] and res.total == 3
    assert "public pool" in res.note
    assert qs(fake_http.calls[-1])["year"] == "2000-"
    assert "+" in res.query  # S2 bulk dialect


PUBMED_XML = """<PubmedArticleSet>{}</PubmedArticleSet>"""
ART = """<PubmedArticle><MedlineCitation><PMID>{0}</PMID><Article><ArticleTitle>Paper {0}</ArticleTitle>
<Journal><Title>Chemosphere</Title><JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue></Journal>
</Article></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">10.3/{0}</ArticleId>
</ArticleIdList></PubmedData></PubmedArticle>"""


def test_pubmed_esearch_then_efetch(fake_http):
    fake_http.route("esearch.fcgi", lambda r: (200, {"esearchresult": {"count": "3", "idlist": ["1", "2", "3"]}}))
    fake_http.route("efetch.fcgi", lambda r: (200, PUBMED_XML.format(
        "".join(ART.format(i) for i in qs(r)["id"].split(",")))))
    res = sm.page_pubmed("oily sludge AND pyrolysis", 100, from_year=2000)
    assert res.total == 3 and [r["doi"] for r in res.records] == ["10.3/1", "10.3/2", "10.3/3"]
    assert qs(fake_http.calls[0])["mindate"] == "2000"


def test_scopus_needs_key(monkeypatch):
    monkeypatch.delenv("SCOPUS_API_KEY", raising=False)
    try:
        sm.page_scopus("x", 10)
    except RuntimeError as e:
        assert "SCOPUS_API_KEY" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


# ------------------------------------------------------------------ helpers
def test_record_id_is_stable():
    a = sm.record_id({"doi": "https://doi.org/10.1/ABC", "title": "x"})
    b = sm.record_id({"doi": "10.1/abc", "title": "different"})
    c = sm.record_id({"doi": "N/A", "title": "Oily  Sludge!"})
    d = sm.record_id({"doi": "", "title": "oily sludge"})
    assert a == b and c == d and a != c and len(a) == 12


def test_engine_names_match_the_orchestrator():
    src = open(os.path.join(os.path.dirname(sm.__file__), "master_orchestrator.py"), encoding="utf-8").read()
    registered = re.findall(r'^\s+\("([^"]+)", (?:\'\w+\'|None), ', src, re.M)
    assert set(registered) == set(sm.ALL_ENGINES)


# ------------------------------------------------------------------ end to end
def _orchestrator():
    from master_orchestrator import ResearchOrchestrator
    o = ResearchOrchestrator()
    o.config.update({"enrich_abstracts": False, "enable_visualization": False, "enable_alerts": False})
    return o


def test_run_string_merges_logs_and_filters_years(fake_http, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def oa(r):
        return 200, {"meta": {"count": 3, "next_cursor": None},
                     "results": [oa_work(1), oa_work(2), oa_work(3, year=1995)]}
    fake_http.route("api.openalex.org", oa)
    fake_http.route("ebi.ac.uk", lambda r: (200, {"hitCount": 1, "resultList": {"result": [
        {"id": "9", "doi": "10.1000/OA.1", "title": "Sludge char paper 1", "pubYear": "2020"}]}}))
    orch = _orchestrator()
    recs, rows, reqs = sm.run_string(orch, "S2", "oily sludge", exhaustive=("OpenAlex", "Europe PMC"),
                                     capped=(), from_year=2000, today="2026-11-05")
    by_doi = {r["doi"].lower(): r for r in recs}
    assert set(by_doi) == {"10.1000/oa.1", "10.1000/oa.2"}  # 1995 dropped; oa.1 deduplicated
    assert "OpenAlex" in by_doi["10.1000/oa.1"]["found_in"] and "Europe PMC" in by_doi["10.1000/oa.1"]["found_in"]
    assert all(r["string_id"] == "S2" and len(r["record_id"]) == 12 for r in recs)
    log = {r["database"]: r for r in rows}
    assert log["OpenAlex"]["hits"] == 3 and log["OpenAlex"]["mode"] == "exhaustive"
    assert log["OpenAlex"]["query_as_run"].startswith("title_and_abstract.search:oily sludge")
    assert log["OpenAlex"]["requests"] == 1 and log["OpenAlex"]["http_status"] == "200"
    assert log["SROrch (all engines)"]["records_retrieved"] == 2
    assert "1 dropped" in log["SROrch (all engines)"]["notes"]


def test_failed_engine_is_logged_not_fatal(fake_http, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_http.route("api.openalex.org", lambda r: (500, {"error": "down"}))
    recs, rows, _ = sm.run_string(_orchestrator(), "S1", "x", exhaustive=("OpenAlex",), capped=())
    assert recs == [] and rows[0]["notes"].startswith("failed: OpenAlex HTTP 500")


def test_truncation_is_reported(fake_http, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_http.route("api.openalex.org", openalex_route(total=7, page_size=2))
    _, rows, _ = sm.run_string(_orchestrator(), "S4", "x", exhaustive=("OpenAlex",), capped=(), max_records=4)
    assert "TRUNCATED: retrieved 4 of 7" in rows[0]["notes"]


def test_capped_engine_rows_say_not_exhaustive(fake_http, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_http.route("api.crossref.org", lambda r: (200, {"message": {"items": [
        {"DOI": "10.5/x", "title": ["Crossref paper"], "type": "journal-article", "author": [{"family": "Lee",
         "given": "A"}], "published": {"date-parts": [[2021]]}}]}}))
    recs, rows, _ = sm.run_string(_orchestrator(), "S6", '"spent FCC" AND sorbent', exhaustive=(),
                                  capped=("Crossref/DOI",), capped_limit=5)
    row = next(r for r in rows if r["database"] == "Crossref/DOI")
    assert row["mode"] == "capped" and "not exhaustive" in row["notes"]
    assert "AND" not in row["query_as_run"]  # Crossref receives plain keywords: now visible in the log
    assert row["requests"] >= 1


def test_batch_cli_writes_single_header_exports_and_logs(fake_http, tmp_path, monkeypatch):
    import srorch_batch
    monkeypatch.chdir(tmp_path)
    fake_http.route("api.openalex.org", openalex_route(total=2, page_size=2))
    yml = tmp_path / "strings.yml"
    yml.write_text('window: {from_year: 2000}\nstrings:\n  - id: S2\n    q: \'"oily sludge" AND kinetic*\'\n'
                   '  - id: S5\n    q: leach*\n', encoding="utf-8")
    out = tmp_path / "run"
    srorch_batch.main(["--strings", str(yml), "--only", "S2", "--out", str(out), "--exhaustive", "OpenAlex",
                       "--capped", "", "--no-enrich", "--email", "owner@example.org"])
    files = sorted(os.listdir(out))
    export = [f for f in files if f.startswith("S2_srorch_")]
    assert export and "QUERY_LOG.csv" in files and "REQUESTS_LOG.csv" in files
    assert not any(f.startswith("S5_") for f in files)
    text = (out / export[0]).read_text(encoding="utf-8")
    assert text.count("string_id,record_id,") == 1
    rows = list(csv.DictReader(open(out / export[0], encoding="utf-8")))
    assert len(rows) == 2 and rows[0]["string_id"] == "S2"
    for name in ("QUERY_LOG.csv", "REQUESTS_LOG.csv"):
        assert "owner@example.org" not in (out / name).read_text(encoding="utf-8")
    log = list(csv.DictReader(open(out / "QUERY_LOG.csv", encoding="utf-8")))
    assert list(log[0])[:6] == ["date", "database", "string_id", "query_as_run", "hits", "notes"]


# ------------------------------------------------------------------ refactor regression
def test_openalex_normal_mode_unchanged(fake_http):
    import openalex_utils
    fake_http.route("api.openalex.org", lambda r: (200, {"results": [oa_work(1)]}))
    recs = openalex_utils.fetch_and_process_openalex("x", max_limit=5, save_csv=False)
    assert recs[0]["ieee_authors"] == "A. Lee" and recs[0]["venue"] == "Fuel" and recs[0]["doi"] == "10.1000/oa.1"


def test_scopus_entry_without_three_links_does_not_crash():
    import scopus_utils
    rec = scopus_utils.process_entry({"dc:title": "t", "prism:coverDate": "2020-01-01", "link": [{"@href": "a"}]})
    assert rec["url"] == "" and rec["year"] == "2020"
