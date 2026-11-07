"""Run a review protocol's search strings through SROrch in systematic mode.

    python srorch_batch.py --strings path/to/search_strings.yml --out runs/2026-11-05
    python srorch_batch.py --strings search_strings.yml --only S2,S5 --max-records 5000
    python srorch_batch.py --query '"spent FCC" AND sorbent' --string-id S6 --out runs/test

The strings file is YAML with a top-level ``strings:`` list of ``{id, q}`` items
(the cocarb-psp protocol format); an optional ``window: {from_year: 2000}`` is honoured.

Outputs in --out:
  <SID>_srorch_<date>.csv   one header; string_id, record_id, then the MASTER_REPORT columns
  QUERY_LOG.csv             one row per engine per string: date, database, string_id,
                            query_as_run, hits, notes (+ mode, records_retrieved, requests,
                            http_status). The first six columns match search_log.csv.
  REQUESTS_LOG.csv          every HTTP request sent (API keys and e-mail addresses redacted)
  raw_<SID>/                run_search's per-engine CSVs

Google Scholar is off by default (each call costs a SerpApi credit); add it with
--capped "Google Scholar,...". The OpenAlex/PubMed contact e-mail comes from --email or
the USER_EMAIL environment variable and is never written to the outputs.
"""
import argparse
import os
import sys
from datetime import date

import systematic as sm


def load_strings(path):
    with open(path, encoding="utf-8") as fh:
        return sm.load_strings_text(fh.read())


def parse_engines(text, default):
    if text is None:
        return tuple(default)
    names = tuple(n.strip() for n in text.split(",") if n.strip())
    unknown = [n for n in names if n not in sm.ALL_ENGINES]
    if unknown:
        sys.exit("unknown engine(s): %s\nknown: %s" % (unknown, ", ".join(sm.ALL_ENGINES)))
    return names


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--strings", help="YAML file with strings: [{id, q}, ...]")
    src.add_argument("--query", help="a single query (use with --string-id)")
    ap.add_argument("--string-id", default="Q1")
    ap.add_argument("--only", help="comma-separated string IDs to run, e.g. S2,S5")
    ap.add_argument("--out", default="runs/" + date.today().isoformat())
    ap.add_argument("--max-records", type=int, default=2000,
                    help="ceiling per exhaustive engine; hitting it is logged as TRUNCATED")
    ap.add_argument("--capped-limit", type=int, default=25, help="results per capped engine")
    ap.add_argument("--exhaustive", help="exhaustive engines (default: %s)" % ", ".join(sm.EXHAUSTIVE_ENGINES))
    ap.add_argument("--capped", help="capped engines (default: all others except Google Scholar); '' for none")
    ap.add_argument("--from-year", type=int, help="earliest publication year (default: the file's window)")
    ap.add_argument("--email", default=os.getenv("USER_EMAIL", ""), help="contact e-mail for OpenAlex/PubMed")
    ap.add_argument("--no-enrich", action="store_true", help="skip the abstract-enrichment chain")
    a = ap.parse_args(argv)

    if a.strings:
        strings, from_year = load_strings(a.strings)
        if a.only:
            want = {s.strip().upper() for s in a.only.split(",")}
            strings = [s for s in strings if s[0].upper() in want]
    else:
        strings, from_year = [(a.string_id, a.query)], None
    from_year = a.from_year or from_year
    exhaustive = parse_engines(a.exhaustive, sm.EXHAUSTIVE_ENGINES)
    capped = parse_engines(a.capped, sm.DEFAULT_CAPPED)
    overlap = set(exhaustive) & set(capped)
    if overlap:
        sys.exit("an engine cannot be both exhaustive and capped: %s" % sorted(overlap))
    if not strings:
        sys.exit("no search strings selected")

    def report(sid, q, records, rows, path):
        print("\n=== %s: %s" % (sid, q))
        for r in rows:
            print("  %-22s %-10s hits=%-6s retrieved=%-5s %s" % (r["database"], r["mode"], r["hits"],
                                                               r["records_retrieved"], r["notes"][:70]))
        print("  -> %s (%d records)" % (path, len(records)))

    sm.run_batch(strings, a.out, exhaustive=exhaustive, capped=capped, max_records=a.max_records,
                 capped_limit=a.capped_limit, from_year=from_year, email=a.email,
                 enrich=not a.no_enrich, on_string_done=report)
    print("\nlogs: %s" % os.path.join(a.out, "QUERY_LOG.csv"))


if __name__ == "__main__":
    main()
