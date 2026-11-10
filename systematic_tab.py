# ==============================================
# systematic_tab.py
# Streamlit tab for systematic-review mode (systematic.py): run a protocol's
# Boolean strings exhaustively and download the run as a zip.
# ==============================================
import os
from datetime import date, datetime

import pandas as pd
import streamlit as st

import systematic as sm

RUNS_DIR = "systematic_runs"
EXAMPLE = """# Paste a strings file, e.g. cocarb-psp's protocol/search_strings.yml
window: {from_year: 2000}
strings:
  - id: S6
    q: '"spent FCC" AND (adsorbent OR sorbent OR composite OR "activated carbon")'
"""


def _parse(text):
    try:
        strings, from_year = sm.load_strings_text(text)
        return strings, from_year, ""
    except Exception as e:  # shown to the user, not raised
        return [], None, str(e)


def render_systematic_tab(api_keys, apply_keys_to_env):
    st.header("Systematic Review Search")
    st.markdown(
        "Runs a review protocol's fixed Boolean strings. **Exhaustive** engines are paged to the last hit "
        "(OpenAlex title+abstract, Europe PMC, Semantic Scholar, PubMed; CORE and Scopus need keys; OpenAlex "
        "needs its free key, `OPENALEX_API_KEY` or the sidebar field, for more than 200 results). "
        "**Capped** engines add a relevance-ranked top N and are logged as not exhaustive. "
        "Every request is logged with the query the API actually received.")

    st.session_state.setdefault("sys_text_area", EXAMPLE)
    up = st.file_uploader("Strings file (YAML)", type=["yml", "yaml"], key="sys_upload")
    if up is not None and st.session_state.get("sys_loaded") != (up.name, up.size):
        st.session_state["sys_text_area"] = up.getvalue().decode("utf-8", errors="replace")
        st.session_state["sys_loaded"] = (up.name, up.size)
    text = st.text_area("Search strings", height=220, key="sys_text_area",
                        help="Format: strings: [{id: S1, q: '...'}], optional window: {from_year: 2000}")
    strings, file_year, err = _parse(text)
    if err:
        st.error("Cannot read the strings: %s" % err)
        return
    st.dataframe(pd.DataFrame(strings, columns=["id", "query"]), hide_index=True, width="stretch")

    ids = [s[0] for s in strings]
    c1, c2 = st.columns(2)
    with c1:
        chosen = st.multiselect("Strings to run", ids, default=ids, key="sys_ids")
        from_year = st.number_input("Earliest publication year (0 = no limit)", 0, 2100,
                                    int(file_year or 0), key="sys_year")
        max_records = st.number_input("Max records per exhaustive engine", 100, 50000, 2000, step=500,
                                      key="sys_max", help="Hitting it is logged as TRUNCATED; then raise it.")
    with c2:
        capped_limit = st.slider("Results per capped engine", 5, 50, 25, 5, key="sys_capped_limit")
        enrich = st.checkbox("Fill missing abstracts (enrichment chain; slower)", True, key="sys_enrich")
        include_raw = st.checkbox("Include per-engine raw CSVs in the zip", False, key="sys_raw")
    with st.expander("Engines"):
        exhaustive = st.multiselect("Exhaustive", list(sm.EXHAUSTIVE_ENGINES), list(sm.EXHAUSTIVE_ENGINES),
                                    key="sys_exh")
        others = [e for e in sm.ALL_ENGINES if e not in exhaustive]
        capped = st.multiselect("Capped", others, [e for e in sm.DEFAULT_CAPPED if e in others], key="sys_cap",
                                help="Google Scholar is off by default: each call costs a SerpApi credit.")
    email = (api_keys.get("email") or "").strip()
    if not email or email == "researcher@example.com":
        st.caption("Tip: set your e-mail in the sidebar; OpenAlex and PubMed ask for a contact address. "
                   "It is never written to the outputs.")
        email = ""

    if st.button("🚀 Run systematic search", type="primary", disabled=not chosen, key="sys_run_btn"):
        selected = [s for s in strings if s[0] in chosen]
        apply_keys_to_env(api_keys)
        out_dir = os.path.join(RUNS_DIR, "%s_%s" % (date.today().isoformat(), datetime.now().strftime("%H%M%S")))
        bar, status = st.progress(0.0), st.empty()

        def start(n, sid, q):
            status.text("Running %s (%d of %d)... exhaustive paging can take a few minutes" % (sid, n + 1, len(selected)))
            bar.progress(n / len(selected))

        try:
            rows, paths = sm.run_batch(selected, out_dir, exhaustive=tuple(exhaustive), capped=tuple(capped),
                                       max_records=int(max_records), capped_limit=int(capped_limit),
                                       from_year=int(from_year) or None, email=email, enrich=enrich,
                                       on_string_start=start)
            bar.progress(1.0)
            status.text("Done.")
            st.session_state["sys_run"] = {"out_dir": out_dir, "rows": rows,
                                           "counts": {k: sum(1 for _ in open(p, encoding="utf-8")) - 1
                                                      for k, p in paths.items()}}
        except Exception as e:
            st.error("Run failed: %s" % e)
            if os.path.isdir(out_dir):
                st.session_state["sys_run"] = {"out_dir": out_dir, "rows": [], "counts": {}}

    run = st.session_state.get("sys_run")
    if not run:
        return
    st.subheader("Last run")
    if run["counts"]:
        st.write(", ".join("**%s**: %d records" % kv for kv in run["counts"].items()))
    log = pd.DataFrame(run["rows"], columns=sm.QUERY_LOG_KEYS)
    if len(log):
        trunc = log["notes"].astype(str).str.contains("TRUNCATED")
        failed = log["notes"].astype(str).str.contains("failed")
        if trunc.any():
            st.warning("Truncated: %s. See the notes: raise the max records, or fix the paging error "
                       "(e.g. a missing OpenAlex key), and re-run those strings."
                       % ", ".join("%s/%s" % (r.string_id, r.database) for r in log[trunc].itertuples()))
        if failed.any():
            st.info("Failed engines: %s" % ", ".join("%s/%s" % (r.string_id, r.database)
                                                    for r in log[failed].itertuples()))
        st.dataframe(log, hide_index=True, width="stretch")
    name = os.path.basename(run["out_dir"])
    st.download_button("⬇️ Download run (zip)", sm.zip_run(run["out_dir"], include_raw=include_raw),
                       file_name="srorch_run_%s.zip" % name, mime="application/zip", key="sys_dl")
    st.caption("Import it in the Cocarb app (Import tab) or with scripts/import_srorch_run.py.")
