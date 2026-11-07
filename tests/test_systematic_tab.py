"""The Streamlit 'Systematic' tab, driven with streamlit.testing (offline; HTTP is faked)."""
import io
import os
import zipfile

from streamlit.testing.v1 import AppTest

from test_systematic import openalex_route

STRINGS = "window: {from_year: 2000}\nstrings:\n  - id: S6\n    q: '\"spent FCC\" AND sorbent'\n  - id: S2\n    q: sludge\n"


def _tab():
    import systematic_tab
    systematic_tab.render_systematic_tab({"email": "researcher@example.com"}, lambda keys: None)


def _app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_function(_tab, default_timeout=60)
    at.run()
    return at


def test_bad_yaml_shows_an_error(fake_http, tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    at.text_area(key="sys_text_area").set_value("strings: nope").run()
    assert at.error and "strings" in at.error[0].value
    assert not at.button  # nothing to run until the strings parse


def test_run_and_download(fake_http, tmp_path, monkeypatch):
    fake_http.route("api.openalex.org", openalex_route(total=3, page_size=2))
    at = _app(tmp_path, monkeypatch)
    at.text_area(key="sys_text_area").set_value(STRINGS).run()
    assert at.number_input(key="sys_year").value == 2000
    at.multiselect(key="sys_ids").set_value(["S6"])
    at.multiselect(key="sys_exh").set_value(["OpenAlex"])
    at.multiselect(key="sys_cap").set_value([])
    at.checkbox(key="sys_enrich").uncheck()
    at.run()
    at.button(key="sys_run_btn").click().run()
    assert not at.exception
    run = at.session_state["sys_run"]
    assert run["counts"] == {"S6": 3}
    log = at.dataframe[-1].value
    assert set(log["database"]) == {"OpenAlex", "SROrch (all engines)"}
    assert int(log.loc[log["database"] == "OpenAlex", "hits"].iloc[0]) == 3
    assert at.get("download_button")  # the zip is offered
    names = zipfile.ZipFile(io.BytesIO(_zip(run))).namelist()
    assert any(n.endswith("QUERY_LOG.csv") for n in names)
    assert any("/S6_srorch_" in n for n in names) and not any("/raw_" in n for n in names)


def _zip(run):
    import systematic as sm
    return sm.zip_run(run["out_dir"])


def test_truncation_warning(fake_http, tmp_path, monkeypatch):
    fake_http.route("api.openalex.org", openalex_route(total=900, page_size=200))
    at = _app(tmp_path, monkeypatch)
    at.text_area(key="sys_text_area").set_value(STRINGS).run()
    at.multiselect(key="sys_ids").set_value(["S2"])
    at.multiselect(key="sys_exh").set_value(["OpenAlex"])
    at.multiselect(key="sys_cap").set_value([])
    at.number_input(key="sys_max").set_value(400)
    at.checkbox(key="sys_enrich").uncheck()
    at.run()
    at.button(key="sys_run_btn").click().run()
    assert at.warning and "S2/OpenAlex" in at.warning[0].value


def test_full_app_shows_the_tab(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "streamlit_app.py")
    at = AppTest.from_file(app, default_timeout=120)
    at.run()
    assert not at.exception
    assert [t.label for t in at.tabs][:3] == ["🔍 Search", "📊 Results", "🧪 Systematic"]
