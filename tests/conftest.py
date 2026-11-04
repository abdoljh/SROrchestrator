"""Test setup: put the repo root on sys.path, stub the optional SerpApi client, and
provide `fake_http`, which answers every `requests` call offline (no network)."""
import json
import os
import sys
import types

import pytest
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:  # google-search-results does not build everywhere; scholar_utils only needs the name
    import serpapi.google_search  # noqa: F401
except Exception:
    mod = types.ModuleType("serpapi.google_search")
    mod.GoogleSearch = object
    pkg = sys.modules.get("serpapi") or types.ModuleType("serpapi")
    pkg.google_search = mod
    sys.modules["serpapi"], sys.modules["serpapi.google_search"] = pkg, mod


class FakeHTTP:
    """Routes requests to handler(request) -> (status, body). body: dict/list -> JSON, bytes/str raw."""

    def __init__(self):
        self.routes, self.calls = [], []

    def route(self, fragment, handler):
        self.routes.append((fragment, handler))

    def send(self, adapter, request, **kw):
        self.calls.append(request)
        for frag, handler in self.routes:
            if frag in request.url:
                status, body = handler(request)
                break
        else:
            status, body = 404, {"error": "no fake route for " + request.url}
        resp = requests.Response()
        resp.status_code = status
        resp._content = (json.dumps(body).encode() if isinstance(body, (dict, list))
                         else body.encode() if isinstance(body, str) else body)
        resp.url, resp.request = request.url, request
        resp.headers["Content-Type"] = "application/json"
        return resp


@pytest.fixture
def fake_http(monkeypatch):
    fake = FakeHTTP()
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send",
                        lambda self, request, **kw: fake.send(self, request, **kw))
    fake.route("api.datamuse.com", lambda r: (200, []))  # wildcard expansion falls back offline
    monkeypatch.setattr("time.sleep", lambda s: None)
    return fake
