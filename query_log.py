# ==============================================
# query_log.py
# Records the HTTP requests each engine actually sends, so a search can be
# reported as run ("query as sent", per engine) for systematic reviews.
# ==============================================
"""
Usage::

    with RequestRecorder() as rec:
        with engine_context("OpenAlex"):
            requests.get(...)          # recorded under "OpenAlex"
    rec.rows                           # one dict per request
    rec.summary()                      # one dict per engine

The recorder wraps ``requests.Session.request`` (which ``requests.get`` and
``requests.post`` also go through), so it captures what the engine modules
really send, including their own dialect translation and any retries, without
those modules having to report anything. Secrets and e-mail addresses are
redacted before anything is stored. Outside a ``RequestRecorder`` block,
``engine_context`` is a cheap no-op.
"""
import contextlib
import json
import threading
import urllib.parse
from datetime import datetime

import requests

_local = threading.local()
_lock = threading.Lock()
_active = []  # stack of active recorders (normally 0 or 1)

# Parameter / header names whose values must never be logged.
REDACT = {"api_key", "apikey", "api-key", "key", "x-api-key", "insttoken", "x-els-apikey",
          "x-els-insttoken", "authorization", "mailto", "email", "access_token"}
# Parameters that carry the search expression, in order of preference.
QUERY_KEYS = ("q", "query", "search_query", "search", "term", "filter", "query.bibliographic", "keyword",
              "keywords", "text")


@contextlib.contextmanager
def engine_context(name):
    """Attribute requests made in this thread to engine `name`."""
    prev = getattr(_local, "engine", None)
    _local.engine = name
    try:
        yield
    finally:
        _local.engine = prev


def current_engine():
    return getattr(_local, "engine", None) or "unattributed"


def redact_url(url):
    parts = urllib.parse.urlsplit(url)
    pairs = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    clean = [(k, "<redacted>" if k.lower() in REDACT else v) for k, v in pairs]
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(clean)))


def query_from(url, body=None):
    """The search expression in a request: a known query parameter, else a JSON body field."""
    pairs = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query, keep_blank_values=True))
    for k in QUERY_KEYS:
        if pairs.get(k):
            return pairs[k]
    if isinstance(body, dict):
        for k in QUERY_KEYS:
            if body.get(k):
                return str(body[k])
    return ""


class RequestRecorder:
    def __init__(self):
        self.rows = []
        self._orig = None

    def __enter__(self):
        with _lock:
            if not _active:
                orig = requests.Session.request

                def recording_request(sess, method, url, params=None, data=None, json=None, **kw):
                    try:
                        full = requests.Request(method, url, params=params).prepare().url
                    except Exception:
                        full = str(url)
                    status, err = "", ""
                    try:
                        resp = orig(sess, method, url, params=params, data=data, json=json, **kw)
                        status = resp.status_code
                        return resp
                    except Exception as e:  # recorded, then re-raised unchanged
                        err = type(e).__name__
                        raise
                    finally:
                        body = json if isinstance(json, dict) else None
                        row = {"time": datetime.now().isoformat(timespec="seconds"),
                               "engine": current_engine(), "method": method.upper(),
                               "url": redact_url(full), "query_sent": query_from(full, body),
                               "status": status or err}
                        with _lock:
                            for rec in _active:
                                rec.rows.append(row)

                requests.Session.request = recording_request
                RequestRecorder._orig = orig
            _active.append(self)
        return self

    def __exit__(self, *exc):
        with _lock:
            _active.remove(self)
            if not _active and RequestRecorder._orig is not None:
                requests.Session.request = RequestRecorder._orig
                RequestRecorder._orig = None
        return False

    def summary(self):
        """One row per engine: distinct queries sent, request count, HTTP statuses."""
        out = {}
        for r in self.rows:
            e = out.setdefault(r["engine"], {"engine": r["engine"], "queries": [], "requests": 0,
                                             "statuses": []})
            e["requests"] += 1
            if r["query_sent"] and r["query_sent"] not in e["queries"]:
                e["queries"].append(r["query_sent"])
            if str(r["status"]) not in e["statuses"]:
                e["statuses"].append(str(r["status"]))
        return list(out.values())


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False)
