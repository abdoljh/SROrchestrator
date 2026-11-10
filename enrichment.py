# ==============================================
# enrichment.py
# Fill in missing abstracts / authors / venue / year after deduplication.
#
# Chain (each step only handles papers still missing data):
#   1. OpenAlex         batched DOI lookup (free)     abstract, authors, venue, year
#   2. Europe PMC       batched DOI lookup (free)     abstract (PubMed-indexed papers)
#   3. Semantic Scholar batch endpoint (key optional)  abstract, authors, TLDR
#   4. Elsevier APIs    per DOI, needs SCOPUS_API_KEY  abstract for 10.1016 (ScienceDirect)
#   5. Springer Meta    batched, needs Springer key    abstract for 10.1007 / 10.1186
#   6. Publisher page   DOI landing page meta tags     abstract (any publisher that exposes it)
#   7. Google Scholar snippet, clearly marked
#
# Elsevier withholds its abstracts from Crossref, OpenAlex and Semantic Scholar,
# so for ScienceDirect papers steps 4 and 6 are the only routes.
# ==============================================

import concurrent.futures
import re
import time
from typing import Callable, Dict, Iterable, List, Optional

import requests

from openalex_utils import auth_headers as openalex_auth_headers
from query_utils import clean_abstract, reconstruct_inverted_index

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36 SROrch/2.0")

_UNKNOWN_AUTHORS = {'unknown author', 'unknown authors', 'unknown', 'research team', ''}


# ---------------------------------------------- helpers

def norm_doi(doi) -> str:
    doi = str(doi or '').strip().lower()
    for prefix in ('https://doi.org/', 'http://doi.org/', 'https://dx.doi.org/', 'http://dx.doi.org/', 'doi:'):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi if doi.startswith('10.') else ''


def is_unknown_author(s) -> bool:
    if not s or not isinstance(s, str):
        return True
    c = s.strip().lower()
    if c in _UNKNOWN_AUTHORS:
        return True
    head = re.split(r'\s+(et\s+al|and)\b', c)[0].strip().rstrip(',').strip()
    return head in _UNKNOWN_AUTHORS


def ieee_names(full_names: Iterable[str]) -> str:
    out = []
    for n in full_names:
        n = (n or '').strip()
        if not n:
            continue
        parts = n.split()
        out.append(f"{parts[0][0]}. {' '.join(parts[1:])}" if len(parts) > 1 else n)
    if not out:
        return ''
    if len(out) >= 3:
        return f"{out[0]} et al."
    if len(out) == 2:
        return f"{out[0]} and {out[1]}"
    return out[0]


def _bad_venue(v) -> bool:
    return not v or str(v).strip().lower() in ('unknown venue', 'unknown journal', 'n/a', 'nan', '')


def _bad_year(y) -> bool:
    return not str(y or '').strip().isdigit() or str(y).strip() in ('0',)


def _needs_abstract(p) -> bool:
    return not clean_abstract(p.get('abstract'))


def _needs_meta(p) -> bool:
    return is_unknown_author(p.get('ieee_authors')) or _bad_venue(p.get('venue')) or _bad_year(p.get('year'))


def _get(url, *, params=None, headers=None, timeout=20, retries=3):
    """GET with backoff on 429/5xx."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
            continue
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            time.sleep(float(r.headers.get('Retry-After', 2 ** (attempt + 1))))
            continue
        return r
    return r


def _els_error(r) -> str:
    try:
        st = r.json().get('service-error', {}).get('status', {})
        return st.get('statusText') or st.get('statusCode') or r.text[:120]
    except Exception:
        return r.text[:120]


class AbstractEnricher:
    def __init__(self, api_keys: Dict, config: Optional[Dict] = None, log: Callable = print):
        self.keys = api_keys or {}
        self.config = config or {}
        self.log = log
        self.stats: Dict[str, int] = {}
        # Diagnostics: e.g. {"Publisher page": {"HTTP 403": 12, "no abstract in page": 3}}
        self.diag: Dict[str, Dict[str, int]] = {}

    def _note(self, step, outcome):
        d = self.diag.setdefault(step, {})
        d[outcome] = d.get(outcome, 0) + 1

    # ---------------------------------------------- bookkeeping

    def _index(self, papers, pred) -> Dict[str, List[Dict]]:
        idx: Dict[str, List[Dict]] = {}
        for p in papers:
            d = norm_doi(p.get('doi')) or norm_doi(p.get('url'))
            if d and pred(p):
                if not norm_doi(p.get('doi')):
                    p['doi'] = d
                idx.setdefault(d, []).append(p)
        return idx

    def _set_abstract(self, targets, text, source) -> bool:
        text = clean_abstract(text)
        if len(text) < 40:          # ignore fragments like "No abstract available"
            return False
        hit = False
        for p in targets:
            if _needs_abstract(p) or len(text) > len(clean_abstract(p.get('abstract'))) * 1.5:
                if _needs_abstract(p):
                    hit = True
                p['abstract'] = text
                p['abstract_source'] = source
        if hit:
            self.stats[source] = self.stats.get(source, 0) + 1
        return hit

    def _set_meta(self, targets, authors=None, venue=None, year=None):
        for p in targets:
            if authors and is_unknown_author(p.get('ieee_authors')):
                p['ieee_authors'] = authors
            if venue and _bad_venue(p.get('venue')):
                p['venue'] = venue
            if year and _bad_year(p.get('year')):
                p['year'] = year

    # ---------------------------------------------- 1. OpenAlex

    def openalex(self, papers, batch=50):
        todo = self._index(papers, lambda p: _needs_abstract(p) or _needs_meta(p))
        dois = list(todo)
        for i in range(0, len(dois), batch):
            chunk = dois[i:i + batch]
            r = _get("https://api.openalex.org/works", params={
                'filter': 'doi:' + '|'.join(chunk), 'per-page': batch,
                'select': 'doi,abstract_inverted_index,authorships,primary_location,publication_year',
                'mailto': self.keys.get('email') or ''}, headers=openalex_auth_headers())
            if r.status_code != 200:
                self.log(f"  ⚠️ OpenAlex lookup returned {r.status_code}")
                continue
            for w in r.json().get('results', []):
                targets = todo.get(norm_doi(w.get('doi')), [])
                if not targets:
                    continue
                self._set_abstract(targets, reconstruct_inverted_index(w.get('abstract_inverted_index')), 'OpenAlex')
                names = [(a.get('author') or {}).get('display_name', '') for a in w.get('authorships') or []]
                src = ((w.get('primary_location') or {}).get('source') or {})
                self._set_meta(targets, ieee_names(names), src.get('display_name'), w.get('publication_year'))

    # ---------------------------------------------- 2. Europe PMC

    def europe_pmc(self, papers, batch=15):
        todo = self._index(papers, _needs_abstract)
        dois = list(todo)
        for i in range(0, len(dois), batch):
            chunk = dois[i:i + batch]
            q = ' OR '.join(f'DOI:"{d}"' for d in chunk)
            r = _get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                     params={'query': q, 'format': 'json', 'resultType': 'core', 'pageSize': batch * 2})
            if r.status_code != 200:
                continue
            for e in r.json().get('resultList', {}).get('result', []):
                targets = todo.get(norm_doi(e.get('doi')), [])
                if targets:
                    self._set_abstract(targets, e.get('abstractText'), 'Europe PMC')
                    if e.get('authorString'):
                        names = [a.strip().rstrip('.') for a in e['authorString'].split(',')]
                        # Europe PMC gives "Surname AB"; reorder to "A. Surname"
                        conv = []
                        for n in names:
                            bits = n.rsplit(' ', 1)
                            conv.append(f"{bits[1][0]}. {bits[0]}" if len(bits) == 2 and bits[1] else n)
                        disp = conv[0] + (' et al.' if len(conv) >= 3 else (f' and {conv[1]}' if len(conv) == 2 else ''))
                        self._set_meta(targets, authors=disp)

    # ---------------------------------------------- 3. Semantic Scholar

    def semantic_scholar(self, papers, batch=100):
        todo = self._index(papers, lambda p: _needs_abstract(p) or is_unknown_author(p.get('ieee_authors')))
        dois = list(todo)
        headers = {'x-api-key': self.keys['s2']} if self.keys.get('s2') else {}
        for i in range(0, len(dois), batch):
            chunk = dois[i:i + batch]
            if i and not headers:
                time.sleep(1.2)   # public pool: ~1 request/second
            for attempt in range(4):
                try:
                    r = requests.post("https://api.semanticscholar.org/graph/v1/paper/batch",
                                      params={'fields': 'abstract,tldr,authors,venue,year'},
                                      json={'ids': [f"DOI:{d}" for d in chunk]}, headers=headers, timeout=30)
                except requests.RequestException:
                    r = None
                if r is not None and r.status_code in (401, 403) and headers:
                    self._note('Semantic Scholar', f"API key rejected (HTTP {r.status_code}) - using public pool")
                    headers = {}
                    continue
                if r is not None and r.status_code == 429:
                    time.sleep(2 ** (attempt + 1))   # unauthenticated pool is rate-limited
                    continue
                break
            if r is None or r.status_code != 200:
                self.log(f"  ⚠️ Semantic Scholar batch lookup failed ({getattr(r, 'status_code', 'network')})")
                continue
            for d, item in zip(chunk, r.json()):
                if not item:
                    continue
                targets = todo.get(d, [])
                self._set_abstract(targets, item.get('abstract'), 'Semantic Scholar')
                tldr = (item.get('tldr') or {}).get('text')
                names = [a.get('name', '') for a in item.get('authors') or []]
                self._set_meta(targets, ieee_names(names), item.get('venue'), item.get('year'))
                for p in targets:
                    if tldr and not p.get('tldr'):
                        p['tldr'] = tldr

    # ---------------------------------------------- 3b. Crossref parent book (chapter editors)

    def crossref_books(self, papers, limit=30):
        """Book chapters often carry no names; the parent book lists its editors."""
        from crossref_common import crossref_authors
        todo = self._index(papers, lambda p: is_unknown_author(p.get('ieee_authors')))
        hdr = {'User-Agent': f"SROrch/2.0 (mailto:{self.keys.get('email') or 'researcher@example.com'})"}
        for d in list(todo)[:limit]:
            try:
                r = _get(f"https://api.crossref.org/works/{d}", headers=hdr, timeout=15, retries=2)
                if r.status_code != 200:
                    continue
                item = r.json().get('message', {})
                names, _ = crossref_authors(item)
                if names == 'Unknown Author':
                    for isbn in (item.get('ISBN') or [])[:2]:
                        rb = _get("https://api.crossref.org/works", headers=hdr, timeout=15, retries=2,
                                  params={'filter': f'isbn:{isbn}', 'rows': 5,
                                          'select': 'DOI,type,author,editor'})
                        if rb.status_code != 200:
                            continue
                        for book in rb.json().get('message', {}).get('items', []):
                            if book.get('type') in ('book', 'edited-book', 'monograph', 'reference-book'):
                                names, _ = crossref_authors(book)
                                if names != 'Unknown Author':
                                    break
                        if names != 'Unknown Author':
                            break
                if names != 'Unknown Author':
                    self._set_meta(todo[d], authors=names)
                    self._note('Crossref book editors', 'filled')
                else:
                    self._note('Crossref book editors', 'no author or editor registered')
            except Exception as e:
                self._note('Crossref book editors', f"error: {type(e).__name__}")

    # ---------------------------------------------- 4. Elsevier (ScienceDirect / Scopus)

    def _elsevier_one(self, doi):
        key = self.keys.get('scopus')
        headers = {'X-ELS-APIKey': key, 'Accept': 'application/json'}
        if self.keys.get('scopus_inst'):
            headers['X-ELS-Insttoken'] = self.keys['scopus_inst']
        # ScienceDirect Article Retrieval, abstract-only view
        try:
            r = _get(f"https://api.elsevier.com/content/article/doi/{doi}",
                     params={'view': 'META_ABS'}, headers=headers, timeout=15, retries=2)
            self._note('Elsevier Article API', f"HTTP {r.status_code}")
            if r.status_code in (401, 403):
                self._els_denied = f"HTTP {r.status_code}: {_els_error(r)}"
                return None, None
            if r.status_code == 200:
                core = r.json().get('full-text-retrieval-response', {}).get('coredata', {})
                authors = core.get('dc:creator') or []
                if isinstance(authors, dict):
                    authors = [authors]
                names = [a.get('$', '') for a in authors if isinstance(a, dict)]
                # Elsevier gives "Surname, Given"
                names = [' '.join(reversed([x.strip() for x in n.split(',', 1)])) for n in names]
                return core.get('dc:description'), ieee_names(names)
        except Exception:
            pass
        # Scopus Abstract Retrieval
        try:
            r = _get(f"https://api.elsevier.com/content/abstract/doi/{doi}",
                     params={'view': 'META_ABS'}, headers=headers, timeout=15, retries=2)
            self._note('Elsevier Abstract API', f"HTTP {r.status_code}")
            if r.status_code == 200:
                core = r.json().get('abstracts-retrieval-response', {}).get('coredata', {})
                return core.get('dc:description'), None
        except Exception:
            pass
        return None, None

    def elsevier(self, papers, limit=80):
        if not (self.keys.get('scopus') and len(str(self.keys['scopus']).strip()) > 5):
            n = sum(1 for p in papers if _needs_abstract(p) and norm_doi(p.get('doi')).startswith('10.1016/'))
            if n:
                self._note('Elsevier API', f"skipped: no SCOPUS_API_KEY ({n} Elsevier papers waiting)")
            return
        todo = self._index(papers, lambda p: _needs_abstract(p) or is_unknown_author(p.get('ieee_authors')))
        dois = [d for d in todo if d.startswith('10.1016/')][:limit]
        if not dois:
            return
        # Probe once: if Elsevier denies this key, don't hammer it for every paper
        self._els_denied = None
        first = self._elsevier_one(dois[0])
        if self._els_denied:
            self._note('Elsevier API', f"access denied - {self._els_denied.rstrip('.')}. "
                       f"Off-campus keys need an institutional token (SCOPUS_INST_TOKEN) or an "
                       f"Elsevier-entitled network; {len(dois)} Elsevier abstracts skipped")
            return
        self._set_abstract(todo[dois[0]], first[0], 'Elsevier API')
        self._set_meta(todo[dois[0]], authors=first[1])
        dois = dois[1:]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            for d, (abstract, authors) in zip(dois, ex.map(self._elsevier_one, dois)):
                self._set_abstract(todo[d], abstract, 'Elsevier API')
                self._set_meta(todo[d], authors=authors)

    # ---------------------------------------------- 5. Springer Nature

    def springer(self, papers, batch=10):
        key = self.keys.get('springer')
        if not (key and len(str(key).strip()) > 5):
            return
        todo = self._index(papers, _needs_abstract)
        dois = [d for d in todo if d.startswith(('10.1007/', '10.1186/', '10.1038/', '10.1057/'))]
        for i in range(0, len(dois), batch):
            chunk = dois[i:i + batch]
            r = _get("https://api.springernature.com/meta/v2/json",
                     params={'q': ' OR '.join(f'doi:{d}' for d in chunk), 'p': batch, 'api_key': key})
            if r.status_code != 200:
                continue
            for rec in r.json().get('records', []):
                abstract = rec.get('abstract')
                if isinstance(abstract, dict):
                    abstract = ' '.join(str(v) for v in abstract.values())
                self._set_abstract(todo.get(norm_doi(rec.get('doi')), []), abstract, 'Springer Nature')

    # ---------------------------------------------- 6. Publisher landing page

    _META_NAMES = ('citation_abstract', 'dc.description', 'dcterms.abstract', 'prism.teaser',
                   'description', 'og:description', 'twitter:description')

    def _landing_one(self, doi) -> Optional[str]:
        sess = requests.Session()
        sess.headers.update({'User-Agent': UA, 'Accept': 'text/html,application/xhtml+xml',
                             'Accept-Language': 'en'})
        try:
            r = sess.get(f"https://doi.org/{doi}", timeout=12, allow_redirects=True)
            # Elsevier's linkinghub uses a JS redirect page
            m = re.search(r"linkinghub\.elsevier\.com/retrieve/pii/(\w+)", r.url)
            if m:
                r = sess.get(f"https://www.sciencedirect.com/science/article/abs/pii/{m.group(1)}", timeout=12)
            host = re.sub(r'^https?://([^/]+).*$', r'\1', r.url)
            if r.status_code != 200 or 'html' not in r.headers.get('Content-Type', ''):
                self._note('Publisher page', f"{host}: HTTP {r.status_code}")
                return None
            text = extract_abstract_from_html(r.text)
            self._note('Publisher page', f"{host}: {'abstract found' if text else 'no abstract in page'}")
            return text
        except Exception as e:
            self._note('Publisher page', f"error: {type(e).__name__}")
            return None

    def landing_pages(self, papers, limit=40):
        todo = self._index(papers, _needs_abstract)
        dois = list(todo)[:limit]
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            for d, text in zip(dois, ex.map(self._landing_one, dois)):
                if text:
                    self._set_abstract(todo[d], text, 'Publisher page')

    # ---------------------------------------------- driver

    def run(self, papers):
        def count():
            return sum(1 for p in papers if not _needs_abstract(p))
        before, unk_before = count(), sum(1 for p in papers if is_unknown_author(p.get('ieee_authors')))
        self.log(f"\n[Enrich] {before}/{len(papers)} papers have abstracts from the search engines")

        steps = [('OpenAlex', self.openalex), ('Europe PMC', self.europe_pmc),
                 ('Semantic Scholar', self.semantic_scholar), ('Crossref books', self.crossref_books),
                 ('Elsevier', self.elsevier),
                 ('Springer', self.springer)]
        if self.config.get('scrape_landing_pages', True):
            steps.append(('Publisher pages',
                          lambda ps: self.landing_pages(ps, self.config.get('landing_page_limit', 40))))
        for name, step in steps:
            if not any(_needs_abstract(p) or _needs_meta(p) for p in papers):
                break
            try:
                step(papers)
            except Exception as e:
                self.log(f"  ⚠️ {name} enrichment failed: {e}")

        for p in papers:
            if _needs_abstract(p) and p.get('snippet'):
                p['abstract'] = f"[Snippet] {p['snippet']}"
                p['abstract_source'] = 'Google Scholar (snippet)'

        after = count()
        unk_after = sum(1 for p in papers if is_unknown_author(p.get('ieee_authors')))
        detail = ', '.join(f"{k}: {v}" for k, v in self.stats.items()) or 'none'
        self.log(f"[Enrich] {after}/{len(papers)} papers have abstracts (+{after - before}; {detail})")
        self.log(f"[Enrich] Unknown authors: {unk_before} -> {unk_after}")
        for step, outcomes in self.diag.items():
            self.log(f"  · {step}: " + ', '.join(f"{k} ×{v}" for k, v in sorted(outcomes.items(), key=lambda x: -x[1])))
        self.summary = {
            'with_abstract_before': before, 'with_abstract_after': after, 'total': len(papers),
            'unknown_authors_before': unk_before, 'unknown_authors_after': unk_after,
            'filled_by': dict(self.stats), 'diagnostics': self.diag,
        }
        return papers


# ---------------------------------------------- HTML abstract extraction

def extract_abstract_from_html(html_text: str) -> Optional[str]:
    """Pull an abstract from a publisher landing page (meta tags, then abstract blocks)."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None
    soup = BeautifulSoup(html_text, 'html.parser')

    best = ''
    for meta in soup.find_all('meta'):
        name = (meta.get('name') or meta.get('property') or '').strip().lower()
        if name in AbstractEnricher._META_NAMES:
            text = clean_abstract(meta.get('content'))
            # og/description are often truncated teasers: prefer the longest
            if len(text) > len(best):
                best = text

    # Abstract sections in the page body (ScienceDirect: div.abstract.author)
    for el in soup.select('div.abstract, section.abstract, div#abstract, section#abstract, '
                          'div[class*="abstract"], section[class*="Abstract"], div.hlFld-Abstract'):
        cls = ' '.join(el.get('class', [])).lower()
        if 'graphical' in cls or 'highlight' in cls:
            continue
        text = clean_abstract(el.get_text(' ', strip=True))
        if len(text) > len(best):
            best = text
            break

    # Reject site boilerplate descriptions
    if len(best) < 150 or re.search(r'(cookies|javascript|sign in|subscribe)', best[:200], re.I):
        return None
    return best[:5000]
