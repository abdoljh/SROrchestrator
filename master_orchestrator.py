
# ==============================================
# master_orchestrator.py
# SROrch (Scholarly Research Orchestrator)
# ==============================================

import os
import csv
import re
import shutil
import json
import concurrent.futures
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Optional

# Import your tools
import s2_utils
import arxiv_utils
import pubmed_utils
import scholar_utils
import doi_utils
import openalex_utils
import core_utils
import scopus_utils

# Additional search engines
import europe_pmc_utils
import plos_utils
import ssrn_utils
import deepdyve_utils
import springer_utils
import wiley_utils
import tf_utils
import acm_utils
import dblp_utils
import sage_utils

# Research Gap Analysis
import gap_utils
from gap_utils import analyze_research_gaps

# Query / abstract helpers
from query_utils import safe_query_slug, clean_abstract, NO_ABSTRACT, wildcard_terms, expand_wildcards
from enrichment import AbstractEnricher, is_unknown_author

load_dotenv()

class ResearchOrchestrator:
    def __init__(self, config: Optional[Dict] = None):
        self.api_keys = {
            # Current Premium Engines
            's2': os.getenv('S2_API_KEY'),
            'serp': os.getenv('SERP_API_KEY'),
            'core': os.getenv('CORE_API_KEY'),
            'scopus': os.getenv('SCOPUS_API_KEY'),
            'scopus_inst': os.getenv('SCOPUS_INST_TOKEN'),
            'springer': os.getenv('META_SPRINGER_API_KEY'),
            'email': os.getenv('USER_EMAIL', 'researcher@example.com'),
        }
        self.output_dir = ""

        self.config = config or {
            'abstract_limit': 5,
            'high_consensus_threshold': 4,
            'citation_weight': 1.0,
            'source_weight': 100,
            'enable_alerts': True,
            'enable_visualization': True,
            'export_formats': ['csv', 'json', 'bibtex'],
            'recency_boost': True,
            'recency_years': 3,
            'recency_multiplier': 1.5,
            'enrich_abstracts': True,
            'scrape_landing_pages': True,
            'landing_page_limit': 40,
        }
        # Older/partial configs (e.g. from the Streamlit UI) get the default
        self.config.setdefault('enrich_abstracts', True)
        self.config.setdefault('scrape_landing_pages', True)
        self.config.setdefault('landing_page_limit', 40)

        self.session_metadata = {
            'start_time': None,
            'end_time': None,
            'query': None,
            'total_api_calls': 0,
            'failed_engines': [],
            'successful_engines': []
        }

    def create_output_directory(self, query):
        # The query can be an arbitrarily long boolean expression; a raw slug of
        # it can exceed the 255-byte filename limit ([Errno 36] File name too long).
        clean_q = safe_query_slug(query)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f"SROrch_{clean_q}_{timestamp}"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        print(f"📂 Created session directory: {self.output_dir}")
        return self.output_dir

    def normalize_venue(self, venue):
        """Normalize venue field to always return a string."""
        if venue is None:
            return ''
        if isinstance(venue, list):
            return ' '.join(str(v) for v in venue if v)
        return str(venue)

    _UNKNOWN_AUTHOR_PATTERNS = {'unknown author', 'unknown authors', 'unknown', 'research team', ''}

    def _is_unknown_author(self, authors_str):
        """Placeholder author values ("Unknown Author", "Unknown et al.", "Unknown and X")."""
        return is_unknown_author(authors_str)

    def deduplicate_and_score(self, all_papers):
        unique_papers = {}
        current_year = datetime.now().year

        for paper in all_papers:
            if 'venue' in paper:
                paper['venue'] = self.normalize_venue(paper['venue'])

            doi = str(paper.get('doi') or 'N/A').lower().strip()
            title = paper.get('title', '').lower().strip()
            clean_title = re.sub(r'[^a-zA-Z0-9]', '', title)
            key = doi if (doi != 'n/a' and len(doi) > 5) else clean_title

            try:
                cites_raw = paper.get('citations', 0)
                cites = int(cites_raw) if str(cites_raw).isdigit() else 0
            except:
                cites = 0

            if key in unique_papers:
                unique_papers[key]['source_count'] += 1
                if paper.get('found_in') and paper['found_in'] not in unique_papers[key].get('found_in', ''):
                    unique_papers[key]['found_in'] = '; '.join(
                        x for x in (unique_papers[key].get('found_in'), paper['found_in']) if x)
                if cites > unique_papers[key].get('citations_int', 0):
                    unique_papers[key]['citations_int'] = cites
                    unique_papers[key]['citations'] = cites

                # Merge authors: prefer real names over "Unknown Author"
                existing_authors = unique_papers[key].get('ieee_authors', '')
                new_authors = paper.get('ieee_authors', '')
                if self._is_unknown_author(existing_authors) and not self._is_unknown_author(new_authors):
                    unique_papers[key]['ieee_authors'] = new_authors

                # Merge venue / year when the first copy lacked them
                if str(unique_papers[key].get('venue', '')).strip().lower() in ('', 'unknown venue', 'unknown journal', 'n/a') \
                        and paper.get('venue') and 'unknown' not in str(paper['venue']).lower():
                    unique_papers[key]['venue'] = paper['venue']
                if not str(unique_papers[key].get('year', '')).isdigit() and str(paper.get('year', '')).isdigit():
                    unique_papers[key]['year'] = paper['year']

                # Merge abstracts: keep the most complete one
                existing = unique_papers[key]
                new_abs = clean_abstract(paper.get('abstract'))
                if new_abs and len(new_abs) > len(clean_abstract(existing.get('abstract'))):
                    existing['abstract'] = new_abs
                    existing['abstract_source'] = paper.get('abstract_source', '')
                for fld in ('snippet', 'tldr', 'keywords'):
                    if paper.get(fld) and not existing.get(fld):
                        existing[fld] = paper[fld]
                if (not existing.get('doi') or str(existing.get('doi')).upper() == 'N/A') and \
                        paper.get('doi') and str(paper['doi']).upper() != 'N/A':
                    existing['doi'] = paper['doi']

                # Merge impact_factor: keep the best non-null value
                existing_if = unique_papers[key].get('impact_factor')
                new_if = paper.get('impact_factor')
                if new_if is not None and (existing_if is None or new_if > existing_if):
                    unique_papers[key]['impact_factor'] = new_if

                # Merge h_index: keep the best non-null value
                existing_h = unique_papers[key].get('h_index')
                new_h = paper.get('h_index')
                if new_h is not None and (existing_h is None or new_h > existing_h):
                    unique_papers[key]['h_index'] = new_h

                # Merge influential_citations: keep the best non-null value
                existing_ic = unique_papers[key].get('influential_citations')
                new_ic = paper.get('influential_citations')
                if new_ic is not None and (existing_ic is None or new_ic > existing_ic):
                    unique_papers[key]['influential_citations'] = new_ic

                if (self.config['enable_alerts'] and
                    unique_papers[key]['source_count'] == self.config['high_consensus_threshold']):
                    print(f"🚨 ALERT: High-Consensus Discovery! Found in {self.config['high_consensus_threshold']}+ engines: \"{paper['title'][:60]}...\"")
            else:
                paper['source_count'] = 1
                paper['citations_int'] = cites
                unique_papers[key] = paper

        for p in unique_papers.values():
            base_score = (p['source_count'] * self.config['source_weight']) + \
                        (p['citations_int'] * self.config['citation_weight'])

            # Impact factor boost (capped at 50 points)
            paper_if = p.get('impact_factor')
            if paper_if is not None:
                try:
                    if_boost = min(float(paper_if) * 2, 50)
                    base_score += if_boost
                except (ValueError, TypeError):
                    pass

            # H-index boost (capped at 25 points)
            paper_h = p.get('h_index')
            if paper_h is not None:
                try:
                    h_boost = min(int(paper_h) * 0.5, 25)
                    base_score += h_boost
                except (ValueError, TypeError):
                    pass

            if self.config['recency_boost']:
                try:
                    year_val = str(p.get('year', ''))
                    if year_val.isdigit() and len(year_val) == 4:
                        paper_year = int(year_val)
                        if 1900 <= paper_year <= 2100:
                            if paper_year >= (current_year - self.config['recency_years']):
                                base_score *= self.config['recency_multiplier']
                                p['recency_boosted'] = True
                except:
                    pass

            p['relevance_score'] = int(base_score)

        return sorted(unique_papers.values(), key=lambda x: x['relevance_score'], reverse=True)

    def enrich_missing_abstracts(self, papers):
        """
        Fill in missing abstracts (and Unknown authors / venue / year) via the
        enrichment chain in enrichment.py: OpenAlex -> Europe PMC -> Semantic
        Scholar -> Elsevier API (SCOPUS_API_KEY) -> Springer -> publisher page.
        """
        log_lines = []

        def log(msg):
            print(msg)
            log_lines.append(msg.strip('\n'))

        enricher = AbstractEnricher(self.api_keys, self.config, log=log)
        enricher.run(papers)
        self.session_metadata['enrichment'] = getattr(enricher, 'summary', {})
        self.session_metadata['enrichment_log'] = log_lines
        return papers

    def fetch_abstracts_for_top_papers(self, top_papers, limit=None):
        limit = limit or self.config['abstract_limit']
        print(f"\n[AI] Performing 'Deep Look' for Top {limit} papers...")
        abstract_summaries = []

        import requests
        headers = {"x-api-key": self.api_keys['s2']} if self.api_keys['s2'] else {}

        for i, paper in enumerate(top_papers[:limit]):
            # Keep the abstract the search engines (or enrichment) already found
            abstract = clean_abstract(paper.get('abstract')) or NO_ABSTRACT
            doi = str(paper.get('doi', '')).strip()
            title = paper.get('title')

            if abstract == NO_ABSTRACT and doi and doi.lower() != 'n/a':
                try:
                    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=abstract,authors,url,title,tldr,s2FieldsOfStudy,publicationTypes"
                    r = requests.get(url, headers=headers, timeout=12)
                    if r.status_code == 200:
                        data = r.json()
                        if clean_abstract(data.get('abstract')):
                            abstract = clean_abstract(data.get('abstract'))
                            paper['abstract_source'] = 'Semantic Scholar'
                        if data.get('url'): paper['url'] = data.get('url')
                        if data.get('tldr'): paper['tldr'] = data.get('tldr', {}).get('text', '')
                        if data.get('fieldsOfStudy'): paper['keywords'] = ', '.join(data.get('fieldsOfStudy', []))
                        # Enrich authors if current value is unknown
                        if self._is_unknown_author(paper.get('ieee_authors', '')) and data.get('authors'):
                            names = [a.get('name', '') for a in data['authors'] if a.get('name')]
                            if names:
                                paper['ieee_authors'] = ', '.join(names[:3]) + (' et al.' if len(names) > 3 else '')
                except: pass

            if abstract == NO_ABSTRACT and title:
                try:
                    r = requests.get(
                        "https://api.semanticscholar.org/graph/v1/paper/search",
                        params={'query': title[:300], 'limit': 1,
                                'fields': 'abstract,authors,url,externalIds,tldr,fieldsOfStudy'},
                        headers=headers, timeout=12)
                    if r.status_code == 200:
                        results_data = r.json().get('data', [])
                        if results_data:
                            hit = results_data[0]
                            # Only trust the hit if it is really the same paper
                            same = re.sub(r'[^a-z0-9]', '', (hit.get('title') or '').lower()) == \
                                   re.sub(r'[^a-z0-9]', '', title.lower())
                            if same and clean_abstract(hit.get('abstract')):
                                abstract = clean_abstract(hit.get('abstract'))
                                paper['abstract_source'] = 'Semantic Scholar'
                            if results_data[0].get('url'): paper['url'] = results_data[0].get('url')
                            hit_doi = (hit.get('externalIds') or {}).get('DOI')
                            if same and hit_doi and str(paper.get('doi', 'N/A')).upper() == 'N/A':
                                paper['doi'] = hit_doi
                            if results_data[0].get('tldr'): paper['tldr'] = results_data[0].get('tldr', {}).get('text', '')
                            if results_data[0].get('fieldsOfStudy'): paper['keywords'] = ', '.join(results_data[0].get('fieldsOfStudy', []))
                            # Enrich authors if current value is unknown
                            if self._is_unknown_author(paper.get('ieee_authors', '')) and results_data[0].get('authors'):
                                names = [a.get('name', '') for a in results_data[0]['authors'] if a.get('name')]
                                if names:
                                    paper['ieee_authors'] = ', '.join(names[:3]) + (' et al.' if len(names) > 3 else '')
                except: pass

            if abstract == NO_ABSTRACT and paper.get('snippet'):
                abstract = f"[Snippet] {paper['snippet']}"
            paper['abstract'] = '' if abstract == NO_ABSTRACT else abstract

            summary_block = (
                f"RANK [{i+1}]\n"
                f"TITLE:    {paper['title']}\n"
                f"AUTHORS:  {paper['ieee_authors']}\n"
                f"YEAR:     {paper.get('year', 'N/A')}\n"
                f"VENUE:    {paper.get('venue', 'N/A')}\n"
                f"LINK:     {paper.get('url', 'No link found')}\n"
                f"DOI:      {paper.get('doi', 'N/A')}\n"
            )

            if paper.get('tldr'):
                summary_block += f"TLDR:     {paper['tldr']}\n"
            if paper.get('keywords'):
                summary_block += f"KEYWORDS: {paper['keywords']}\n"

            summary_block += f"ABSTRACT: {paper['abstract'] or NO_ABSTRACT}\n{'-'*60}\n"
            abstract_summaries.append(summary_block)

        return abstract_summaries

    def generate_research_statistics(self, results):
        if not self.config['enable_visualization']:
            print("\n📊 Visualization disabled in config.")
            return

        print("\n📊 Generating Research Statistics...")

        try:
            import matplotlib.pyplot as plt
            from collections import Counter

            total_papers = len(results)
            valid_cites = [int(p.get('citations_int', 0)) for p in results]
            avg_citations = sum(valid_cites) / total_papers if total_papers > 0 else 0
            max_citations = max(valid_cites) if valid_cites else 0
            median_citations = sorted(valid_cites)[len(valid_cites)//2] if valid_cites else 0
            high_consensus_papers = sum(1 for p in results if p.get('source_count', 0) >= self.config['high_consensus_threshold'])

            recent_papers = 0
            for p in results:
                try:
                    year_val = str(p.get('year', ''))
                    if year_val.isdigit() and len(year_val) == 4:
                        year_int = int(year_val)
                        if year_int >= (datetime.now().year - self.config['recency_years']):
                            recent_papers += 1
                except:
                    pass

            source_counts = Counter(p.get('source_count', 1) for p in results)

            years = []
            for p in results:
                year_val = str(p.get('year', ''))
                if year_val.isdigit() and len(year_val) == 4:
                    try:
                        year_int = int(year_val)
                        if 1900 <= year_int <= 2100:
                            years.append(year_val)
                    except:
                        pass
            year_counts = Counter(years)
            sorted_years = sorted(year_counts.keys())
            counts = [year_counts[y] for y in sorted_years]

            if sorted_years:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))

                axes[0].bar(sorted_years, counts, color='#2acaea', edgecolor='black')
                axes[0].set_xlabel('Publication Year', fontsize=12)
                axes[0].set_ylabel('Number of Papers', fontsize=12)
                axes[0].set_title(f'Publication Trend: {total_papers} Papers', fontsize=14, fontweight='bold')
                axes[0].tick_params(axis='x', rotation=45)
                axes[0].grid(axis='y', linestyle='--', alpha=0.7)

                consensus_data = sorted(source_counts.items())
                consensus_labels = [f"{k} source{'s' if k > 1 else ''}" for k, v in consensus_data]
                consensus_values = [v for k, v in consensus_data]

                axes[1].barh(consensus_labels, consensus_values, color='#FF6B6B', edgecolor='black')
                axes[1].set_xlabel('Number of Papers', fontsize=12)
                axes[1].set_ylabel('Source Consensus', fontsize=12)
                axes[1].set_title('Cross-Database Coverage', fontsize=14, fontweight='bold')
                axes[1].grid(axis='x', linestyle='--', alpha=0.7)

                plt.tight_layout()
                stats_plot_path = os.path.join(self.output_dir, "research_analytics.png")
                plt.savefig(stats_plot_path, dpi=300)
                print(f"📈 Analytics dashboard saved as: research_analytics.png")

                try:
                    plt.show()
                except:
                    pass

                plt.close()

            print("\n" + "="*50)
            print(f"       RESEARCH METRICS SUMMARY")
            print("="*50)
            print(f"Total Unique Papers Discovered: {total_papers}")
            print(f"High-Consensus Papers (≥{self.config['high_consensus_threshold']} sources): {high_consensus_papers}")
            print(f"Recent Papers (last {self.config['recency_years']} years): {recent_papers}")
            print("-" * 50)
            print(f"CITATION METRICS:")
            print(f"  - Average:  {avg_citations:.2f}")
            print(f"  - Median:   {median_citations}")
            print(f"  - Maximum:  {max_citations}")
            print("-" * 50)
            print(f"CONSENSUS DISTRIBUTION:")
            for count, freq in sorted(source_counts.items(), reverse=True):
                percentage = (freq / total_papers) * 100
                print(f"  - Found in {count} engine(s): {freq} papers ({percentage:.1f}%)")
            print("="*50 + "\n")

        except Exception as e:
            print(f"⚠️ Statistics generation failed: {e}")

    def run_search(self, query, limit_per_engine=15):
        self.session_metadata['start_time'] = datetime.now()
        self.session_metadata['query'] = query

        self.create_output_directory(query)
        print(f"\n[Master] Orchestrating search for: '{query}'...")
        import crossref_common
        crossref_common.INCLUDE_REFERENCE_ENTRIES = bool(self.config.get('include_reference_entries', False))

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            tasks = {}
            
            def is_valid_key(key):
                return key and isinstance(key, str) and len(key.strip()) > 5

            disabled = set(self.config.get('disabled_engines') or [])

            # Many engines silently ignore trailing wildcards (co-pyroly* matches
            # nothing in OpenAlex/Scholar/...). Pre-compute an expanded query and
            # retry any engine that returns nothing with it.
            expanded_query = expand_wildcards(query) if wildcard_terms(query) else None
            if expanded_query:
                print(f"  ✱ Wildcards expanded for retries: {expanded_query[:160]}...")
                self.session_metadata['expanded_query'] = expanded_query
            retried = self.session_metadata.setdefault('wildcard_retries', [])

            def run_engine(name, fn, args, kwargs):
                first_error = None
                try:
                    res = fn(*args, **kwargs)
                except Exception as e:
                    res, first_error = None, e
                if res or not expanded_query:
                    if first_error:
                        raise first_error
                    return res
                args2 = tuple(expanded_query if a == query else a for a in args)
                res = fn(*args2, **kwargs)
                retried.append(f"{name}: {len(res or [])} papers")
                return res
            k = self.api_keys
            L = limit_per_engine
            # (name, required API key or None, callable, args, kwargs)
            engines = [
                ("Semantic Scholar", 's2', s2_utils.fetch_and_process_papers, (k.get('s2'), query), {'csv_limit': L}),
                ("Google Scholar", 'serp', scholar_utils.fetch_and_process_scholar, (k.get('serp'), query), {'max_limit': L}),
                ("CORE", 'core', core_utils.fetch_and_process_core, (k.get('core'), query), {'max_limit': L}),
                ("SCOPUS", 'scopus', scopus_utils.fetch_and_process_scopus, (k.get('scopus'), query), {'max_limit': L, 'save_csv': False}),
                ("Springer Nature", 'springer', springer_utils.fetch_and_process_springer, (query,), {'max_limit': L}),
                ("arXiv", None, arxiv_utils.fetch_and_process_arxiv, (query,), {'max_limit': L}),
                ("PubMed", None, pubmed_utils.fetch_and_process_pubmed, (query,), {'max_limit': L}),
                ("Crossref/DOI", None, doi_utils.fetch_and_process_doi, (query,), {'max_limit': L}),
                ("OpenAlex", None, openalex_utils.fetch_and_process_openalex, (query,), {'max_limit': L}),
                ("Europe PMC", None, europe_pmc_utils.fetch_and_process_europe_pmc, (query,), {'max_limit': L}),
                ("PLOS", None, plos_utils.fetch_and_process_plos, (query,), {'max_limit': L}),
                ("SSRN", None, ssrn_utils.fetch_and_process_ssrn, (query,), {'max_limit': L}),
                ("DeepDyve", None, deepdyve_utils.fetch_and_process_deepdyve, (query,), {'max_limit': L}),
                ("Wiley", None, wiley_utils.fetch_and_process_wiley, (query,), {'max_limit': L}),
                ("Taylor & Francis", None, tf_utils.fetch_and_process_tf, (query,), {'max_limit': L}),
                ("ACM Digital Library", None, acm_utils.fetch_and_process_acm, (query,), {'max_limit': L}),
                ("DBLP", None, dblp_utils.fetch_and_process_dblp, (query,), {'max_limit': L}),
                ("SAGE Journals", None, sage_utils.fetch_and_process_sage, (query,), {'max_limit': L}),
            ]
            for name, key_name, fn, args, kwargs in engines:
                if name in disabled:
                    print(f"  ⏸  {name} disabled by user")
                    self.session_metadata.setdefault('disabled_engines', []).append(name)
                    continue
                if key_name and not is_valid_key(k.get(key_name)):
                    print(f"  ✗ {name} skipped (no valid API key)")
                    self.session_metadata['failed_engines'].append(f"{name} (no API key)")
                    continue
                tasks[executor.submit(run_engine, name, fn, args, kwargs)] = name
                print(f"  ✓ {name} enabled")

            combined_results = []
            for future in concurrent.futures.as_completed(tasks):
                engine_name = tasks[future]
                try:
                    data = future.result()
                    if data:
                        for p in data:
                            p['abstract'] = clean_abstract(p.get('abstract'))
                            if p['abstract']:
                                p.setdefault('abstract_source', engine_name)
                            p['found_in'] = engine_name
                        combined_results.extend(data)
                        self.session_metadata['successful_engines'].append(engine_name)
                        self.session_metadata['total_api_calls'] += 1
                        print(f"  ✓ {engine_name} completed successfully ({len(data)} papers)")
                    else:
                        print(f"  ⚠️  {engine_name} returned no results")
                        if engine_name not in self.session_metadata['failed_engines']:
                            self.session_metadata['failed_engines'].append(f"{engine_name} (no results)")
                except Exception as e:
                    print(f"  ⚠️  {engine_name} failed: {e}")
                    if engine_name not in self.session_metadata['failed_engines']:
                        self.session_metadata['failed_engines'].append(f"{engine_name} (error: {str(e)[:160]})")

        final_list = self.deduplicate_and_score(combined_results)

        if self.config.get('enrich_abstracts', True):
            try:
                self.enrich_missing_abstracts(final_list)
            except Exception as e:
                print(f"⚠️ Abstract enrichment failed: {e}")

        for file in os.listdir('.'):
            if file.endswith(".csv") and not file.startswith("MASTER_"):
                try: shutil.move(file, os.path.join(self.output_dir, file))
                except: pass

        self.session_metadata['end_time'] = datetime.now()
        
        print(f"\n📊 Search Summary:")
        print(f"  Successful: {len(self.session_metadata['successful_engines'])} engines")
        print(f"  Failed: {len(self.session_metadata['failed_engines'])} engines")
        print(f"  Total unique papers: {len(final_list)}")

        return final_list

    def export_bibtex(self, results, filename="references.bib"):
        bibtex_path = os.path.join(self.output_dir, filename)

        with open(bibtex_path, 'w', encoding='utf-8') as f:
            for i, paper in enumerate(results, 1):
                first_author = paper.get('ieee_authors', 'Unknown').split(',')[0].strip().replace(' ', '')
                year = paper.get('year', 'n.d.')
                cite_key = f"{first_author}{year}_{i}"

                venue_raw = paper.get('venue', '')
                if isinstance(venue_raw, list):
                    venue = ' '.join(str(v) for v in venue_raw if v).lower()
                else:
                    venue = str(venue_raw).lower() if venue_raw else ''
                
                if 'conference' in venue or 'proceedings' in venue:
                    entry_type = 'inproceedings'
                elif 'arxiv' in venue:
                    entry_type = 'misc'
                else:
                    entry_type = 'article'

                f.write(f"@{entry_type}{{{cite_key},\n")
                f.write(f"  title = {{{paper.get('title', 'No title')}}},\n")
                f.write(f"  author = {{{paper.get('ieee_authors', 'Unknown')}}},\n")
                f.write(f"  year = {{{year}}},\n")

                if paper.get('venue'):
                    journal_key = 'booktitle' if entry_type == 'inproceedings' else 'journal'
                    venue_value = paper['venue']
                    if isinstance(venue_value, list):
                        venue_str = ' '.join(str(v) for v in venue_value if v)
                    else:
                        venue_str = str(venue_value) if venue_value else ''
                    if venue_str:
                        f.write(f"  {journal_key} = {{{venue_str}}},\n")

                if paper.get('doi') and paper['doi'] != 'N/A':
                    f.write(f"  doi = {{{paper['doi']}}},\n")
                if paper.get('url'):
                    f.write(f"  url = {{{paper['url']}}},\n")
                if paper.get('abstract'):
                    abstract = paper['abstract'].replace('{', '\\{').replace('}', '\\}')
                    f.write(f"  abstract = {{{abstract}}},\n")

                f.write(f"  note = {{Citations: {paper.get('citations', 0)}, Sources: {paper.get('source_count', 1)}}}\n")
                f.write("}\n\n")

        print(f"📚 BibTeX export saved as: {filename}")

    def export_json(self, results, filename="research_data.json"):
        json_path = os.path.join(self.output_dir, filename)

        export_data = {
            'metadata': {
                'query': self.session_metadata['query'],
                'search_date': self.session_metadata['start_time'].isoformat() if self.session_metadata['start_time'] else None,
                'total_papers': len(results),
                'successful_engines': self.session_metadata['successful_engines'],
                'failed_engines': self.session_metadata['failed_engines'],
                'execution_time_seconds': (self.session_metadata['end_time'] - self.session_metadata['start_time']).total_seconds() if self.session_metadata['end_time'] and self.session_metadata['start_time'] else None
            },
            'papers': results
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"📄 JSON export saved as: {filename}")

    def generate_session_report(self):
        report_path = os.path.join(self.output_dir, "SESSION_REPORT.txt")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("          SROrch SESSION REPORT\n")
            f.write("="*60 + "\n\n")

            f.write(f"Query: {self.session_metadata['query']}\n")
            f.write(f"Start Time: {self.session_metadata['start_time']}\n")
            f.write(f"End Time: {self.session_metadata['end_time']}\n")

            if self.session_metadata['start_time'] and self.session_metadata['end_time']:
                duration = self.session_metadata['end_time'] - self.session_metadata['start_time']
                f.write(f"Duration: {duration.total_seconds():.2f} seconds\n")

            f.write(f"\nTotal API Calls: {self.session_metadata['total_api_calls']}\n")
            f.write(f"Successful Engines: {', '.join(self.session_metadata['successful_engines'])}\n")

            if self.session_metadata['failed_engines']:
                f.write(f"Failed Engines: {', '.join(self.session_metadata['failed_engines'])}\n")

            if self.session_metadata.get('wildcard_retries'):
                f.write(f"Retried with expanded wildcards: {', '.join(self.session_metadata['wildcard_retries'])}\n")
                f.write(f"  Expanded query: {self.session_metadata.get('expanded_query', '')}\n")

            if self.session_metadata.get('disabled_engines'):
                f.write(f"Disabled by user: {', '.join(self.session_metadata['disabled_engines'])}\n")

            if self.session_metadata.get('enrichment_log'):
                f.write("\n" + "="*60 + "\n")
                f.write("ABSTRACT / METADATA ENRICHMENT:\n")
                f.write("="*60 + "\n")
                for line in self.session_metadata['enrichment_log']:
                    f.write(line + "\n")

            f.write("\n" + "="*60 + "\n")
            f.write("CONFIGURATION USED:\n")
            f.write("="*60 + "\n")
            for key, value in self.config.items():
                f.write(f"{key}: {value}\n")

            f.write("\n" + "="*60 + "\n")

        print(f"📋 Session report saved as: SESSION_REPORT.txt")

    def save_master_csv(self, results, query):
        # 1. Trigger Deep Look
        top_abstract_blocks = self.fetch_abstracts_for_top_papers(results)

        # ✅ ENHANCED: Identify Research Gaps with improved detection
        print("\n🔍 Scanning for Research Gaps and Future Directions...")
        gap_data = analyze_research_gaps(results, query)  # Pass query for domain-specific patterns
        
        # Save enhanced Gap Report
        gap_report_path = os.path.join(self.output_dir, "RESEARCH_GAPS.txt")
        with open(gap_report_path, 'w', encoding='utf-8') as f:
            f.write(f"=== IDENTIFIED RESEARCH GAPS & FUTURE DIRECTIONS ===\n")
            f.write(f"QUERY: {query}\n")
            f.write(f"DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"SUMMARY:\n")
            f.write(f"  Total gaps found: {gap_data['total_gaps_found']}\n")
            f.write(f"  Papers analyzed: {gap_data['papers_analyzed']}\n")
            f.write(f"  Pattern types used: {', '.join(gap_data['pattern_types_used'])}\n")
            f.write(f"  Top keywords: {', '.join([k for k, count in gap_data['top_keywords'][:5]])}\n")
            f.write("\n" + "="*60 + "\n\n")
            
            # ✅ NEW: Gap Categories
            if gap_data.get('gap_categories'):
                f.write("GAP CATEGORIES:\n")
                f.write("="*60 + "\n")
                for category, count in sorted(gap_data['gap_categories'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  • {category}: {count} gaps\n")
                f.write("\n" + "="*60 + "\n\n")
            
            if gap_data['gap_list']:
                f.write("DETAILED GAP STATEMENTS (Sorted by Citation Impact):\n")
                f.write("="*60 + "\n\n")
                for i, gap in enumerate(gap_data['gap_list'], 1):
                    f.write(f"[{i}] SOURCE: {gap['title']} ({gap['year']})\n")
                    f.write(f"    CATEGORY: {gap['category']}\n")
                    f.write(f"    CITATIONS: {gap['citations']}\n")
                    f.write(f"    GAP SIGNAL: \"{gap['gap_statement']}\"\n")
                    f.write("-" * 60 + "\n\n")
            else:
                f.write("No explicit research gap statements found in the analyzed papers.\n")
                f.write("This may indicate:\n")
                f.write("  - Papers don't have abstracts available\n")
                f.write("  - Gap statements use different phrasing\n")
                f.write("  - Field is well-established with fewer open questions\n\n")
            
            # Keyword analysis
            if gap_data['top_keywords']:
                f.write("\n" + "="*60 + "\n")
                f.write("TOP RESEARCH KEYWORDS (Emerging Themes):\n")
                f.write("="*60 + "\n\n")
                for keyword, count in gap_data['top_keywords']:
                    f.write(f"  • {keyword.title()} (appears in {count} papers)\n")
        
        print(f"📋 Research gaps report saved: RESEARCH_GAPS.txt")
        print(f"   Found {gap_data['total_gaps_found']} gap statements in {len(gap_data.get('gap_categories', {}))} categories")
        
        # 2. Save CSV
        csv_filename = os.path.join(self.output_dir, "MASTER_REPORT_FINAL.csv")
        keys = ['relevance_score', 'source_count', 'ieee_authors', 'title', 'venue', 'year',
                'citations', 'impact_factor', 'h_index', 'influential_citations',
                'doi', 'url', 'abstract', 'abstract_source', 'found_in', 'keywords', 'tldr', 'recency_boosted']

        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in results:
                writer.writerow({k: row.get(k, '') for k in keys})

        # 3. Save Executive Summary
        summary_path = os.path.join(self.output_dir, "EXECUTIVE_SUMMARY.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"=== SROrch EXECUTIVE RESEARCH SUMMARY ===\n")
            f.write(f"QUERY: {query}\n")
            f.write(f"DATE:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*42 + "\n\n")
            f.writelines(top_abstract_blocks)

        # 4. Visualization & Console Output
        self.generate_research_statistics(results)

        # Export formats
        if 'json' in self.config['export_formats']:
            self.export_json(results)
        if 'bibtex' in self.config['export_formats']:
            self.export_bibtex(results)

        self.generate_session_report()

        # 5. Console TOP 5
        print(f"\n--- TOP 5 DISCOVERIES ---")
        for i, p in enumerate(results[:5], 1):
            boost_indicator = " 🔥" if p.get('recency_boosted') else ""
            print(f"[{i}] Score: {p['relevance_score']} | Sources: {p['source_count']}{boost_indicator}")
            print(f"    {p['ieee_authors']}, \"{p['title']}\"")
            print(f"    🔗 Link: {p.get('url', 'N/A')}")
            if p.get('tldr'):
                print(f"    💡 TLDR: {p['tldr'][:100]}...")

        # 6. Archive
        try:
            shutil.make_archive(self.output_dir, 'zip', self.output_dir)
            print(f"\n📦 Portable Archive Created: {self.output_dir}.zip")
        except: pass

if __name__ == "__main__":
    custom_config = {
        'abstract_limit': 10,
        'high_consensus_threshold': 4,
        'citation_weight': 1.5,
        'source_weight': 100,
        'enable_alerts': True,
        'enable_visualization': True,
        'export_formats': ['csv', 'json', 'bibtex'],
        'recency_boost': True,
        'recency_years': 5,
        'recency_multiplier': 1.2
    }

    search_query = "Langerhans Cell Histiocytosis"
    orchestrator = ResearchOrchestrator(config=custom_config)
    results = orchestrator.run_search(search_query, limit_per_engine=25)
    orchestrator.save_master_csv(results, search_query)

    print(f"\n--- ENHANCED SUMMARY ---")
    print(f"Search completed in {(orchestrator.session_metadata['end_time'] - orchestrator.session_metadata['start_time']).total_seconds():.2f} seconds")
    print(f"Engines used: {len(orchestrator.session_metadata['successful_engines'])}/{orchestrator.session_metadata['total_api_calls']}")
    print(f"Output directory: {orchestrator.output_dir}")
