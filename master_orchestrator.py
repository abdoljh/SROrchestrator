# SROrch (Scholarly Research Orchestrator) - Enhanced Version
# master_orchestrator_enhanced.py

# master_orchestrator.py
import os
import csv
import re
import shutil
import json
import concurrent.futures
import requests
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Optional

# Import your utility modules
import s2_utils
import arxiv_utils
import pubmed_utils
import scholar_utils
import doi_utils
import openalex_utils
import core_utils
import scopus_utils

load_dotenv()

class ResearchOrchestrator:
    def __init__(self, config: Optional[Dict] = None):
        """
        Initializes the Orchestrator with API keys and custom research weights.
        """
        self.api_keys = {
            's2': os.getenv('S2_API_KEY'),
            'serp': os.getenv('SERP_API_KEY'),
            'core': os.getenv('CORE_API_KEY'),
            'scopus': os.getenv('SCOPUS_API_KEY'),
            'email': os.getenv('USER_EMAIL', 'researcher@example.com')
        }
        self.output_dir = ""
        
        # Comprehensive Configuration
        self.config = config or {
            'abstract_limit': 10,
            'high_consensus_threshold': 3,
            'citation_weight': 1.5,
            'source_weight': 100,
            'enable_alerts': True,
            'export_formats': ['csv', 'json', 'txt']
        }
        
        self.session_metadata = {
            'query': "",
            'start_time': None,
            'end_time': None,
            'successful_engines': []
        }

    def create_output_directory(self, query: str):
        """Creates a timestamped directory for session outputs."""
        clean_q = re.sub(r'[^a-zA-Z0-9]', '_', query).strip('_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f"search_{clean_q}_{timestamp}"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        return self.output_dir

    def run_search(self, query: str, limit_per_engine: int = 25):
        """
        Executes parallel searches across 8 engines.
        """
        self.session_metadata['query'] = query
        self.session_metadata['start_time'] = datetime.now()
        self.create_output_directory(query)
        all_papers = []

        print(f"\n[Master] Starting federated search for: '{query}'...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            tasks = {
                executor.submit(s2_utils.fetch_and_process_papers, self.api_keys['s2'], query, csv_limit=limit_per_engine): "Semantic Scholar",
                executor.submit(arxiv_utils.fetch_and_process_arxiv, query, max_limit=limit_per_engine): "arXiv",
                executor.submit(pubmed_utils.fetch_and_process_pubmed, query, max_limit=limit_per_engine): "PubMed",
                executor.submit(scholar_utils.fetch_and_process_scholar, self.api_keys['serp'], query, max_limit=limit_per_engine): "Google Scholar",
                executor.submit(doi_utils.fetch_and_process_doi, query, max_limit=limit_per_engine): "Crossref/DOI",
                executor.submit(openalex_utils.fetch_and_process_openalex, query, max_limit=limit_per_engine): "OpenAlex",
                executor.submit(core_utils.fetch_and_process_core, self.api_keys['core'], query, max_limit=limit_per_engine): "CORE",
                executor.submit(scopus_utils.fetch_and_process_scopus, self.api_keys['scopus'], query, max_limit=limit_per_engine): "Scopus"
            }

            for future in concurrent.futures.as_completed(tasks):
                engine_name = tasks[future]
                try:
                    data = future.result()
                    if data:
                        all_papers.extend(data)
                        self.session_metadata['successful_engines'].append(engine_name)
                        print(f"✅ {engine_name}: Found {len(data)} results")
                except Exception as e:
                    print(f"❌ {engine_name} Failed: {e}")

        results = self.deduplicate_and_score(all_papers)
        self.session_metadata['end_time'] = datetime.now()
        return results

    def deduplicate_and_score(self, all_papers: List[Dict]):
        """
        Deduplicates by Title/DOI and applies the SROrch Ranking Algorithm.
        """
        unique_papers = {}
        for paper in all_papers:
            # Normalize title for matching
            title_key = re.sub(r'[^a-z0-9]', '', paper['title'].lower())
            doi_key = paper.get('doi', '').lower().strip() if paper.get('doi') else None
            
            match_key = doi_key if (doi_key and doi_key != 'n/a') else title_key
            
            if match_key not in unique_papers:
                paper['source_count'] = 1
                paper['found_in'] = [paper.get('venue', 'Unknown')]
                unique_papers[match_key] = paper
            else:
                unique_papers[match_key]['source_count'] += 1
                if paper.get('venue') not in unique_papers[match_key]['found_in']:
                    unique_papers[match_key]['found_in'].append(paper.get('venue'))

        # Final Scoring
        scored_list = []
        for p in unique_papers.values():
            cites = int(p.get('citations')) if str(p.get('citations')).isdigit() else 0
            p['relevance_score'] = (p['source_count'] * self.config['source_weight']) + (cites * self.config['citation_weight'])
            scored_list.append(p)

            # Consensus Alert logic
            if self.config['enable_alerts'] and p['source_count'] >= self.config['high_consensus_threshold']:
                print(f"🚨 HIGH CONSENSUS: '{p['title'][:60]}...' found in {p['source_count']} sources.")

        return sorted(scored_list, key=lambda x: x['relevance_score'], reverse=True)

    def fetch_abstracts_for_top_papers(self, results: List[Dict]):
        """
        DEEP LOOK: Explicitly fetches abstracts for the top ranked papers.
        """
        print(f"\n[AI] Performing 'Deep Look' for top {self.config['abstract_limit']} papers...")
        summaries = []
        for i, paper in enumerate(results[:self.config['abstract_limit']]):
            abstract = "Abstract not available via public API."
            doi = paper.get('doi')
            
            if doi and doi != 'N/A':
                try:
                    # Semantic Scholar lookup for abstract
                    s2_url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=abstract"
                    resp = requests.get(s2_url, timeout=5)
                    if resp.status_code == 200:
                        abstract = resp.json().get('abstract') or abstract
                except: pass
            
            paper['abstract'] = abstract
            summary_block = (
                f"RANK [{i+1}] | SCORE: {paper['relevance_score']}\n"
                f"TITLE: {paper['title']}\n"
                f"AUTHORS: {paper['ieee_authors']}\n"
                f"ABSTRACT: {abstract[:500]}...\n"
                f"LINK: {paper.get('url')}\n"
                f"{'-'*40}"
            )
            summaries.append(summary_block)
        return summaries

    def save_master_csv(self, results: List[Dict], query: str):
        """Generates all output files and wraps them in a ZIP archive."""
        # 1. Enrich top results
        abstract_blocks = self.fetch_abstracts_for_top_papers(results)

        # 2. Export CSV
        csv_path = os.path.join(self.output_dir, "MASTER_REPORT_FINAL.csv")
        headers = ['relevance_score', 'source_count', 'ieee_authors', 'title', 'venue', 'year', 'citations', 'doi', 'url', 'abstract']
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in results:
                writer.writerow({k: row.get(k, 'N/A') for k in headers})

        # 3. Export JSON
        if 'json' in self.config['export_formats']:
            json_path = os.path.join(self.output_dir, "research_data.json")
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=4)

        # 4. Save Session Report
        report_path = os.path.join(self.output_dir, "SESSION_REPORT.txt")
        with open(report_path, 'w') as f:
            f.write("SROrch SESSION REPORT\n" + "="*30 + "\n")
            f.write(f"Query: {query}\n")
            f.write(f"Successful Engines: {', '.join(self.session_metadata['successful_engines'])}\n")
            f.write(f"Duration: {self.session_metadata['end_time'] - self.session_metadata['start_time']}\n")
            f.write(f"Total Unique Papers: {len(results)}\n")

        # 5. Archive
        shutil.make_archive(self.output_dir, 'zip', self.output_dir)
        print(f"\n✅ All reports generated in {self.output_dir}")
        print(f"📦 Archive ready: {self.output_dir}.zip")

# --- Execute ---
if __name__ == "__main__":
    # ✨ NEW: Customizable configuration
    custom_config = {
        'abstract_limit': 10,  # Get abstracts for top 10 papers
        'high_consensus_threshold': 4,
        'citation_weight': 1.5,  # Give more weight to citations
        'source_weight': 100,
        'enable_alerts': True,
        'enable_visualization': True,
        'export_formats': ['csv', 'json', 'bibtex'],  # Export all formats
        'recency_boost': True,  # Boost papers from last 5 years
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
