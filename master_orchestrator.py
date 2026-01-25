"""
SROrch Master Orchestrator - Fixed Version
Handles missing API keys gracefully and prevents crashes
"""

import sys
import time
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import utilities with try-except to handle missing dependencies
try:
    from s2_utils import search_semantic_scholar
    S2_AVAILABLE = True
except ImportError:
    S2_AVAILABLE = False
    print("Warning: s2_utils not available. Semantic Scholar disabled.")

try:
    from arxiv_utils import search_arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False
    print("Warning: arxiv_utils not available. arXiv disabled.")

try:
    from pubmed_utils import search_pubmed
    PUBMED_AVAILABLE = True
except ImportError:
    PUBMED_AVAILABLE = False
    print("Warning: pubmed_utils not available. PubMed disabled.")

try:
    from scholar_utils import search_google_scholar
    SCHOLAR_AVAILABLE = True
except ImportError:
    SCHOLAR_AVAILABLE = False
    print("Warning: scholar_utils not available. Google Scholar disabled.")

try:
    from doi_utils import search_crossref
    DOI_AVAILABLE = True
except ImportError:
    DOI_AVAILABLE = False
    print("Warning: doi_utils not available. Crossref disabled.")

try:
    from openalex_utils import search_openalex
    OPENALEX_AVAILABLE = True
except ImportError:
    OPENALEX_AVAILABLE = False
    print("Warning: openalex_utils not available. OpenAlex disabled.")

try:
    from core_utils import search_core
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    print("Warning: core_utils not available. CORE disabled.")


class SROrchestrator:
    """
    Master orchestrator with robust error handling and graceful degradation
    """
    
    def __init__(self, api_keys: Dict[str, str], config: Optional[Dict] = None):
        """
        Initialize orchestrator with API keys and configuration
        
        Args:
            api_keys: Dictionary with keys: 's2', 'serp', 'core', 'email'
            config: Optional configuration dictionary
        """
        self.api_keys = api_keys or {}
        self.config = config or {}
        
        # Validate and sanitize API keys
        self.s2_key = self.api_keys.get('s2', '').strip()
        self.serp_key = self.api_keys.get('serp', '').strip()
        self.core_key = self.api_keys.get('core', '').strip()
        self.email = self.api_keys.get('email', 'researcher@example.com').strip()
        
        # Track available engines
        self.available_engines = self._check_available_engines()
        
    def _check_available_engines(self) -> Dict[str, bool]:
        """Check which engines are available based on API keys and imports"""
        engines = {
            'semantic_scholar': S2_AVAILABLE and bool(self.s2_key),
            'arxiv': ARXIV_AVAILABLE and bool(self.email),
            'pubmed': PUBMED_AVAILABLE and bool(self.email),
            'google_scholar': SCHOLAR_AVAILABLE and bool(self.serp_key),
            'crossref': DOI_AVAILABLE and bool(self.email),
            'openalex': OPENALEX_AVAILABLE and bool(self.email),
            'core': CORE_AVAILABLE and bool(self.core_key)
        }
        
        # Log available engines
        available = [k for k, v in engines.items() if v]
        unavailable = [k for k, v in engines.items() if not v]
        
        print(f"\n✓ Available engines ({len(available)}): {', '.join(available)}")
        if unavailable:
            print(f"✗ Unavailable engines ({len(unavailable)}): {', '.join(unavailable)}")
            self._log_missing_requirements(unavailable)
        
        return engines
    
    def _log_missing_requirements(self, unavailable: List[str]):
        """Log why engines are unavailable"""
        for engine in unavailable:
            if engine == 'semantic_scholar':
                if not S2_AVAILABLE:
                    print(f"  - {engine}: s2_utils module not found")
                elif not self.s2_key:
                    print(f"  - {engine}: S2_API_KEY not configured")
            
            elif engine == 'google_scholar':
                if not SCHOLAR_AVAILABLE:
                    print(f"  - {engine}: scholar_utils module not found")
                elif not self.serp_key:
                    print(f"  - {engine}: SERP_API_KEY not configured (required for Google Scholar)")
            
            elif engine == 'core':
                if not CORE_AVAILABLE:
                    print(f"  - {engine}: core_utils module not found")
                elif not self.core_key:
                    print(f"  - {engine}: CORE_API_KEY not configured")
            
            elif engine == 'arxiv':
                if not ARXIV_AVAILABLE:
                    print(f"  - {engine}: arxiv_utils module not found")
                elif not self.email:
                    print(f"  - {engine}: USER_EMAIL not configured")
            
            elif engine == 'pubmed':
                if not PUBMED_AVAILABLE:
                    print(f"  - {engine}: pubmed_utils module not found")
                elif not self.email:
                    print(f"  - {engine}: USER_EMAIL not configured")
            
            elif engine == 'crossref':
                if not DOI_AVAILABLE:
                    print(f"  - {engine}: doi_utils module not found")
                elif not self.email:
                    print(f"  - {engine}: USER_EMAIL not configured")
            
            elif engine == 'openalex':
                if not OPENALEX_AVAILABLE:
                    print(f"  - {engine}: openalex_utils module not found")
                elif not self.email:
                    print(f"  - {engine}: USER_EMAIL not configured")
    
    def search(self, query: str, max_results: int = 20) -> Dict[str, List]:
        """
        Execute parallel search across all available engines
        
        Args:
            query: Search query string
            max_results: Maximum results per engine
            
        Returns:
            Dictionary mapping engine names to result lists
        """
        if not self.available_engines:
            raise RuntimeError("No search engines available. Please configure API keys.")
        
        results = {}
        futures = {}
        
        with ThreadPoolExecutor(max_workers=7) as executor:
            # Submit tasks only for available engines
            if self.available_engines['semantic_scholar']:
                futures['semantic_scholar'] = executor.submit(
                    self._safe_search, search_semantic_scholar, query, max_results, self.s2_key
                )
            
            if self.available_engines['arxiv']:
                futures['arxiv'] = executor.submit(
                    self._safe_search, search_arxiv, query, max_results, self.email
                )
            
            if self.available_engines['pubmed']:
                futures['pubmed'] = executor.submit(
                    self._safe_search, search_pubmed, query, max_results, self.email
                )
            
            if self.available_engines['google_scholar']:
                futures['google_scholar'] = executor.submit(
                    self._safe_search, search_google_scholar, query, max_results, self.serp_key
                )
            
            if self.available_engines['crossref']:
                futures['crossref'] = executor.submit(
                    self._safe_search, search_crossref, query, max_results, self.email
                )
            
            if self.available_engines['openalex']:
                futures['openalex'] = executor.submit(
                    self._safe_search, search_openalex, query, max_results, self.email
                )
            
            if self.available_engines['core']:
                futures['core'] = executor.submit(
                    self._safe_search, search_core, query, max_results, self.core_key
                )
            
            # Collect results
            for engine, future in as_completed(futures):
                try:
                    results[engine] = future.result(timeout=30)
                    print(f"✓ {engine}: {len(results[engine])} results")
                except Exception as e:
                    print(f"✗ {engine} failed: {str(e)}")
                    results[engine] = []
        
        return results
    
    def _safe_search(self, search_func, query: str, max_results: int, api_key: str) -> List:
        """
        Wrapper for safe search execution with error handling
        
        Args:
            search_func: Search function to call
            query: Search query
            max_results: Maximum results
            api_key: API key or email
            
        Returns:
            List of results or empty list on error
        """
        try:
            return search_func(query, max_results, api_key)
        except Exception as e:
            print(f"Error in {search_func.__name__}: {str(e)}")
            return []
    
    def get_status_report(self) -> str:
        """Generate a status report of available engines"""
        available_count = sum(1 for v in self.available_engines.values() if v)
        total_count = len(self.available_engines)
        
        report = [
            f"SROrch Status Report",
            f"=" * 50,
            f"Available Engines: {available_count}/{total_count}",
            f"",
            f"Configuration:"
        ]
        
        for engine, available in self.available_engines.items():
            status = "✓ Ready" if available else "✗ Unavailable"
            report.append(f"  {engine:20s}: {status}")
        
        return "\n".join(report)


def create_orchestrator(api_keys: Dict[str, str], config: Optional[Dict] = None) -> SROrchestrator:
    """
    Factory function to create orchestrator instance
    
    Args:
        api_keys: Dictionary with API keys
        config: Optional configuration
        
    Returns:
        Configured SROrchestrator instance
    """
    return SROrchestrator(api_keys, config)


if __name__ == "__main__":
    # Example usage
    test_keys = {
        's2': '',  # Optional
        'serp': '',  # Optional - Google Scholar won't work without this
        'core': '',  # Optional
        'email': 'researcher@example.com'  # Required for arXiv, PubMed, Crossref, OpenAlex
    }
    
    orchestrator = create_orchestrator(test_keys)
    print(orchestrator.get_status_report())
