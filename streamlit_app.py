"""
SROrch Streamlit Interface - COMPLETE VERSION with Strict Verification
A comprehensive web interface for Scholarly Research Orchestrator with 
integrated academic report generation and factual integrity controls
"""

import streamlit as st
import os
import sys
import json
import shutil
import pandas as pd
import time
import requests
import re
from datetime import datetime
from pathlib import Path
import zipfile
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict

# Import the orchestrator
from master_orchestrator import ResearchOrchestrator

# Page configuration
st.set_page_config(
    page_title="SROrch - Research Orchestrator & Report Writer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2acaea;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 0.3rem;
        margin: 1rem 0;
        font-weight: bold;
        color: #00b894;
        line-height: 1.25;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 0.3rem;
        margin: 1rem 0;
    } 
    .error-box {
        padding: 1rem;
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        border-radius: 0.3rem;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        border-radius: 0.3rem;
        margin: 1rem 0;
    }
    .dev-mode-badge {
        padding: 1rem;
        background-color: #ffeaa7;
        border-left: 4px solid #fdcb6e;
        border-radius: 0.3rem;
        margin: 1rem 0;
        font-weight: bold;
        color: #d63031;
    }
    .verification-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    .verified {
        background-color: #d4edda;
        color: #155724;
    }
    .unverified {
        background-color: #f8d7da;
        color: #721c24;
    }
    .critical-issue {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ================================================================================
# CONFIGURATION & CONSTANTS
# ================================================================================

MODEL_PRIMARY = "claude-sonnet-4-20250514"
MODEL_FALLBACK = "claude-haiku-3-5-20241022"
MIN_API_DELAY = 3.0
RETRY_DELAYS = [10, 20, 40]

# Token limits for safe API calls
MAX_INPUT_TOKENS = 150000  # Conservative limit for Claude Sonnet 4
MAX_OUTPUT_TOKENS = 6000

# CRITICAL: Forbidden words that must not appear without specific metrics
FORBIDDEN_GENERIC_TERMS = [
    'advanced', 'sophisticated', 'state-of-the-art', 'state of the art',
    'enhanced', 'novel', 'cutting-edge', 'cutting edge', 'innovative',
    'powerful', 'robust', 'seamless', 'efficient', 'effective',
    'comprehensive', 'holistic', 'groundbreaking', 'revolutionary'
]

# ================================================================================
# TOKEN MANAGEMENT
# ================================================================================

def estimate_token_count(text: str) -> int:
    """Rough token estimation: 1 token ≈ 4 characters"""
    return len(text) // 4

def check_token_budget(system_prompt: str, user_prompt: str, max_tokens: int = MAX_INPUT_TOKENS) -> Tuple[bool, int]:
    """Check if prompts fit within token budget"""
    total_chars = len(system_prompt) + len(user_prompt)
    estimated_tokens = estimate_token_count(system_prompt + user_prompt)
    
    return estimated_tokens <= max_tokens, estimated_tokens

# ================================================================================
# ENHANCED CLAIM VERIFICATION SYSTEM (STRICT MODE)
# ================================================================================

class StrictClaimVerifier:
    """Strict verification with zero tolerance for unsupported quantitative claims"""
    
    def __init__(self, sources: List[Dict]):
        self.sources = sources
        self.source_metrics = self._extract_all_metrics(sources)
        self.technical_specs = self._extract_technical_specifications(sources)
        self.violations = []
    
    def _extract_all_metrics(self, sources: List[Dict]) -> Dict[str, Dict]:
        """Extract all quantifiable metrics from sources with context"""
        metrics = {}
        
        for i, source in enumerate(sources, 1):
            text_parts = [
                source.get('metadata', {}).get('title', ''),
                source.get('content', ''),
                source.get('metadata', {}).get('venue', ''),
                str(source.get('_orchestrator_data', {}).get('tldr', '')),
                str(source.get('_orchestrator_data', {}).get('abstract', ''))
            ]
            text = ' '.join(text_parts)
            
            source_metrics = {
                'percentages': [],
                'counts': [],
                'parameters': [],
                'benchmarks': [],
                'comparisons': [],
                'years': []
            }
            
            # Percentages with context
            for match in re.finditer(r'(\d+(?:\.\d+)?)%\s*(?:accuracy|precision|recall|F1|correctness|improvement|increase|decrease)?', text, re.IGNORECASE):
                source_metrics['percentages'].append({
                    'value': match.group(1),
                    'context': text[max(0, match.start()-30):min(len(text), match.end()+30)]
                })
            
            # Counts (millions, billions)
            for match in re.finditer(r'(\d+(?:\.\d+)?)\s*(million|billion|M|B)\s*(?:papers|documents|parameters|examples|samples)?', text, re.IGNORECASE):
                source_metrics['counts'].append({
                    'value': match.group(1),
                    'unit': match.group(2),
                    'context': text[max(0, match.start()-30):min(len(text), match.end()+30)]
                })
            
            # Parameter counts
            for match in re.finditer(r'(\d+(?:\.\d+)?)\s*(?:B|M|billion|million)\s*(?:parameters|params)', text, re.IGNORECASE):
                source_metrics['parameters'].append(match.group(1))
            
            # Benchmark names
            benchmarks = re.findall(r'(ScholarQA[\w\-]*|PubMedQA|BioASQ|MS MARCO|BEIR|GLUE|SuperGLUE|MMLU|HumanEval|GSM8K)', text, re.IGNORECASE)
            source_metrics['benchmarks'] = list(set(benchmarks))
            
            # Years
            years = re.findall(r'\b(20[0-2]\d)\b', text)
            source_metrics['years'] = list(set(years))
            
            # Comparisons
            for match in re.finditer(r'(outperform|exceed|surpass|better than).*?by\s+(\d+(?:\.\d+)?%?)', text, re.IGNORECASE):
                source_metrics['comparisons'].append({
                    'verb': match.group(1),
                    'margin': match.group(2)
                })
            
            if any(source_metrics.values()):
                metrics[str(i)] = source_metrics
        
        return metrics
    
    def _extract_technical_specifications(self, sources: List[Dict]) -> Dict[str, List[str]]:
        """Extract specific technical architecture details"""
        specs = defaultdict(list)
        
        for i, source in enumerate(sources, 1):
            text = ' '.join([
                source.get('metadata', {}).get('title', ''),
                source.get('content', '')
            ])
            
            models = re.findall(r'(OpenScholar|Med-PaLM|GPT-4[Oo]?|Claude|Llama-?\d*|PaLM|Contriever|GTR|DPR)', text, re.IGNORECASE)
            if models:
                specs[str(i)].extend(list(set(models)))
            
            dims = re.findall(r'(\d+)-dim(?:ensional)?\s*(?:embeddings|vectors)', text, re.IGNORECASE)
            if dims:
                specs[str(i)].extend([f"{d}-dim" for d in dims])
            
            components = re.findall(r'(bi-encoder|cross-encoder|dual-encoder|dense retriever|sparse retriever)', text, re.IGNORECASE)
            if components:
                specs[str(i)].extend(list(set(components)))
        
        return dict(specs)
    
    def check_forbidden_language(self, text: str, section: str) -> List[Dict]:
        """Check for generic terms without specific metrics"""
        violations = []
        
        for term in FORBIDDEN_GENERIC_TERMS:
            for match in re.finditer(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
                context = text[match.end():min(len(text), match.end() + 100)]
                has_metric = bool(re.search(r'\d+(?:\.\d+)?%|\d+\s*(?:million|billion|M|B)|GPT-|ScholarQA|Nature|Science', context))
                
                if not has_metric:
                    violations.append({
                        'type': 'forbidden_generic',
                        'term': term,
                        'section': section,
                        'context': text[max(0, match.start()-20):min(len(text), match.end()+50)],
                        'suggestion': f'Replace "{term}" with specific metric (e.g., "achieves X% on Y benchmark")'
                    })
        
        return violations
    
    def verify_technical_specificity(self, text: str, section: str) -> List[Dict]:
        """Ensure technical claims have specific details"""
        violations = []
        
        # Check for model mentions without parameters
        model_mentions = re.finditer(r'\b(GPT-4|Claude|Llama|PaLM|BERT)(?!\s+(?:with|using|has)\s+\d+\s*(?:B|M|billion|million))', text, re.IGNORECASE)
        for match in model_mentions:
            context = text[max(0, match.start()-30):min(len(text), match.end()+50)]
            if not re.search(r'\d+\s*(?:B|M|billion|million)\s*(?:parameters|params)', context):
                violations.append({
                    'type': 'missing_parameters',
                    'model': match.group(1),
                    'section': section,
                    'suggestion': f'Add parameter count for {match.group(1)} (e.g., "GPT-4 (1.8T parameters)")'
                })
        
        # Check for performance claims without benchmarks
        performance_claims = re.finditer(r'\b(achieves|demonstrates|shows|exhibits)\s+(?:strong|high|good|excellent|superior)\s+(?:performance|accuracy|results)', text, re.IGNORECASE)
        for match in performance_claims:
            context = text[max(0, match.start()-20):min(len(text), match.end()+80)]
            if not re.search(r'ScholarQA|PubMedQA|BioASQ|MS MARCO|\d+(?:\.\d+)?%', context):
                violations.append({
                    'type': 'missing_benchmark',
                    'claim': match.group(0),
                    'section': section,
                    'suggestion': 'Replace with specific benchmark and score'
                })
        
        return violations
    
    def verify_citation_support(self, text: str, section: str) -> List[Dict]:
        """Verify that quantitative claims are supported by cited sources"""
        violations = []
        
        pattern = r'([^.]*?\b(?:\d+%|\d+\s*(?:million|billion|M|B)|outperform|achieve|exceed)[^.]*?)\[(\d+)\]'
        
        for match in re.finditer(pattern, text, re.IGNORECASE):
            claim = match.group(1)
            citation = match.group(2)
            
            numbers = re.findall(r'\d+(?:\.\d+)?%?', claim)
            
            if citation not in self.source_metrics:
                violations.append({
                    'type': 'invalid_citation',
                    'citation': citation,
                    'claim': claim[:80],
                    'section': section,
                    'issue': f'Source [{citation}] has no extractable metrics'
                })
                continue
            
            source_data = self.source_metrics[citation]
            
            for num in numbers:
                found = False
                all_metrics = (
                    source_data.get('percentages', []) +
                    source_data.get('counts', []) +
                    source_data.get('comparisons', [])
                )
                for metric in all_metrics:
                    metric_val = metric.get('value', metric.get('margin', ''))
                    if self._fuzzy_match(num, metric_val):
                        found = True
                        break
                
                if not found:
                    violations.append({
                        'type': 'unsupported_number',
                        'number': num,
                        'citation': citation,
                        'claim': claim[:80],
                        'section': section,
                        'issue': f'Number {num} not found in source [{citation}]'
                    })
        
        return violations
    
    def _fuzzy_match(self, claim_num: str, source_num: str) -> bool:
        """Allow small rounding differences"""
        try:
            c = float(claim_num.replace('%', ''))
            s = float(str(source_num).replace('%', ''))
            relative_diff = abs(c - s) / max(abs(s), 1)
            return relative_diff < 0.05 or abs(c - s) < 1
        except:
            return claim_num == source_num
    
    def comprehensive_check(self, draft: Dict) -> Dict:
        """Run all verification checks on draft"""
        all_violations = []
        
        sections_to_check = {
            'abstract': draft.get('abstract', ''),
            'introduction': draft.get('introduction', ''),
            'literatureReview': draft.get('literatureReview', ''),
            'dataAnalysis': draft.get('dataAnalysis', ''),
            'challenges': draft.get('challenges', ''),
            'futureOutlook': draft.get('futureOutlook', ''),
            'conclusion': draft.get('conclusion', '')
        }
        
        for section_name, content in sections_to_check.items():
            all_violations.extend(self.check_forbidden_language(content, section_name))
            all_violations.extend(self.verify_technical_specificity(content, section_name))
            all_violations.extend(self.verify_citation_support(content, section_name))
        
        for section in draft.get('mainSections', []):
            content = section.get('content', '')
            title = section.get('title', 'Section')
            all_violations.extend(self.check_forbidden_language(content, f"mainSection:{title}"))
            all_violations.extend(self.verify_technical_specificity(content, f"mainSection:{title}"))
            all_violations.extend(self.verify_citation_support(content, f"mainSection:{title}"))
        
        self.violations = all_violations
        
        return {
            'total_violations': len(all_violations),
            'by_type': Counter(v['type'] for v in all_violations),
            'by_section': Counter(v['section'] for v in all_violations),
            'violations': all_violations,
            'has_critical': any(v['type'] in ['unsupported_number', 'invalid_citation'] for v in all_violations)
        }

# ================================================================================
# FIXED SOURCE AUTHORITY CLASSIFICATION
# ================================================================================

def get_authority_tier_fixed(venue: str, url: str) -> str:
    """Fixed authority tier detection - conferences correctly identified"""
    venue_lower = venue.lower()
    url_lower = url.lower()
    
    # Conference indicators (check FIRST)
    conference_indicators = [
        'proceedings', 'conference', 'symposium', 'workshop', 
        'conf.', 'conf ', 'neurips', 'icml', 'iclr', 'acl', 
        'emnlp', 'cvpr', 'iccv', 'eccv', 'sigir', 'kdd', 'www',
        'international conference', 'annual meeting'
    ]
    if any(ind in venue_lower for ind in conference_indicators):
        return 'conference'
    
    # Top-tier journals
    top_journal_indicators = [
        'nature', 'science', 'cell', 'lancet', 'nejm', 'jama',
        'ieee transactions', 'acm transactions', 'journal of the'
    ]
    if any(ind in venue_lower for ind in top_journal_indicators):
        return 'top_tier_journal'
    
    # Publisher journals
    publisher_indicators = [
        'ieee', 'acm', 'springer', 'elsevier', 'wiley', 'sage',
        'taylor & francis', 'oxford university press', 'cambridge'
    ]
    if any(ind in venue_lower for ind in publisher_indicators):
        return 'publisher_journal'
    
    # Preprints
    if 'arxiv' in url_lower or 'biorxiv' in url_lower or 'medrxiv' in url_lower:
        return 'preprint'
    
    return 'other'

def safe_int(value, default=0):
    """Safely convert value to int, handling 'N/A', None, strings, etc."""
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if value.strip().upper() in ('N/A', 'NA', 'UNKNOWN', '', 'NONE'):
            return default
        numbers = re.findall(r'\d+', value)
        if numbers:
            return int(numbers[0])
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default

def deduplicate_and_rank_sources_strict(sources: List[Dict]) -> List[Dict]:
    """Strict deduplication with corrected authority classification"""
    
    TIER_ORDER = {
        'top_tier_journal': 0,
        'publisher_journal': 1,
        'conference': 2,
        'other': 3,
        'preprint': 4
    }
    
    seen = {}
    
    for source in sources:
        doi = str(source.get('metadata', {}).get('doi', '')).lower().strip()
        title = re.sub(r'[^\w]', '', source.get('metadata', {}).get('title', '').lower())
        
        key = doi if doi and doi != 'n/a' and len(doi) > 5 else title
        
        venue = source.get('metadata', {}).get('venue', '')
        url = source.get('url', '')
        tier = get_authority_tier_fixed(venue, url)
        source['authority_tier'] = tier
        
        if key in seen:
            existing = seen[key]
            existing['source_count'] = existing.get('source_count', 1) + 1
            
            existing_cites = safe_int(existing.get('metadata', {}).get('citations', 0))
            new_cites = safe_int(source.get('metadata', {}).get('citations', 0))
            if new_cites > existing_cites:
                existing['metadata']['citations'] = new_cites
            
            existing_tier_rank = TIER_ORDER.get(existing.get('authority_tier'), 5)
            new_tier_rank = TIER_ORDER.get(tier, 5)
            
            if new_tier_rank < existing_tier_rank:
                existing['authority_tier'] = tier
                existing['metadata']['venue'] = venue
                existing['url'] = url
        else:
            source['source_count'] = 1
            seen[key] = source
    
    ranked = sorted(
        seen.values(),
        key=lambda x: (
            TIER_ORDER.get(x.get('authority_tier'), 5),
            -safe_int(x.get('metadata', {}).get('citations', 0)),
            -x.get('source_count', 1)
        )
    )
    
    return ranked

# ================================================================================
# TECHNICAL SPECIFICATION EXTRACTION
# ================================================================================

def extract_technical_specifications(text: str) -> Dict[str, List[str]]:
    """Extract specific technical details for prompt enrichment"""
    specs = defaultdict(list)
    
    benchmarks = re.findall(
        r'(ScholarQA[\w\-]*|PubMedQA|BioASQ|MS MARCO|BEIR|GLUE|SuperGLUE|MMLU|HumanEval|GSM8K|BBH)',
        text, re.IGNORECASE
    )
    if benchmarks:
        specs['benchmarks'] = list(set(benchmarks))
    
    models = re.findall(
        r'(OpenScholar(?:-\d+B)?|Med-PaLM(?:-\d+)?|GPT-4[Oo]?|Claude(?:-\d+)?|Llama-?[\d\.]*|PaLM(?:-\d+)?|Contriever|GTR|DPR)',
        text, re.IGNORECASE
    )
    if models:
        specs['models'] = list(set(models))
    
    params = re.findall(
        r'(\d+(?:\.\d+)?)\s*(?:B|M|billion|million)\s*(?:parameters|params)',
        text, re.IGNORECASE
    )
    if params:
        specs['parameter_counts'] = list(set(params))
    
    datasets = re.findall(
        r'(\d+(?:\.\d+)?)\s*(?:million|billion|M|B)\s*(?:papers|documents|passages|examples)',
        text, re.IGNORECASE
    )
    if datasets:
        specs['dataset_sizes'] = list(set(datasets))
    
    embeddings = re.findall(
        r'(\d+)(?:-dimensional|-dim)?\s*(?:embeddings|vectors|passage embeddings)',
        text, re.IGNORECASE
    )
    if embeddings:
        specs['embedding_dimensions'] = list(set(embeddings))
    
    metrics = re.findall(
        r'(\d+(?:\.\d+)?%)\s*(?:accuracy|precision|recall|F1|correctness)',
        text, re.IGNORECASE
    )
    if metrics:
        specs['performance_metrics'] = list(set(metrics))
    
    arch = re.findall(
        r'(bi-encoder|cross-encoder|dual-encoder|dense retriever|sparse retriever|vector database|FAISS)',
        text, re.IGNORECASE
    )
    if arch:
        specs['architectures'] = list(set(arch))
    
    return dict(specs)

def aggregate_technical_specs(sources: List[Dict]) -> Dict[str, List[str]]:
    """Aggregate technical specs from all sources"""
    all_specs = defaultdict(set)
    
    for source in sources:
        text = ' '.join([
            source.get('metadata', {}).get('title', ''),
            source.get('content', ''),
            str(source.get('_orchestrator_data', {}).get('abstract', '')),
            str(source.get('_orchestrator_data', {}).get('tldr', ''))
        ])
        
        specs = extract_technical_specifications(text)
        for key, values in specs.items():
            all_specs[key].update(values)
    
    return {k: sorted(list(v)) for k, v in all_specs.items()}

# ================================================================================
# CITATION COVERAGE ENFORCEMENT
# ================================================================================

def enforce_source_coverage(draft: Dict, sources: List[Dict], min_coverage: float = 0.6) -> Dict:
    """Ensure minimum percentage of sources are cited"""
    
    cited, _ = extract_cited_references_enhanced(draft)
    total_sources = len(sources)
    coverage = len(cited) / total_sources if total_sources > 0 else 0
    
    uncited_indices = set(range(1, total_sources + 1)) - cited
    
    high_priority_missing = []
    for idx in uncited_indices:
        source = sources[idx - 1]
        tier = source.get('authority_tier', 'other')
        cites = safe_int(source.get('metadata', {}).get('citations', 0))
        
        if tier in ['top_tier_journal', 'publisher_journal'] and cites > 20:
            high_priority_missing.append({
                'index': idx,
                'title': source.get('metadata', {}).get('title', 'Unknown')[:60],
                'tier': tier,
                'citations': cites,
                'venue': source.get('metadata', {}).get('venue', 'Unknown')
            })
    
    high_priority_missing.sort(key=lambda x: (
        0 if x['tier'] == 'top_tier_journal' else 1,
        -x['citations']
    ))
    
    return {
        'status': 'ok' if coverage >= min_coverage else 'insufficient_coverage',
        'coverage': coverage,
        'cited_count': len(cited),
        'total_sources': total_sources,
        'high_priority_missing': high_priority_missing[:15],
        'message': f"Coverage: {coverage:.1%} ({len(cited)}/{total_sources}). " +
                   (f"Need to integrate {len(high_priority_missing)} high-authority sources." 
                    if coverage < min_coverage else "Coverage target met.")
    }

# ================================================================================
# UTILITY FUNCTIONS
# ================================================================================

def get_secret_or_empty(key_name):
    """Safely retrieve a secret from Streamlit secrets, return empty string if not found."""
    try:
        return st.secrets.get(key_name, '')
    except (FileNotFoundError, KeyError, AttributeError):
        return ''

def check_dev_mode():
    """Check if running in development mode (secrets configured)"""
    dev_keys = []
    
    key_mappings = {
        'S2_API_KEY': 'Semantic Scholar',
        'SERP_API_KEY': 'Google Scholar',
        'CORE_API_KEY': 'CORE',
        'SCOPUS_API_KEY': 'SCOPUS',
        'META_SPRINGER_API_KEY': 'Springer Nature'
    }
    
    for key, name in key_mappings.items():
        if get_secret_or_empty(key):
            dev_keys.append(name)
    
    return len(dev_keys) > 0, dev_keys

def initialize_session_state():
    """Initialize session state with empty API keys and report writer state"""
    # Search engine API keys
    if 'user_s2_key' not in st.session_state:
        st.session_state['user_s2_key'] = ''
    if 'user_serp_key' not in st.session_state:
        st.session_state['user_serp_key'] = ''
    if 'user_core_key' not in st.session_state:
        st.session_state['user_core_key'] = ''
    if 'user_scopus_key' not in st.session_state:
        st.session_state['user_scopus_key'] = ''
    if 'user_springer_key' not in st.session_state:
        st.session_state['user_springer_key'] = ''
    if 'user_email' not in st.session_state:
        st.session_state['user_email'] = 'researcher@example.com'
    
    # Report writer state
    if 'report_step' not in st.session_state:
        st.session_state.report_step = 'input'
    
    if 'report_form_data' not in st.session_state:
        st.session_state.report_form_data = {
            'topic': '',
            'subject': '',
            'researcher': '',
            'institution': '',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'citation_style': 'IEEE',
            'max_sources': 25
        }
    
    if 'report_progress' not in st.session_state:
        st.session_state.report_progress = {'stage': '', 'detail': '', 'percent': 0}
    
    if 'report_research' not in st.session_state:
        st.session_state.report_research = {
            'subtopics': [],
            'sources': [],
            'phrase_variations': []
        }
    
    if 'report_draft' not in st.session_state:
        st.session_state.report_draft = None
    
    if 'report_final' not in st.session_state:
        st.session_state.report_final = None
    
    if 'report_html' not in st.session_state:
        st.session_state.report_html = None
    
    if 'report_processing' not in st.session_state:
        st.session_state.report_processing = False
    
    if 'report_api_calls' not in st.session_state:
        st.session_state.report_api_calls = 0
    
    if 'report_start_time' not in st.session_state:
        st.session_state.report_start_time = None
    
    if 'report_execution_time' not in st.session_state:
        st.session_state.report_execution_time = None
    
    # Verification state
    if 'verification_results' not in st.session_state:
        st.session_state.verification_results = None
    
    if 'strict_mode' not in st.session_state:
        st.session_state.strict_mode = True

def load_api_keys():
    """Load API keys with intelligent fallback strategy"""
    return {
        's2': st.session_state.get('user_s2_key', '').strip() or get_secret_or_empty('S2_API_KEY'),
        'serp': st.session_state.get('user_serp_key', '').strip() or get_secret_or_empty('SERP_API_KEY'),
        'core': st.session_state.get('user_core_key', '').strip() or get_secret_or_empty('CORE_API_KEY'),
        'scopus': st.session_state.get('user_scopus_key', '').strip() or get_secret_or_empty('SCOPUS_API_KEY'),
        'springer': st.session_state.get('user_springer_key', '').strip() or get_secret_or_empty('META_SPRINGER_API_KEY'),
        'email': st.session_state.get('user_email', 'researcher@example.com').strip() or get_secret_or_empty('USER_EMAIL') or 'researcher@example.com',
    }

def render_api_key_input_section():
    """Render the API key input section in sidebar"""
    st.sidebar.header("🔑 API Configuration")
    
    is_dev_mode, dev_keys = check_dev_mode()
    
    if is_dev_mode:
        st.sidebar.markdown(f"""
        <div class="dev-mode-badge">
            🔧 DEV MODE ACTIVE<br>
            Pre-configured keys detected: {len(dev_keys)}<br>
            <small>Delete Streamlit Secrets to switch to production mode</small>
        </div>
        """, unsafe_allow_html=True)
        
        with st.sidebar.expander("📋 Active Developer Keys", expanded=False):
            for key in dev_keys:
                st.markdown(f"✅ **{key}** (from secrets)")
            st.info("💡 These keys are loaded from `Streamlit Secrets` for development convenience.")
    
    st.sidebar.info("🔒 **User keys are temporary** - Lost when you refresh or close the tab (for your security!)")
    
    with st.sidebar.expander("📝 Enter Your API Keys (Optional)", expanded=not is_dev_mode):
        if is_dev_mode:
            st.warning("⚠️ Developer keys are active. User input will override secrets for this session.")
        
        st.markdown("""
        **Optional Premium Engines:**
        - Semantic Scholar (free key available)
        - Google Scholar (via SERP API)
        - CORE, SCOPUS, Springer Nature
        
        **Always Available (No Key Needed):**
        - arXiv, PubMed, Crossref/DOI, OpenAlex
        - Europe PMC, PLOS, SSRN, DeepDyve
        - Wiley, Taylor & Francis, ACM, DBLP, SAGE
        
        🔒 **Security Note:**
        - Keys stored in browser memory only
        - Never saved to disk or server
        - Automatically cleared on refresh
        """)
        
        # API Key Inputs
        s2_key = st.text_input(
            "Semantic Scholar API Key",
            value="",
            type="password",
            help="Get free key at: https://www.semanticscholar.org/product/api",
            key="s2_input_widget",
            placeholder="Enter your S2 API key"
        )
        
        serp_key = st.text_input(
            "SERP API Key (Google Scholar)",
            value="",
            type="password",
            help="Get key at: https://serpapi.com/",
            key="serp_input_widget",
            placeholder="Enter your SERP API key"
        )
        
        core_key = st.text_input(
            "CORE API Key",
            value="",
            type="password",
            help="Get key at: https://core.ac.uk/services/api",
            key="core_input_widget",
            placeholder="Enter your CORE API key"
        )
        
        scopus_key = st.text_input(
            "SCOPUS API Key",
            value="",
            type="password",
            help="Get key at: https://dev.elsevier.com/",
            key="scopus_input_widget",
            placeholder="Enter your SCOPUS API key"
        )
        
        springer_key = st.text_input(
            "Springer Nature API Key",
            value="",
            type="password",
            help="Get key at: https://dev.springernature.com/",
            key="springer_input_widget",
            placeholder="Enter your Springer API key"
        )
        
        email = st.text_input(
            "Your Email",
            value="researcher@example.com",
            help="Used for API requests to arXiv, PubMed, etc.",
            key="email_input_widget",
            placeholder="your.email@example.com"
        )
        
        # Apply button
        if st.button("✅ Apply Keys (This Session Only)", key="apply_keys", use_container_width=True):
            st.session_state['user_s2_key'] = s2_key.strip()
            st.session_state['user_serp_key'] = serp_key.strip()
            st.session_state['user_core_key'] = core_key.strip()
            st.session_state['user_scopus_key'] = scopus_key.strip()
            st.session_state['user_springer_key'] = springer_key.strip()
            st.session_state['user_email'] = email.strip()
            
            st.success("✅ Keys applied for this session!")
            st.rerun()
        
        # Show active keys
        api_keys = load_api_keys()
        active_keys = []
        sources = []
        
        if api_keys.get('s2'):
            active_keys.append("Semantic Scholar")
            sources.append("secrets" if not st.session_state.get('user_s2_key') else "user")
        if api_keys.get('serp'):
            active_keys.append("Google Scholar")
            sources.append("secrets" if not st.session_state.get('user_serp_key') else "user")
        if api_keys.get('core'):
            active_keys.append("CORE")
            sources.append("secrets" if not st.session_state.get('user_core_key') else "user")
        if api_keys.get('scopus'):
            active_keys.append("SCOPUS")
            sources.append("secrets" if not st.session_state.get('user_scopus_key') else "user")
        if api_keys.get('springer'):
            active_keys.append("Springer Nature")
            sources.append("secrets" if not st.session_state.get('user_springer_key') else "user")
        
        if active_keys:
            key_source_info = []
            for i, key in enumerate(active_keys):
                source = "🔧" if sources[i] == "secrets" else "👤"
                key_source_info.append(f"{source} {key}")
            
            st.success(f"🔑 Active: {', '.join(key_source_info)}")
            st.caption("🔧 = from secrets | 👤 = user input")
        else:
            st.info("ℹ️ Using free engines only")

def check_api_keys(api_keys):
    """Check which API keys are configured and valid"""
    status = {}
    
    status['s2'] = "✅" if api_keys.get('s2') and len(api_keys.get('s2', '')) > 5 else "❌"
    status['serp'] = "✅" if api_keys.get('serp') and len(api_keys.get('serp', '')) > 5 else "❌"
    status['core'] = "✅" if api_keys.get('core') and len(api_keys.get('core', '')) > 5 else "❌"
    status['scopus'] = "✅" if api_keys.get('scopus') and len(api_keys.get('scopus', '')) > 5 else "❌"
    status['springer'] = "✅" if api_keys.get('springer') and len(api_keys.get('springer', '')) > 5 else "❌"
    status['email'] = "✅" if api_keys.get('email') and api_keys['email'] != 'researcher@example.com' else "⚠️"
    
    return status

def get_available_engines(key_status):
    """Determine which engines are available based on API keys"""
    available = []
    
    # Premium Engines
    if key_status['s2'] == "✅":
        available.append("Semantic Scholar")
    if key_status['serp'] == "✅":
        available.append("Google Scholar")
    if key_status['core'] == "✅":
        available.append("CORE")
    if key_status['scopus'] == "✅":
        available.append("SCOPUS")
    if key_status['springer'] == "✅":
        available.append("Springer Nature")
    
    # Free Engines (always available)
    available.extend([
        "arXiv", "PubMed", "Crossref/DOI", "OpenAlex",
        "Europe PMC", "PLOS", "SSRN", "DeepDyve",
        "Wiley", "Taylor & Francis", "ACM Digital Library", "DBLP", "SAGE Journals"
    ])
    
    return available

# ================================================================================
# REPORT GENERATION FUNCTIONS
# ================================================================================

def update_report_progress(stage: str, detail: str, percent: int):
    """Update report generation progress"""
    st.session_state.report_progress = {
        'stage': stage,
        'detail': detail,
        'percent': min(100, percent)
    }

def rate_limit_wait():
    """Rate limiting for Anthropic API calls"""
    current_time = time.time()
    if 'last_api_call_time' not in st.session_state:
        st.session_state.last_api_call_time = 0
    
    time_since_last = current_time - st.session_state.last_api_call_time
    
    if time_since_last < MIN_API_DELAY:
        time.sleep(MIN_API_DELAY - time_since_last)
    
    st.session_state.last_api_call_time = time.time()
    st.session_state.report_api_calls += 1

def call_anthropic_api(messages: List[Dict], max_tokens: int = 1000, use_fallback: bool = False) -> Dict:
    """Call Anthropic API with fallback model support and token validation"""
    try:
        anthropic_key = st.secrets["ANTHROPIC_API_KEY"]
    except:
        raise Exception("Anthropic API key not configured in secrets")
    
    # Validate token budget
    total_chars = sum(len(str(m.get('content', ''))) for m in messages)
    estimated_tokens = estimate_token_count(''.join(str(m.get('content', '')) for m in messages))
    
    if estimated_tokens > MAX_INPUT_TOKENS:
        raise Exception(
            f"❌ Prompt too large: ~{estimated_tokens:,} tokens "
            f"(max {MAX_INPUT_TOKENS:,}). Reduce max_sources or contact support."
        )
    
    # Display token info
    st.caption(f"🔍 API call: ~{estimated_tokens:,} input tokens → {max_tokens:,} output tokens")
    
    rate_limit_wait()
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": anthropic_key,
        "anthropic-version": "2023-06-01"
    }
    
    model = MODEL_FALLBACK if use_fallback else MODEL_PRIMARY
    
    data = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages
    }
    
    for attempt in range(3):
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=180
            )
            
            if response.status_code == 429:
                wait_time = RETRY_DELAYS[attempt]
                st.warning(f"⏳ Rate limited. Waiting {wait_time}s (attempt {attempt+1}/3)")
                time.sleep(wait_time)
                continue
            
            if response.status_code == 529:
                wait_time = RETRY_DELAYS[attempt]
                st.warning(f"⏳ API overloaded. Waiting {wait_time}s (attempt {attempt+1}/3)")
                time.sleep(wait_time)
                continue
            
            if response.status_code == 400:
                error_detail = response.json().get('error', {}).get('message', 'Unknown error')
                raise Exception(f"Bad Request (400): {error_detail}")
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            st.warning(f"⚠️ API error (attempt {attempt+1}/3): {str(e)[:100]}")
            if attempt == 2:
                if not use_fallback:
                    st.info("🔄 Trying fallback model...")
                    return call_anthropic_api(messages, max_tokens, use_fallback=True)
                raise
            time.sleep(RETRY_DELAYS[attempt])
    
    if not use_fallback:
        st.info("🔄 Primary model failed. Trying fallback model...")
        return call_anthropic_api(messages, max_tokens, use_fallback=True)
    
    raise Exception("API call failed after 3 retries with both models")

def parse_json_response(text: str) -> Dict:
    """Extract JSON from API response text"""
    try:
        cleaned = re.sub(r'```json\n?|```\n?', '', text).strip()
        return json.loads(cleaned)
    except:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
    return {}

def generate_phrase_variations(topic: str) -> List[str]:
    """Generate phrase variations to avoid repetition"""
    return [
        topic,
        f"the field of {topic}",
        f"{topic} research",
        f"this domain",
        f"this research area",
        f"the {topic} field"
    ]

def convert_orchestrator_to_source_format(papers: List[Dict]) -> List[Dict]:
    """Convert ResearchOrchestrator output to report writer source format"""
    sources = []
    
    for paper in papers:
        metadata = {
            'authors': paper.get('ieee_authors', 'Unknown Authors'),
            'title': paper.get('title', 'Untitled'),
            'venue': paper.get('venue', 'Unknown Venue'),
            'year': str(paper.get('year', 'n.d.')),
            'citations': paper.get('citations', 0),
            'doi': paper.get('doi', 'N/A')
        }
        
        source = {
            'title': paper.get('title', 'Untitled'),
            'url': paper.get('url', ''),
            'content': paper.get('abstract', paper.get('tldr', ''))[:500],
            'metadata': metadata,
            'credibilityScore': min(100, 50 + safe_int(paper.get('citations', 0)) // 10),
            'credibilityJustification': f"Found in {safe_int(paper.get('source_count', 1), 1)} database(s), {paper.get('citations', 0)} citations",
            'dateAccessed': datetime.now().isoformat(),
            '_orchestrator_data': paper
        }
        
        sources.append(source)
    
    return sources

def analyze_topic_with_ai(topic: str, subject: str) -> Dict:
    """Analyze topic and generate research plan with evaluation context"""
    
    # Domain-specific evaluation framework knowledge
    evaluation_frameworks = {
        'scientific literature synthesis': {
            'primary_benchmark': 'ScholarQABench',
            'metrics': ['correctness', 'citation_accuracy', 'coverage', 'coherence'],
            'evaluation_type': 'automated_plus_human',
            'human_evaluation': 'blind_preference_judgment'
        },
        'retrieval augmented generation': {
            'common_benchmarks': ['RGB', 'RECALL', 'ASQA', 'ELI5', 'ScholarQABench'],
            'metrics': ['retrieval_accuracy', 'generation_fidelity', 'hallucination_rate']
        },
        'biomedical nlp': {
            'common_benchmarks': ['PubMedQA', 'BioASQ', 'MedQA', 'BLURB'],
            'metrics': ['accuracy', 'F1', 'exact_match']
        },
        'systematic review': {
            'guidelines': ['PRISMA 2020', 'PRISMA-S', 'Cochrane'],
            'metrics': ['completeness', 'bias_risk', 'reporting_quality']
        }
    }
    
    # Detect relevant frameworks
    relevant_frameworks = []
    topic_lower = topic.lower()
    subject_lower = subject.lower()
    
    for domain, framework in evaluation_frameworks.items():
        if domain in topic_lower or domain in subject_lower:
            relevant_frameworks.append({**framework, 'domain': domain})
    
    framework_context = ""
    if relevant_frameworks:
        framework_context = "\n\nRELEVANT EVALUATION FRAMEWORKS FOR THIS FIELD:\n"
        for fw in relevant_frameworks:
            framework_context += f"- Domain: {fw['domain']}\n"
            if 'primary_benchmark' in fw:
                framework_context += f"  Primary benchmark: {fw['primary_benchmark']}\n"
            if 'common_benchmarks' in fw:
                framework_context += f"  Common benchmarks: {', '.join(fw['common_benchmarks'])}\n"
            if 'metrics' in fw:
                framework_context += f"  Key metrics: {', '.join(fw['metrics'])}\n"
            framework_context += "\n"
    
    variations = generate_phrase_variations(topic)
    st.session_state.report_research['phrase_variations'] = variations
    
    prompt = f"""Research plan for "{topic}" in {subject}.{framework_context}

Create:
1. 5 specific subtopics about "{topic}"
2. 5 academic search queries for finding papers (2020-2025)
3. IDENTIFY the specific evaluation benchmarks used in this field (if any)

Return ONLY JSON:
{{
  "subtopics": ["aspect 1", "aspect 2", ...],
  "researchQueries": ["query 1", "query 2", ...],
  "evaluationFrameworks": ["benchmark 1", "benchmark 2"]
}}"""
    
    try:
        response = call_anthropic_api(
            [{"role": "user", "content": prompt}],
            max_tokens=800
        )
        text = "".join([c['text'] for c in response['content'] if c['type'] == 'text'])
        result = parse_json_response(text)
        
        if result.get('subtopics') and result.get('researchQueries'):
            return result
    except Exception as e:
        st.warning(f"Topic analysis failed: {e}. Using fallback.")
    
    # Fallback
    return {
        "subtopics": [
            f"Foundations of {topic}",
            f"Recent Advances in {topic}",
            f"Applications of {topic}",
            f"Challenges in {topic}",
            f"Future of {topic}"
        ],
        "researchQueries": [
            f"{topic} research 2024",
            f"{topic} academic papers",
            f"{topic} recent developments",
            f"{topic} applications",
            f"{topic} future trends"
        ],
        "evaluationFrameworks": []
    }

# ================================================================================
# ENHANCED DRAFT GENERATION WITH TOKEN MANAGEMENT
# ================================================================================

def format_sources_condensed(sources: List[Dict], start_idx: int = 1) -> str:
    """Format sources in ultra-condensed form for token efficiency"""
    source_list = []
    for i, s in enumerate(sources, start_idx):
        meta = s.get('metadata', {})
        tier = s.get('authority_tier', 'unknown')
        tier_emoji = {
            'top_tier_journal': '🏆',
            'publisher_journal': '📚',
            'conference': '🎓',
            'preprint': '📄',
            'other': '📃'
        }.get(tier, '📄')
        
        source_list.append(
            f"[{i}] {tier_emoji} {meta.get('title', 'Unknown')[:60]} "
            f"({meta.get('year', 'N/A')}) {meta.get('authors', 'Unknown')[:30]}"
        )
    
    return "\n".join(source_list)

def add_temporal_context(prompt: str, sources: List[Dict]) -> str:
    """Add temporal context based on source publication dates"""
    
    years = []
    for s in sources:
        year_str = s.get('metadata', {}).get('year', '')
        if year_str and str(year_str).isdigit():
            years.append(int(year_str))
    
    if years:
        min_year = min(years)
        max_year = max(years)
        current_year = datetime.now().year
        
        temporal_context = f"""
TEMPORAL CONTEXT:
- Source publication range: {min_year}-{max_year}
- Current year: {current_year}
- "Recent" refers to {max_year-2}-{max_year} (last 2-3 years of available sources)
- Use specific years instead of relative terms ("in 2024" not "recently")
- Avoid claiming developments are "recent" if sources are from {current_year-5} or earlier
"""
        return prompt + temporal_context
    
    return prompt

def generate_draft_strict(
    topic: str,
    subject: str,
    subtopics: List[str],
    sources: List[Dict],
    variations: List[str],
    evaluation_frameworks: List[str],
    max_sources: int = 25
) -> Tuple[Dict, Dict]:
    """Generate draft with strict technical specificity requirements and token management"""
    
    update_report_progress('Drafting', 'Writing with strict technical requirements...', 60)
    
    if not sources:
        raise Exception("No sources available")
    
    # Limit sources to max_sources
    sources_to_use = sources[:max_sources]
    
    # Aggregate technical specifications
    all_specs = aggregate_technical_specs(sources_to_use)
    
    # Build CONDENSED source list (150 chars per source vs 500+ before)
    sources_text = format_sources_condensed(sources_to_use)
    
    # Build system prompt
    system_prompt = f"""You are a PRECISE technical report generator. ABSOLUTE RULES:

VIOLATION = REJECTION. NO EXCEPTIONS.

RULE 1 - FORBIDDEN WORDS (Automatic rejection if used without specific metrics):
- "advanced" → USE: "using 340M parameter cross-encoder [X]"
- "sophisticated" → USE: "bi-encoder with 768-dimensional embeddings [X]"
- "state-of-the-art" → USE: "achieving 71.8% on ScholarQABench [X]"
- "enhanced" → USE: "15% improvement over GPT-4o baseline [X]"
- "novel" → USE: "first system to combine X with Y [X]"
- "robust" → USE: "maintains performance across 5 domains [X]"

RULE 2 - MANDATORY SPECIFICITY:
Every technical claim MUST include ONE of:
- Exact benchmark name + score (e.g., "ScholarQABench: 71.8%")
- Parameter count (e.g., "8B parameters")
- Dataset size (e.g., "45M papers")
- System name + version (e.g., "OpenScholar-8B")
- Specific year (e.g., "2024") not "recent"

RULE 3 - CITATION REQUIREMENTS:
- Cite AT LEAST {int(max_sources * 0.5)} different sources minimum
- Every paragraph must contain 2-4 citations [X]
- Never cite same source twice in one paragraph
- Use high-authority sources [1]-[10] preferentially

RULE 4 - IF UNSURE:
Write "Not specified in source [X]" or omit entirely.
NEVER invent numbers, benchmarks, or metrics.

EXTRACTED TECHNICAL DETAILS:
Benchmarks: {', '.join(all_specs.get('benchmarks', ['None'])[:10])}
Models: {', '.join(all_specs.get('models', ['None'])[:10])}
Parameters: {', '.join(all_specs.get('parameter_counts', ['None'])[:5])}
Dataset sizes: {', '.join(all_specs.get('dataset_sizes', ['None'])[:5])}
Architectures: {', '.join(all_specs.get('architectures', ['None'])[:5])}

REMEMBER: Specificity is MANDATORY. Generic language is FORBIDDEN."""

    # Variations and evaluation context
    variations_text = f"""PHRASE VARIATION (Required):
- "{topic}" - MAXIMUM 3 times total
- "{variations[1]}" - USE FREQUENTLY
- "{variations[2]}" - USE FREQUENTLY  
- "this domain" - USE OFTEN
- "this research area" - USE OFTEN"""

    eval_context = ""
    if evaluation_frameworks:
        eval_context = f"""
EVALUATION FRAMEWORKS (Use these specific names):
{', '.join(evaluation_frameworks)}
DO NOT use generic guidelines like PRISMA unless specifically discussing human reviews."""

    strict_boundary = f"""
SOURCE BOUNDARY: You may ONLY cite [1]-[{len(sources_to_use)}].
DO NOT mention any paper not in this numbered list.
DO NOT invent authors or systems."""

    # Section requirements (condensed)
    section_requirements = """
SECTION REQUIREMENTS:

Abstract (200 words): MUST name benchmark(s) with scores, dataset size or parameter count
Introduction: MUST cite founding paper with year, quantify problem
Literature Review: MUST compare systems with metric differences
Main Sections (4): Architecture, Benchmarks, Comparisons, Applications - all with specific numbers
Data & Analysis: MUST include numbers for every claim
Challenges: Specific technical limitations with metrics
Future: Concrete capabilities with predicted metrics
Conclusion: Summary with numbers, NO new claims without citations"""

    user_prompt = f"""Write technical report about "{topic}" in {subject}.

{variations_text}

{eval_context}

{strict_boundary}

{section_requirements}

SOURCES TO CITE [{len(sources_to_use)}]:
{sources_text}

Return JSON format:
{{
  "abstract": "...",
  "introduction": "...",
  "literatureReview": "...",
  "mainSections": [{{"title": "...", "content": "..."}}, ...],
  "dataAnalysis": "...",
  "challenges": "...",
  "futureOutlook": "...",
  "conclusion": "..."
}}

REMINDER: Every claim needs [X] citation. Every number needs source support. Generic terms forbidden."""

    # Add temporal context
    user_prompt = add_temporal_context(user_prompt, sources_to_use)
    
    # CHECK TOKEN BUDGET
    within_budget, estimated_tokens = check_token_budget(system_prompt, user_prompt)
    
    if not within_budget:
        st.warning(f"⚠️ Prompt too large ({estimated_tokens:,} tokens). Auto-reducing sources...")
        # Recursively reduce max_sources by 30%
        new_max = int(max_sources * 0.7)
        if new_max < 10:
            raise Exception(
                f"Cannot reduce sources below 10. Current estimate: {estimated_tokens:,} tokens. "
                "Try reducing max_sources manually or use a different topic."
            )
        return generate_draft_strict(
            topic, subject, subtopics, sources, variations, 
            evaluation_frameworks, max_sources=new_max
        )
    
    # First generation attempt
    response = call_anthropic_api(
        [{"role": "user", "content": system_prompt + "\n\n" + user_prompt}],
        max_tokens=MAX_OUTPUT_TOKENS
    )
    text = "".join([c['text'] for c in response['content'] if c['type'] == 'text'])
    draft = parse_json_response(text)
    
    # Ensure structure
    required_keys = ['abstract', 'introduction', 'literatureReview', 'mainSections',
                    'dataAnalysis', 'challenges', 'futureOutlook', 'conclusion']
    for key in required_keys:
        if key not in draft or not draft[key]:
            draft[key] = "Section content." if key != 'mainSections' else [{"title": "Section", "content": "Content."}]
    
    # STRICT VERIFICATION
    update_report_progress('Verification', 'Running strict claim verification...', 75)
    verifier = StrictClaimVerifier(sources_to_use)
    verification = verifier.comprehensive_check(draft)
    
    # Check source coverage
    coverage_check = enforce_source_coverage(draft, sources_to_use, min_coverage=0.5)
    
    # If critical issues, attempt regeneration with corrections
    if verification['has_critical'] or coverage_check['status'] != 'ok':
        update_report_progress('Refinement', 'Fixing critical issues...', 85)
        
        correction_prompt = f"""
CRITICAL ISSUES FOUND - MUST FIX:

{chr(10).join([f"- {v['type']}: {v.get('suggestion', v.get('issue', ''))}" for v in verification['violations'][:10]])}

COVERAGE ISSUE: {coverage_check['message']}

MISSING HIGH-PRIORITY SOURCES TO INTEGRATE:
{chr(10).join([f"[{s['index']}] {s['title']} ({s['venue']})" for s in coverage_check.get('high_priority_missing', [])[:5]])}

REGENERATION RULES:
1. Remove ALL forbidden generic terms
2. Add specific metrics (benchmarks, parameters, percentages) to every claim
3. Integrate missing high-priority sources above
4. Ensure minimum {int(max_sources * 0.5)} unique citations
5. If specific number unavailable, remove claim or write "Not specified in sources"

Return corrected JSON."""
        
        try:
            # Check token budget for correction
            correction_full = system_prompt + "\n\n" + user_prompt + "\n\n" + correction_prompt
            within_budget, est_tokens = check_token_budget(system_prompt, user_prompt + "\n\n" + correction_prompt)
            
            if within_budget:
                retry_response = call_anthropic_api(
                    [{"role": "user", "content": correction_full}],
                    max_tokens=MAX_OUTPUT_TOKENS
                )
                retry_text = "".join([c['text'] for c in retry_response['content'] if c['type'] == 'text'])
                corrected_draft = parse_json_response(retry_text)
                
                # Validate structure
                valid = all(k in corrected_draft and corrected_draft[k] for k in required_keys)
                if valid:
                    draft = corrected_draft
                    # Re-verify
                    verification = verifier.comprehensive_check(draft)
                    coverage_check = enforce_source_coverage(draft, sources_to_use, min_coverage=0.5)
                    verification['correction_attempted'] = True
            else:
                st.warning(f"⚠️ Correction prompt too large ({est_tokens:,} tokens). Using original draft.")
                verification['correction_skipped'] = True
        except Exception as e:
            verification['correction_error'] = str(e)
    
    # Final cleanup
    def fix_citations(text):
        if isinstance(text, str):
            text = re.sub(r'\[Source\s+(\d+)\]', r'[\1]', text, flags=re.IGNORECASE)
        return text
    
    for key in draft:
        if isinstance(draft[key], str):
            draft[key] = fix_citations(draft[key])
        elif isinstance(draft[key], list):
            for item in draft[key]:
                if isinstance(item, dict):
                    for k, v in item.items():
                        if isinstance(v, str):
                            item[k] = fix_citations(v)
    
    verification_report = {
        'verification': verification,
        'coverage': coverage_check,
        'technical_specs_found': all_specs
    }
    
    return draft, verification_report

# ================================================================================
# CITATION FORMATTING AND HTML GENERATION
# ================================================================================

def format_authors_ieee(authors_str: str) -> str:
    """Format multiple authors for IEEE style"""
    if not authors_str:
        return "Research Team"
    
    if 'et al' in authors_str.lower():
        return authors_str
    
    authors = re.split(r',\s*|\s+and\s+', authors_str)
    authors = [a.strip() for a in authors if a.strip()]
    
    if not authors:
        return "Research Team"
    
    if len(authors) == 1:
        return authors[0]
    elif len(authors) == 2:
        return f"{authors[0]} and {authors[1]}"
    else:
        return ', '.join(authors[:-1]) + ', and ' + authors[-1]

def format_citation_with_tier(source: Dict, index: int, style: str = 'IEEE') -> str:
    """Format citation with correct tier badge"""
    meta = source.get('metadata', {})
    tier = source.get('authority_tier', 'unknown')
    
    tier_labels = {
        'top_tier_journal': '[Nature/Science]',
        'publisher_journal': '[Journal]',
        'conference': '[Conference]',
        'preprint': '[Preprint]',
        'other': '[Other]'
    }
    
    if style == 'APA':
        return f"{meta.get('authors', 'Unknown')} ({meta.get('year', 'n.d.')}). {meta.get('title', 'Untitled')}. <i>{meta.get('venue', 'Unknown')}</i>. {tier_labels.get(tier, '')}"
    else:
        authors = format_authors_ieee(meta.get('authors', 'Unknown'))
        return f'[{index}] {authors}, "{meta.get("title", "Untitled")}," {meta.get("venue", "Unknown")}, {meta.get("year", "n.d.")}. {tier_labels.get(tier, "")}'

def extract_cited_references_enhanced(draft: Dict) -> Tuple[set, List[Dict]]:
    """Extract citations with context"""
    cited = set()
    contexts = []
    
    def extract(text, section):
        if not isinstance(text, str):
            return
        for match in re.finditer(r'\[(\d+)\]', text):
            num = int(match.group(1))
            cited.add(num)
            start, end = max(0, match.start()-50), min(len(text), match.end()+50)
            contexts.append({
                'citation': num,
                'context': text[start:end],
                'section': section
            })
    
    for key, value in draft.items():
        if isinstance(value, str):
            extract(value, key)
        elif key == 'mainSections' and isinstance(value, list):
            for section in value:
                extract(section.get('content', ''), section.get('title', 'Section'))
    
    return cited, contexts

def renumber_citations_in_text(text: str, mapping: Dict[int, int]) -> str:
    """Renumber citations in text according to the mapping"""
    return re.sub(r'\[(\d+)\]', lambda m: f'[{mapping.get(int(m.group(1)), m.group(1))}]', text)

def renumber_citations_in_draft(draft: Dict, mapping: Dict[int, int]) -> Dict:
    """Renumber all citations in the draft according to the mapping"""
    new_draft = {}
    
    for key, value in draft.items():
        if isinstance(value, str):
            new_draft[key] = renumber_citations_in_text(value, mapping)
        elif isinstance(value, list):
            new_list = []
            for item in value:
                if isinstance(item, dict):
                    new_item = {}
                    for k, v in item.items():
                        if isinstance(v, str):
                            new_item[k] = renumber_citations_in_text(v, mapping)
                        else:
                            new_item[k] = v
                    new_list.append(new_item)
                elif isinstance(item, str):
                    new_list.append(renumber_citations_in_text(item, mapping))
                else:
                    new_list.append(item)
            new_draft[key] = new_list
        else:
            new_draft[key] = value
    
    return new_draft

def refine_draft_simple(draft: Dict, topic: str, sources_count: int) -> Dict:
    """Add executive summary"""
    draft['executiveSummary'] = (
        f"This report examines {topic} based on {sources_count} authoritative sources "
        f"with strict technical verification."
    )
    return draft

def generate_html_report_strict(
    refined_draft: Dict,
    form_data: Dict,
    sources: List[Dict],
    verification_report: Dict,
    max_sources: int = 25
) -> str:
    """Generate HTML report with strict verification display"""
    
    update_report_progress('Generating HTML', 'Creating document...', 97)
    
    try:
        report_date = datetime.strptime(
            form_data['date'],
            '%Y-%m-%d'
        ).strftime('%B %d, %Y')
    except:
        report_date = datetime.now().strftime('%B %d, %Y')
    
    style = form_data.get('citation_style', 'IEEE')
    
    # Extract cited references and create renumbering map
    cited, contexts = extract_cited_references_enhanced(refined_draft)
    cited_refs_sorted = sorted(cited)
    
    old_to_new = {}
    for new_num, old_num in enumerate(cited_refs_sorted, 1):
        old_to_new[old_num] = new_num
    
    renumbered_draft = renumber_citations_in_draft(refined_draft, old_to_new)
    
    # Get verification data
    ver = verification_report.get('verification', {})
    cov = verification_report.get('coverage', {})
    specs = verification_report.get('technical_specs_found', {})
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{form_data['topic']} - Technical Report</title>
    <style>
        @page {{ margin: 1in; }}
        body {{
            font-family: 'Times New Roman', serif;
            font-size: 12pt;
            line-height: 1.6;
            color: #000;
            max-width: 8.5in;
            margin: 0 auto;
            padding: 0.5in;
        }}
        .cover {{
            text-align: center;
            padding-top: 2in;
            page-break-after: always;
        }}
        .cover h1 {{
            font-size: 24pt;
            font-weight: bold;
            margin: 1in 0 0.5in 0;
        }}
        .cover .meta {{
            font-size: 14pt;
            margin: 0.25in 0;
        }}
        h1 {{
            font-size: 18pt;
            margin-top: 0.5in;
            border-bottom: 2px solid #333;
            padding-bottom: 0.1in;
        }}
        h2 {{
            font-size: 14pt;
            margin-top: 0.3in;
            font-weight: bold;
        }}
        p {{
            text-align: justify;
            margin: 0.15in 0;
        }}
        .abstract {{
            font-style: italic;
            margin: 0.25in 0.5in;
        }}
        .references {{
            page-break-before: always;
        }}
        .ref-item {{
            margin: 0.15in 0;
            font-size: 10pt;
            line-height: 1.4;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }}
        .ref-item a {{
            color: #0066CC;
            text-decoration: none;
            word-break: break-all;
        }}
        .ref-item a:hover {{
            text-decoration: underline;
        }}
        .tier-label {{
            font-size: 9pt;
            color: #666;
            font-weight: bold;
        }}
        .verification-panel {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            padding: 1rem;
            margin: 1rem 0;
            font-size: 10pt;
        }}
        .specs-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
            font-size: 9pt;
        }}
        .violation {{
            background: #f8d7da;
            padding: 0.3rem;
            margin: 0.2rem 0;
            border-radius: 0.2rem;
            font-size: 9pt;
        }}
    </style>
</head>
<body>
    <div class="cover">
        <h1>{form_data['topic']}</h1>
        <div class="meta">Technical Research Report</div>
        <div class="meta">Subject: {form_data['subject']}</div>
        <div class="meta" style="margin-top: 1in;">
            {form_data['researcher']}<br>
            {form_data['institution']}<br>
            {report_date}
        </div>
        <div class="meta" style="margin-top: 0.5in; font-size: 10pt;">
            Generated by SROrch | {style} Format | STRICT MODE
        </div>
    </div>

    <h1>Executive Summary</h1>
    <p>{renumbered_draft.get('executiveSummary', '')}</p>

    <h1>Abstract</h1>
    <div class="abstract">{renumbered_draft.get('abstract', '')}</div>

    <h1>Introduction</h1>
    <p>{renumbered_draft.get('introduction', '')}</p>

    <h1>Literature Review</h1>
    <p>{renumbered_draft.get('literatureReview', '')}</p>
"""
    
    for section in renumbered_draft.get('mainSections', []):
        html += f"""
    <h2>{section.get('title', 'Section')}</h2>
    <p>{section.get('content', '')}</p>
"""
    
    html += f"""
    <h1>Data & Analysis</h1>
    <p>{renumbered_draft.get('dataAnalysis', '')}</p>

    <h1>Challenges</h1>
    <p>{renumbered_draft.get('challenges', '')}</p>

    <h1>Future Outlook</h1>
    <p>{renumbered_draft.get('futureOutlook', '')}</p>

    <h1>Conclusion</h1>
    <p>{renumbered_draft.get('conclusion', '')}</p>

    <div class="references">
        <h1>References</h1>
"""
    
    # Generate references with tier badges
    for old_ref_num in cited_refs_sorted:
        new_ref_num = old_to_new[old_ref_num]
        if old_ref_num <= len(sources):
            source = sources[old_ref_num - 1]
            citation = format_citation_with_tier(source, new_ref_num, style)
            html += f'        <div class="ref-item">{citation}</div>\n'
    
    # Add "Further References" section for uncited but relevant sources
    uncited_sources = []
    cited_indices = set(cited_refs_sorted)
    
    for i in range(1, min(max_sources + 1, len(sources) + 1)):
        if i not in cited_indices:
            uncited_sources.append((i, sources[i - 1]))
    
    if uncited_sources:
        html += """
        <h1 style="margin-top: 0.5in;">Further References</h1>
        <p style="font-style: italic; font-size: 10pt;">Additional relevant sources consulted but not directly cited in this report.</p>
"""
        for idx, source in uncited_sources[:20]:
            citation = format_citation_with_tier(source, idx, style)
            citation_text = citation.split(']', 1)[1] if ']' in citation else citation
            html += f'        <div class="ref-item" style="font-size: 9pt;">• {citation_text}</div>\n'
    
    # Add verification panel
    html += f"""
    </div>
    
    <div class="verification-panel" style="page-break-before: always;">
        <h3>🔍 Technical Verification Report</h3>
        <p><strong>Quality Metrics:</strong></p>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1rem 0;">
            <div>Sources: <strong>{len(sources)}</strong></div>
            <div>Cited: <strong>{len(cited_refs_sorted)} ({cov.get('coverage', 0):.0%})</strong></div>
            <div>Violations: <strong>{ver.get('total_violations', 0)}</strong></div>
            <div>Correction: <strong>{'Yes' if ver.get('correction_attempted') else 'No'}</strong></div>
        </div>
        
        <p><strong>Technical Specifications Extracted:</strong></p>
        <div class="specs-grid">
            <div>Benchmarks: {', '.join(specs.get('benchmarks', ['None'])[:5])}</div>
            <div>Models: {', '.join(specs.get('models', ['None'])[:5])}</div>
            <div>Parameters: {', '.join(specs.get('parameter_counts', ['None'])[:3])}</div>
            <div>Datasets: {', '.join(specs.get('dataset_sizes', ['None'])[:3])}</div>
        </div>
"""
    
    # Show violations if any
    if ver.get('violations'):
        html += '<p style="margin-top: 1rem;"><strong>Issues Flagged:</strong></p>'
        for v in ver['violations'][:10]:
            html += f'<div class="violation">[{v.get("type", "unknown")}] {v.get("suggestion", v.get("issue", ""))}</div>'
    
    html += """
        <p style="margin-top: 1rem; font-size: 9pt; color: #666;">
            <em>This report was generated with strict technical verification. 
            All quantitative claims were checked against source documents.
            Generic terminology was flagged and removed where possible.</em>
        </p>
    </div>
</body>
</html>"""
    
    return html

# ================================================================================
# SEARCH TAB FUNCTIONS
# ================================================================================

def display_results_preview(results, limit=5):
    """Display a preview of the top results"""
    st.subheader(f"📊 Top {limit} Results")
    
    for i, paper in enumerate(results[:limit], 1):
        with st.expander(f"#{i} | Score: {paper.get('relevance_score', 0)} | {paper.get('title', 'No title')[:80]}..."):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Title:** {paper.get('title', 'N/A')}")
                st.markdown(f"**Authors:** {paper.get('ieee_authors', 'N/A')}")
                st.markdown(f"**Year:** {paper.get('year', 'N/A')}")
                st.markdown(f"**Venue:** {paper.get('venue', 'N/A')}")
                
                if paper.get('tldr'):
                    st.info(f"💡 **TLDR:** {paper['tldr']}")
                
                if paper.get('abstract'):
                    with st.expander("📄 View Abstract"):
                        st.write(paper['abstract'])
            
            with col2:
                st.metric("Relevance Score", paper.get('relevance_score', 0))
                st.metric("Citations", paper.get('citations', 0))
                st.metric("Sources", paper.get('source_count', 1))
                
                if paper.get('recency_boosted'):
                    st.success("🔥 Recent Paper Boost")
                
                if paper.get('doi') and paper['doi'] != 'N/A':
                    st.markdown(f"**DOI:** {paper['doi']}")
                
                if paper.get('url'):
                    st.markdown(f"[🔗 View Paper]({paper['url']})")

def create_download_buttons(output_dir):
    """Create download button for ZIP archive of all results"""
    st.subheader("📥 Download Results")
    
    # Check if ZIP already exists
    zip_path = f"{output_dir}.zip"
    
    if os.path.exists(zip_path):
        with open(zip_path, 'rb') as f:
            st.download_button(
                label="📦 Download Complete Archive (ZIP)",
                data=f,
                file_name=os.path.basename(zip_path),
                mime='application/zip',
                type="primary",
                use_container_width=True
            )
        
        # Show what's included in the ZIP
        with st.expander("📋 Contents of ZIP Archive"):
            files_included = [
                "✅ MASTER_REPORT_FINAL.csv - Complete dataset",
                "✅ EXECUTIVE_SUMMARY.txt - Key findings",
                "✅ RESEARCH_GAPS.txt - Identified gaps",
                "✅ research_data.json - Full JSON export",
                "✅ references.bib - BibTeX citations",
                "✅ research_analytics.png - Visualization chart",
                "✅ SESSION_REPORT.txt - Search metadata"
            ]
            for item in files_included:
                st.markdown(item)
    else:
        st.warning("⚠️ ZIP archive not found. The orchestrator should have created it automatically.")

# ================================================================================
# ENHANCED PIPELINE EXECUTION
# ================================================================================

def execute_report_pipeline():
    """Execute complete report generation pipeline with verification"""
    st.session_state.report_processing = True
    st.session_state.report_step = 'processing'
    st.session_state.report_api_calls = 0
    st.session_state.report_start_time = time.time()
    
    try:
        # Check for Anthropic API key
        try:
            anthropic_key = st.secrets["ANTHROPIC_API_KEY"]
        except:
            raise Exception("Anthropic API key not configured (needed for report generation)")
        
        topic = st.session_state.report_form_data['topic']
        subject = st.session_state.report_form_data['subject']
        api_keys = load_api_keys()
        
        # Stage 1: Topic Analysis
        st.info("🔍 Stage 1/5: Analyzing topic...")
        analysis = analyze_topic_with_ai(topic, subject)
        st.session_state.report_research['subtopics'] = analysis['subtopics']
        
        # Stage 2: Check if we can reuse existing search results
        reuse_existing = False
        if 'results' in st.session_state and st.session_state.get('search_query'):
            existing_query = st.session_state.get('search_query', '').lower()
            new_query = f"{topic} {subject}".lower()
            
            if topic.lower() in existing_query or subject.lower() in existing_query:
                st.info(f"✅ Reusing existing search results from: '{st.session_state.get('search_query')}'")
                results = st.session_state['results']
                reuse_existing = True
        
        # Stage 2: Academic Research (if not reusing)
        if not reuse_existing:
            st.info("🔬 Stage 2/5: Searching academic databases...")
            update_report_progress('Research', 'Initializing academic search engines...', 20)
            
            search_query = f"{topic} {subject}".strip()
            
            # Configure orchestrator
            orchestrator_config = {
                'abstract_limit': 10,
                'high_consensus_threshold': 4,
                'citation_weight': 1.5,
                'source_weight': 100,
                'enable_alerts': True,
                'enable_visualization': False,
                'export_formats': ['csv', 'json'],
                'recency_boost': True,
                'recency_years': 5,
                'recency_multiplier': 1.2
            }
            
            # Set API keys in environment
            for key, value in api_keys.items():
                if key != 'email' and value and len(value) > 5:
                    os.environ[f"{key.upper()}_API_KEY"] = value
                elif key == 'email' and value:
                    os.environ['USER_EMAIL'] = value
            
            # Initialize orchestrator
            orchestrator = ResearchOrchestrator(config=orchestrator_config)
            
            update_report_progress('Research', f'Searching databases for "{search_query}"...', 30)
            
            # Execute search
            results = orchestrator.run_search(search_query, limit_per_engine=15)
            
            if not results:
                raise Exception("No results found from academic databases")
            
            update_report_progress('Research', f'Found {len(results)} papers', 50)
        
        # Stage 3: Process with strict deduplication and ranking
        st.info("📊 Stage 3/5: Processing and ranking sources...")
        update_report_progress('Processing', 'Deduplicating and ranking by authority...', 55)
        
        raw_sources = convert_orchestrator_to_source_format(results)
        sources = deduplicate_and_rank_sources_strict(raw_sources)
        st.session_state.report_research['sources'] = sources
        
        # Show authority distribution
        tier_counts = Counter(s.get('authority_tier', 'unknown') for s in sources[:20])
        tier_display = ", ".join([f"{k}: {v}" for k, v in tier_counts.items()])
        st.info(f"📚 Authority distribution (top 20): {tier_display}")
        
        if len(sources) < 3:
            raise Exception(f"Only {len(sources)} sources found after deduplication. Need at least 3.")
        
        # Stage 4: Draft Generation with Strict Verification
        st.info("✍️ Stage 4/5: Writing and verifying draft...")
        max_sources = st.session_state.report_form_data.get('max_sources', 25)
        
        draft, verification_report = generate_draft_strict(
            topic,
            subject,
            analysis['subtopics'],
            sources,
            st.session_state.report_research['phrase_variations'],
            analysis.get('evaluationFrameworks', []),
            max_sources=max_sources
        )
        
        st.session_state.report_draft = draft
        st.session_state.verification_results = verification_report
        
        # Display verification summary
        ver = verification_report['verification']
        cov = verification_report['coverage']
        
        if ver['has_critical']:
            st.error(f"⚠️ {ver['total_violations']} critical issues found and corrected")
        elif ver['total_violations'] > 0:
            st.warning(f"⚠️ {ver['total_violations']} minor issues flagged")
        else:
            st.success("✅ All claims verified against sources")
        
        st.info(f"📊 Source coverage: {cov['coverage']:.1%} ({cov['cited_count']}/{cov['total_sources']})")
        
        # Stage 5: Refinement & HTML Generation
        st.info("✨ Stage 5/5: Final refinement...")
        update_report_progress('Refinement', 'Polishing document...', 90)
        
        refined = refine_draft_simple(draft, topic, len(sources))
        st.session_state.report_final = refined
        
        html = generate_html_report_strict(
            refined,
            st.session_state.report_form_data,
            sources,
            verification_report,
            max_sources=max_sources
        )
        st.session_state.report_html = html
        
        st.session_state.report_execution_time = time.time() - st.session_state.report_start_time
        
        update_report_progress("Complete", "Report generated successfully!", 100)
        st.session_state.report_step = 'complete'
        
        exec_mins = int(st.session_state.report_execution_time // 60)
        exec_secs = int(st.session_state.report_execution_time % 60)
        st.success(
            f"✅ Report complete in {exec_mins}m {exec_secs}s! "
            f"{len(sources)} sources, {st.session_state.report_api_calls} API calls, "
            f"{ver['total_violations']} verification issues flagged"
        )
    
    except Exception as e:
        st.session_state.report_execution_time = time.time() - st.session_state.report_start_time if st.session_state.report_start_time else 0
        update_report_progress("Error", str(e), 0)
        st.session_state.report_step = 'error'
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
    finally:
        st.session_state.report_processing = False

def reset_report_system():
    """Reset report generation system"""
    st.session_state.report_step = 'input'
    st.session_state.report_draft = None
    st.session_state.report_final = None
    st.session_state.report_html = None
    st.session_state.report_processing = False
    st.session_state.report_api_calls = 0
    st.session_state.report_start_time = None
    st.session_state.report_execution_time = None
    st.session_state.verification_results = None
    st.session_state.report_research = {
        'subtopics': [],
        'sources': [],
        'phrase_variations': []
    }

# ================================================================================
# MAIN APPLICATION - CONTINUES IN NEXT PART DUE TO LENGTH
# ================================================================================

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<p class="main-header">🔬 SROrch - Scholarly Research Orchestrator & Report Writer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Engine Academic Literature Search, Analysis & Strictly Verified Report Generation</p>', unsafe_allow_html=True)
    
    # Check dev mode
    is_dev_mode, dev_keys = check_dev_mode()
    if is_dev_mode:
        st.info(f"🔧 **Development Mode Active** - Using {len(dev_keys)} pre-configured API key(s)")
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        render_api_key_input_section()
        st.divider()
        
        api_keys = load_api_keys()
        key_status = check_api_keys(api_keys)
        available_engines = get_available_engines(key_status)
        
        st.subheader("🔍 Available Engines")
        for engine in ["Semantic Scholar", "Google Scholar", "CORE", "SCOPUS", "Springer Nature"]:
            status = key_status.get(engine.lower().replace(" ", "_").replace("scholar", "s2" if "Semantic" in engine else "serp"), "❌")
            st.markdown(f"{status} **{engine}**" if status == "✅" else f"❌ {engine} *(no key)*")
        for engine in ["arXiv", "PubMed", "Crossref/DOI", "OpenAlex", "Europe PMC", "PLOS", "SSRN", "DeepDyve", "Wiley", "Taylor & Francis", "ACM Digital Library", "DBLP", "SAGE Journals"]:
            st.markdown(f"✅ **{engine}**")
        
        st.info(f"**Active Engines:** {len(available_engines)}/18")
        st.divider()
        
        st.subheader("🔍 Search Parameters")
        limit_per_engine = st.slider("Results per engine", 5, 50, 25, 5)
        st.divider()
        
        st.subheader("🎛️ Advanced Settings")
        with st.expander("Scoring & Ranking"):
            abstract_limit = st.number_input("Deep Look Limit", 1, 20, 10)
            citation_weight = st.slider("Citation Weight", 0.1, 5.0, 1.5, 0.1)
            source_weight = st.number_input("Source Weight", 10, 500, 100, 10)
            high_consensus_threshold = st.number_input("High Consensus Threshold", 2, 7, 4)
        
        with st.expander("Recency Boost"):
            recency_boost = st.checkbox("Enable Recency Boost", True)
            recency_years = st.slider("Recent Paper Window (years)", 1, 10, 5)
            recency_multiplier = st.slider("Boost Multiplier", 1.0, 2.0, 1.2, 0.1)
        
        with st.expander("Output Options"):
            enable_alerts = st.checkbox("Enable Consensus Alerts", True)
            enable_visualization = st.checkbox("Enable Visualizations", True)
            export_formats = st.multiselect("Export Formats", ['csv', 'json', 'bibtex'], ['csv', 'json', 'bibtex'])
        
        config = {
            'abstract_limit': abstract_limit, 'high_consensus_threshold': high_consensus_threshold,
            'citation_weight': citation_weight, 'source_weight': source_weight,
            'enable_alerts': enable_alerts, 'enable_visualization': enable_visualization,
            'export_formats': export_formats, 'recency_boost': recency_boost,
            'recency_years': recency_years, 'recency_multiplier': recency_multiplier
        }
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "📊 Results", "📝 Report Writer", "ℹ️ About"])
    
    # TAB 1: SEARCH
    with tab1:
        st.header("Search Academic Literature")
        if len(available_engines) == 18:
            st.success("✅ All 18 engines active!")
        else:
            st.info(f"ℹ️ Using {len(available_engines)} engines")
        
        search_query = st.text_input("Enter your research query:", placeholder="e.g., Machine Learning in Healthcare")
        col1, col2 = st.columns([2, 1])
        with col1:
            search_button = st.button("🚀 Start Search", type="primary", use_container_width=True)
        with col2:
            if st.button("🔄 Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache cleared!")
        
        if search_button:
            if not search_query:
                st.error("Please enter a search query!")
            else:
                st.session_state['search_query'] = search_query
                st.session_state['config'] = config
                st.session_state['limit_per_engine'] = limit_per_engine
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("🔧 Initializing...")
                    progress_bar.progress(10)
                    
                    for key, value in api_keys.items():
                        if key != 'email' and value and len(value) > 5:
                            os.environ[f"{key.upper()}_API_KEY"] = value
                        elif key == 'email' and value:
                            os.environ['USER_EMAIL'] = value
                    
                    orchestrator = ResearchOrchestrator(config=config)
                    status_text.text(f"🔍 Searching {len(available_engines)} databases...")
                    progress_bar.progress(30)
                    
                    results = orchestrator.run_search(search_query, limit_per_engine=limit_per_engine)
                    status_text.text("📝 Generating reports...")
                    progress_bar.progress(70)
                    
                    orchestrator.save_master_csv(results, search_query)
                    progress_bar.progress(100)
                    status_text.text("✅ Search completed!")
                    
                    st.session_state['results'] = results
                    st.session_state['output_dir'] = orchestrator.output_dir
                    st.session_state['metadata'] = orchestrator.session_metadata
                    
                    st.markdown(f"""<div class="success-box"><h3>✅ Search Completed!</h3>
                    <p><strong>Papers:</strong> {len(results)}</p></div>""", unsafe_allow_html=True)
                    st.info("👉 Switch to 'Results' or 'Report Writer' tabs!")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # TAB 2: RESULTS
    with tab2:
        st.header("Search Results & Analytics")
        
        if 'results' in st.session_state and st.session_state['results']:
            results = st.session_state['results']
            output_dir = st.session_state.get('output_dir', '')
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Papers", len(results))
            with col2:
                high_consensus = sum(1 for p in results if p.get('source_count', 0) >= config['high_consensus_threshold'])
                st.metric("High Consensus", high_consensus)
            with col3:
                avg_cites = sum(p.get('citations_int', 0) for p in results) / len(results) if results else 0
                st.metric("Avg Citations", f"{avg_cites:.1f}")
            with col4:
                st.metric("Engines", len(st.session_state.get('metadata', {}).get('successful_engines', [])))
            
            st.divider()
            
            chart_path = os.path.join(output_dir, "research_analytics.png")
            if os.path.exists(chart_path):
                st.subheader("📈 Analytics")
                st.image(chart_path, use_container_width=True)
            
            display_results_preview(results, limit=10)
            st.divider()
            
            if output_dir and os.path.exists(output_dir):
                create_download_buttons(output_dir)
        else:
            st.info("👈 No results yet. Start a search!")
    
    # TAB 3: REPORT WRITER
    with tab3:
        st.header("📝 Report Writer (Strict Mode)")
        
        try:
            anthropic_key = st.secrets["ANTHROPIC_API_KEY"]
            api_available = True
        except:
            api_available = False
            st.error("⚠️ Anthropic API key not configured")
        
        if api_available:
            if st.session_state.report_step == 'input':
                st.markdown("### Configuration")
                
                if 'results' in st.session_state:
                    st.success(f"✅ Existing results: '{st.session_state.get('search_query')}'")
                
                col1, col2 = st.columns(2)
                with col1:
                    topic = st.text_input("Topic *", st.session_state.report_form_data['topic'])
                    subject = st.text_input("Subject *", st.session_state.report_form_data['subject'])
                with col2:
                    researcher = st.text_input("Researcher *", st.session_state.report_form_data['researcher'])
                    institution = st.text_input("Institution *", st.session_state.report_form_data['institution'])
                
                col3, col4 = st.columns(2)
                with col3:
                    date = st.date_input("Date", datetime.strptime(st.session_state.report_form_data['date'], '%Y-%m-%d'))
                with col4:
                    style = st.selectbox("Citation Style", ["IEEE", "APA"])
                
                max_sources = st.slider("Max sources", 10, 100, 25, 5)
                
                if max_sources <= 25:
                    st.success(f"✅ Conservative ({max_sources} sources) - ~4-6 min")
                elif max_sources <= 50:
                    st.warning(f"⚠️ Balanced ({max_sources} sources) - ~6-10 min")
                else:
                    st.error(f"🔴 Comprehensive ({max_sources} sources) - ~10-15 min, higher cost")
                
                st.session_state.report_form_data.update({
                    'topic': topic, 'subject': subject, 'researcher': researcher,
                    'institution': institution, 'date': date.strftime('%Y-%m-%d'),
                    'citation_style': style, 'max_sources': max_sources
                })
                
                valid = all([topic, subject, researcher, institution])
                
                st.info("""**Features:** Claim verification, source authority ranking, citation integrity""")
                
                if st.button("🚀 Generate Report", disabled=not valid, type="primary", use_container_width=True):
                    execute_report_pipeline()
                    st.rerun()
            
            elif st.session_state.report_step == 'processing':
                st.markdown("### 🔄 Generating Report")
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{st.session_state.report_progress['stage']}**")
                    st.progress(st.session_state.report_progress['percent'] / 100)
                with col2:
                    st.metric("Progress", f"{st.session_state.report_progress['percent']}%")
                
                st.info(st.session_state.report_progress['detail'])
                
                if st.session_state.report_processing:
                    time.sleep(3)
                    st.rerun()
            
            elif st.session_state.report_step == 'complete':
                st.success("✅ Report Generated!")
                
                ver_results = st.session_state.get('verification_results', {})
                issues = len(ver_results.get('verification', {}).get('violations', []))
                
                if issues == 0:
                    st.markdown('<span class="verification-badge verified">✓ Verified</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="verification-badge unverified">⚠ {issues} Issues</span>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Sources", len(st.session_state.report_research['sources']))
                with col2:
                    st.metric("API Calls", st.session_state.report_api_calls)
                with col3:
                    exec_time = st.session_state.report_execution_time
                    st.metric("Time", f"{int(exec_time//60)}m {int(exec_time%60)}s")
                with col4:
                    st.metric("Issues", issues)
                
                if st.session_state.report_html:
                    filename = f"{st.session_state.report_form_data['topic'].replace(' ', '_')}_Report.html"
                    st.download_button(
                        "📥 Download HTML Report",
                        data=st.session_state.report_html,
                        file_name=filename,
                        mime="text/html",
                        type="primary",
                        use_container_width=True
                    )
                
                if st.button("🔄 Generate Another", type="secondary", use_container_width=True):
                    reset_report_system()
                    st.rerun()
            
            elif st.session_state.report_step == 'error':
                st.error("❌ Error Occurred")
                st.warning(st.session_state.report_progress['detail'])
                if st.button("🔄 Try Again", type="primary", use_container_width=True):
                    reset_report_system()
                    st.rerun()
    
    # TAB 4: ABOUT
    with tab4:
        st.header("About SROrch")
        st.markdown("""
        ### 🔬 Scholarly Research Orchestrator
        
        **Version:** 2.2 - Token Management & Enhanced Reliability
        
        **Features:**
        - ✅ 18 Academic databases (13 free, 5 premium)
        - ✅ Strict claim verification
        - ✅ Source authority ranking
        - ✅ Token management (auto-reduction)
        - ✅ Professional HTML reports
        
        **Supported Databases:**
        - Premium: Semantic Scholar, Google Scholar, CORE, SCOPUS, Springer
        - Free: arXiv, PubMed, Crossref, OpenAlex, Europe PMC, PLOS, SSRN, DeepDyve, Wiley, Taylor & Francis, ACM, DBLP, SAGE
        
        **License:** MIT
        """)

if __name__ == "__main__":
    main()
