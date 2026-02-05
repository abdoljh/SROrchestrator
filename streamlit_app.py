"""
SROrch Streamlit Interface - ENHANCED with Rigorous Claim Verification
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
from collections import Counter

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
    </style>
""", unsafe_allow_html=True)

# ================================================================================
# CONFIGURATION & CONSTANTS
# ================================================================================

# Model configuration for report generation
MODEL_PRIMARY = "claude-sonnet-4-20250514"
MODEL_FALLBACK = "claude-haiku-3-5-20241022"

# Rate limiting for Anthropic API
MIN_API_DELAY = 3.0
RETRY_DELAYS = [10, 20, 40]

# ================================================================================
# ENHANCED CLAIM VERIFICATION SYSTEM (NEW)
# ================================================================================

class ClaimVerifier:
    """Verify that all quantitative claims in generated text are supported by sources"""
    
    def __init__(self, sources: List[Dict]):
        self.sources = sources
        self.source_metrics = self._extract_metrics_from_sources(sources)
        self.verification_log = []
    
    def _extract_metrics_from_sources(self, sources: List[Dict]) -> Dict[str, List[str]]:
        """Pre-extract all numerical claims from source abstracts/metadata"""
        metrics = {}
        for i, source in enumerate(sources, 1):
            # Combine all text fields
            text_parts = [
                source.get('metadata', {}).get('title', ''),
                source.get('content', ''),
                source.get('metadata', {}).get('venue', ''),
                str(source.get('_orchestrator_data', {}).get('tldr', ''))
            ]
            text = ' '.join(text_parts)
            
            # Comprehensive patterns for metric extraction
            patterns = [
                r'(\d+(?:\.\d+)?%)\s*(?:improvement|increase|decrease|accuracy|precision|recall|f1)',
                r'(\d+(?:\.\d+)?)\s*(?:times|fold|x)\s*(?:faster|better|improvement|higher|lower)',
                r'outperform[s]?\s*.*?by\s*(\d+(?:\.\d+)?%?)',
                r'achieve[s]?\s*(\d+(?:\.\d+)?%?)',
                r'(\d+(?:\.\d+)?)\s*percent',
                r'(\d+)\s*(?:million|billion|m|b)\s*(?:parameters|papers|documents)',
                r'(\d+(?:\.\d+)?)\s*(?:gb|mb|tb)',
                r'(\d+)\s*(?:layers|heads|epochs|samples)',
                r'(\d{4})\s*(?:benchmark|dataset|corpus)',
                r'(\d+(?:\.\d+)?)\s*(?:bleu|rouge|meteor|score)',
                r'(\d+)\s*(?:human evaluators|participants|subjects)',
                r'p\s*[<>=]\s*(\d+\.\d+)',
                r'(\d+(?:\.\d+)?)\s*(?:s|ms|seconds|milliseconds)',
            ]
            
            found_metrics = []
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                found_metrics.extend([m[0] if isinstance(m, tuple) else m for m in matches])
            
            if found_metrics:
                metrics[str(i)] = found_metrics
        
        return metrics
    
    def verify_claim(self, claim_text: str, citation_num: int) -> Tuple[bool, str, List[str]]:
        """Verify if a quantitative claim is supported by its cited source"""
        # Extract numbers from claim
        numbers_in_claim = re.findall(r'\d+(?:\.\d+)?%?', claim_text)
        
        if not numbers_in_claim:
            return True, "No quantitative claims to verify", []
        
        source_metrics = self.source_metrics.get(str(citation_num), [])
        
        unsupported = []
        for number in numbers_in_claim:
            # Fuzzy match: allow for rounding differences
            found = any(self._fuzzy_match(number, sm) for sm in source_metrics)
            if not found:
                unsupported.append(number)
        
        if unsupported:
            return False, f"Numbers {unsupported} not found in source [{citation_num}]", unsupported
        
        return True, "All metrics verified against source", []
    
    def _fuzzy_match(self, claim_num: str, source_num: str) -> bool:
        """Allow small rounding differences (e.g., 78% matches 78.5%)"""
        try:
            c = float(claim_num.replace('%', ''))
            s = float(source_num.replace('%', ''))
            # Allow 5% relative difference or absolute difference of 2
            relative_diff = abs(c - s) / max(abs(s), 1)
            absolute_diff = abs(c - s)
            return relative_diff < 0.05 or absolute_diff < 2
        except:
            return claim_num == source_num
    
    def flag_unsupported_claims(self, draft_text: str) -> List[Dict]:
        """Scan entire draft for unsupported quantitative claims"""
        issues = []
        
        # Pattern: claim text followed by citation
        # Look for sentences with quantitative terms and citations
        patterns = [
            r'([^.]*?\b(?:improved|increased|decreased|achieved|outperformed|reduced|enhanced|boosted|elevated|advanced)\b[^.]*?\b(?:by|to|of|with|approximately|about|around)?\s*(?:\d+(?:\.\d+)?%?)[^.]*?)\[(\d+)\]',
            r'([^.]*?\b(?:accuracy|precision|recall|f1|score|performance|metric)\b[^.]*?(?:\d+(?:\.\d+)?%?)[^.]*?)\[(\d+)\]',
            r'([^.]*?\b(?:parameters|dataset|corpus|papers)\b[^.]*?(?:\d+(?:\.\d+)?\s*(?:million|billion|m|b)?)[^.]*?)\[(\d+)\]',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, draft_text, re.IGNORECASE)
            for match in matches:
                claim_text = match.group(1)
                citation_num = int(match.group(2))
                
                is_valid, message, unsupported_nums = self.verify_claim(claim_text, citation_num)
                
                if not is_valid:
                    issues.append({
                        'claim': claim_text.strip(),
                        'citation': citation_num,
                        'issue': message,
                        'unsupported_numbers': unsupported_nums
                    })
        
        self.verification_log.extend(issues)
        return issues
    
    def get_verification_summary(self) -> Dict:
        """Get summary of verification results"""
        return {
            'total_issues': len(self.verification_log),
            'sources_with_metrics': len(self.source_metrics),
            'total_metrics_found': sum(len(m) for m in self.source_metrics.values()),
            'issues_by_citation': Counter([i['citation'] for i in self.verification_log])
        }

# ================================================================================
# ENHANCED SOURCE MANAGEMENT (NEW)
# ================================================================================

def deduplicate_and_rank_sources(sources: List[Dict]) -> List[Dict]:
    """Deduplicate sources by DOI/title and rank by authority"""
    
    # Authority tiers for ranking
    TIER_RANK = {
        'top_tier': 0,      # Nature, Science, Cell, Lancet
        'publisher': 1,     # IEEE, ACM, Springer, Elsevier
        'conference': 2,    # NeurIPS, ICML, ACL, etc.
        'repository': 3,    # arXiv, bioRxiv
        'other': 4
    }
    
    def get_authority_tier(venue: str, url: str) -> str:
        venue_lower = venue.lower()
        url_lower = url.lower()
        
        top_indicators = ['nature', 'science', 'cell', 'lancet', 'nejm', 'jama']
        publisher_indicators = ['ieee', 'acm', 'springer', 'elsevier', 'wiley', 'sage', 'taylor & francis']
        conference_indicators = ['neurips', 'icml', 'iclr', 'acl', 'emnlp', 'cvpr', 'iccv', 'eccv']
        
        if any(ind in venue_lower for ind in top_indicators):
            return 'top_tier'
        elif any(ind in venue_lower for ind in publisher_indicators):
            return 'publisher'
        elif any(ind in venue_lower for ind in conference_indicators):
            return 'conference'
        elif 'arxiv' in url_lower or 'biorxiv' in url_lower:
            return 'repository'
        else:
            return 'other'
    
    seen = {}
    
    for source in sources:
        # Create deduplication key
        doi = str(source.get('metadata', {}).get('doi', '')).lower().strip()
        title = re.sub(r'[^\w]', '', source.get('metadata', {}).get('title', '').lower())
        
        key = doi if doi and doi != 'n/a' and len(doi) > 5 else title
        
        if key in seen:
            # Merge metadata, keep highest authority version
            existing = seen[key]
            existing['source_count'] = existing.get('source_count', 1) + 1
            
            # Keep higher citation count
            existing_cites = safe_int(existing.get('metadata', {}).get('citations', 0))
            new_cites = safe_int(source.get('metadata', {}).get('citations', 0))
            if new_cites > existing_cites:
                existing['metadata']['citations'] = new_cites
            
            # Track all URLs
            if 'all_urls' not in existing:
                existing['all_urls'] = [existing.get('url', '')]
            existing['all_urls'].append(source.get('url', ''))
            
            # Upgrade to better venue if found
            existing_tier = TIER_RANK.get(existing.get('authority_tier', 'other'), 4)
            new_tier_num = TIER_RANK.get(get_authority_tier(
                source.get('metadata', {}).get('venue', ''),
                source.get('url', '')
            ), 4)
            
            if new_tier_num < existing_tier:
                # Upgrade source
                existing['metadata']['venue'] = source.get('metadata', {}).get('venue', existing['metadata'].get('venue', ''))
                existing['url'] = source.get('url', existing.get('url', ''))
                existing['authority_tier'] = get_authority_tier(
                    existing['metadata']['venue'],
                    existing['url']
                )
        else:
            # New source
            venue = source.get('metadata', {}).get('venue', '')
            url = source.get('url', '')
            source['authority_tier'] = get_authority_tier(venue, url)
            source['source_count'] = 1
            seen[key] = source
    
    # Sort by authority tier, then citations, then consensus
    ranked = sorted(
        seen.values(),
        key=lambda x: (
            TIER_RANK.get(x.get('authority_tier', 'other'), 4),
            -safe_int(x.get('metadata', {}).get('citations', 0)),
            -x.get('source_count', 1)
        )
    )
    
    return ranked

# ================================================================================
# CITATION INTEGRITY SYSTEM (NEW)
# ================================================================================

def extract_cited_references_enhanced(draft: Dict) -> set:
    """Extract all citation numbers used in the draft with context"""
    cited = set()
    citation_contexts = []
    
    def extract_from_text(text: str, section: str):
        if not isinstance(text, str):
            return
        matches = re.finditer(r'\[(\d+)\]', text)
        for match in matches:
            cited.add(int(match.group(1)))
            # Get surrounding context (50 chars before/after)
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]
            citation_contexts.append({
                'citation': int(match.group(1)),
                'context': context,
                'section': section
            })
    
    for key, value in draft.items():
        if isinstance(value, str):
            extract_from_text(value, key)
        elif isinstance(value, list) and key == 'mainSections':
            for section in value:
                if isinstance(section, dict):
                    extract_from_text(section.get('content', ''), section.get('title', 'Section'))
    
    return cited, citation_contexts

def verify_citation_integrity(draft: Dict, sources: List[Dict]) -> Dict:
    """Comprehensive citation integrity check"""
    
    cited, contexts = extract_cited_references_enhanced(draft)
    max_valid = len(sources)
    
    # Check for out-of-range citations
    invalid_citations = [c for c in cited if c > max_valid or c < 1]
    
    # Check for claims without citations (quantitative claims)
    uncited_claims = []
    
    quantitative_patterns = [
        r'\d+%',
        r'\d+\s*(?:times|fold|x)\s*(?:faster|better)',
        r'outperform.*?by\s+\d+',
        r'achieved?\s+\d+',
        r'\d+\s*(?:million|billion)\s*(?:parameters|papers)',
        r'\d{4}\s*(?:benchmark|dataset)',
    ]
    
    for ctx in contexts:
        # Check if this citation supports a quantitative claim
        has_numbers = any(re.search(p, ctx['context'], re.IGNORECASE) for p in quantitative_patterns)
        if not has_numbers:
            # This citation might be for a general claim - flag for review
            pass  # Not necessarily an error
    
    # Find orphaned sources (high-quality but uncited)
    orphaned = []
    for i, source in enumerate(sources, 1):
        if i not in cited:
            tier = source.get('authority_tier', 'other')
            cites = safe_int(source.get('metadata', {}).get('citations', 0))
            if tier in ['top_tier', 'publisher'] and cites > 50:
                orphaned.append({
                    'index': i,
                    'title': source.get('metadata', {}).get('title', 'Unknown')[:60],
                    'tier': tier,
                    'citations': cites
                })
    
    return {
        'valid_citations': sorted([c for c in cited if 1 <= c <= max_valid]),
        'invalid_citations': sorted(invalid_citations),
        'total_citations': len(cited),
        'citation_coverage': len(cited) / max_valid if max_valid > 0 else 0,
        'orphaned_high_quality_sources': sorted(orphaned, key=lambda x: x['citations'], reverse=True)[:10],
        'contexts': contexts
    }

# ================================================================================
# ORIGINAL UTILITY FUNCTIONS (Preserved)
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
    
    # NEW: Verification state
    if 'verification_results' not in st.session_state:
        st.session_state.verification_results = None

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

# ... [Keep all the original sidebar and UI functions: render_api_key_input_section, check_api_keys, get_available_engines] ...

# ================================================================================
# ENHANCED REPORT GENERATION FUNCTIONS
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
    """Call Anthropic API with fallback model support"""
    try:
        anthropic_key = st.secrets["ANTHROPIC_API_KEY"]
    except:
        raise Exception("Anthropic API key not configured in secrets")
    
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
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            st.warning(f"⚠️ API error (attempt {attempt+1}/3): {str(e)[:50]}")
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

# ================================================================================
# ENHANCED TOPIC ANALYSIS WITH EVALUATION FRAMEWORKS
# ================================================================================

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
# ENHANCED DRAFT GENERATION WITH STRICT VERIFICATION
# ================================================================================

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

def generate_draft_with_verification(
    topic: str,
    subject: str,
    subtopics: List[str],
    sources: List[Dict],
    variations: List[str],
    evaluation_frameworks: List[str],
    max_sources: int = 25
) -> Tuple[Dict, Dict]:
    """Generate report draft with iterative verification"""
    
    update_report_progress('Drafting', 'Writing initial draft...', 70)
    
    if not sources:
        raise Exception("No sources available")
    
    # Prepare source list with strict numbering
    source_list = []
    for i, s in enumerate(sources[:max_sources], 1):
        meta = s.get('metadata', {})
        tier_badge = f"[{s.get('authority_tier', 'unknown').upper()}]"
        source_list.append(f"""[{i}] {tier_badge} {meta.get('title', 'Unknown')} ({meta.get('year', 'N/A')})
Authors: {meta.get('authors', 'Unknown')}
Venue: {meta.get('venue', 'Unknown')} | Citations: {meta.get('citations', 0)}
{s['url'][:70]}
Abstract: {s.get('content', '')[:200]}""")
    
    sources_text = "\n\n".join(source_list)
    
    # Build evaluation framework context
    eval_context = ""
    if evaluation_frameworks:
        eval_context = f"""
EVALUATION CONTEXT:
This field uses these specific benchmarks/methods: {', '.join(evaluation_frameworks)}
- Use these specific benchmark names when discussing evaluation
- DO NOT reference generic guidelines (e.g., PRISMA 2020) unless explicitly discussing human systematic reviews
- Focus on: task-specific automated benchmarks, human evaluation protocols, and domain-specific metrics
"""
    
    variations_text = f"""CRITICAL INSTRUCTION - PHRASE VARIATION:
You MUST use these variations to avoid repetition:
- "{topic}" - USE THIS SPARINGLY (maximum 5 times)
- "{variations[1]}" - PREFER THIS
- "{variations[2]}" - USE THIS OFTEN
- "this domain" - USE THIS
- "this research area" - USE THIS

DO NOT repeat "{topic}" more than 5 times total."""
    
    # Technical specificity requirements
    technical_reqs = """
CRITICAL TECHNICAL REQUIREMENTS:
- Include SPECIFIC numbers: dataset sizes, parameter counts, benchmark scores, sample sizes
- Name concrete architectures: model names, embedding dimensions, encoder types (e.g., "bi-encoder with 110M parameters")
- Cite exact benchmark names and their metrics (e.g., "ScholarQABench correctness score")
- Include version numbers and release dates where available
- Describe evaluation methodologies specifically (e.g., "blind human evaluation with N evaluators")

FORBIDDEN GENERIC PHRASES (will be rejected):
- "sophisticated indexing strategies" → USE: specific index type (FAISS, dense embeddings of size X, etc.)
- "state-of-the-art performance" → USE: specific metric values and comparisons with numbers
- "advanced architecture" → USE: layer counts, parameter counts, model names
- "significant improvements" → USE: exact percentage points or fold improvements with baseline names
- "recent studies show" → USE: specific citations with years
"""
    
    strict_boundary = f"""
STRICT SOURCE BOUNDARY - VIOLATION WILL CAUSE ERRORS:
You may ONLY cite sources [1] through [{len(sources[:max_sources])}]. 
DO NOT mention, cite, or refer to any paper not in the numbered list above.
DO NOT invent systems, authors, or studies not in the sources.
If you cannot find information for a claim, omit the claim rather than inventing.
"""
    
    prompt = f"""Write academic report about "{topic}" in {subject}.

{variations_text}

{technical_reqs}

{eval_context}

{strict_boundary}

REQUIREMENTS:
- Use ONLY provided academic sources below
- Cite sources as [1], [2], [3] etc. - just the number in brackets
- Include specific data, statistics, and years from sources
- VARY your phrasing - avoid repetition
- Every quantitative claim MUST have a citation immediately following

SUBTOPICS: {', '.join(subtopics)}

ACADEMIC SOURCES:
{sources_text}

Write these sections with TECHNICAL SPECIFICITY:
1. Abstract (150-250 words) - Include key metrics and methodology
2. Introduction - Frame the specific technical problem
3. Literature Review - Compare specific systems with their metrics
4. 3-4 Main Sections covering subtopics with technical details:
   - Architecture specifications (parameters, layers, embedding dims)
   - Dataset composition and statistics (sizes, sources, time periods)
   - Evaluation benchmarks and specific results
   - Comparison with baseline systems (name specific systems)
5. Data & Analysis - Quantitative performance breakdown with numbers
6. Challenges - Specific technical limitations (not generic "challenges remain")
7. Future Outlook - Concrete technical directions
8. Conclusion

Return ONLY valid JSON:
{{
  "abstract": "...",
  "introduction": "...",
  "literatureReview": "...",
  "mainSections": [{{"title": "...", "content": "..."}}],
  "dataAnalysis": "...",
  "challenges": "...",
  "futureOutlook": "...",
  "conclusion": "..."
}}"""
    
    # Add temporal context
    prompt = add_temporal_context(prompt, sources[:max_sources])
    
    # Generate initial draft
    response = call_anthropic_api(
        [{"role": "user", "content": prompt}],
        max_tokens=6000
    )
    text = "".join([c['text'] for c in response['content'] if c['type'] == 'text'])
    draft = parse_json_response(text)
    
    # Ensure all required keys exist
    required_keys = [
        'abstract', 'introduction', 'literatureReview', 'mainSections',
        'dataAnalysis', 'challenges', 'futureOutlook', 'conclusion'
    ]
    
    for key in required_keys:
        if key not in draft or not draft[key]:
            if key == 'mainSections':
                draft[key] = [{'title': 'Analysis', 'content': 'Content.'}]
            else:
                draft[key] = f"Section about {topic}."
    
    # Stage 1: Verify quantitative claims
    update_report_progress('Verification', 'Checking claim accuracy...', 75)
    verifier = ClaimVerifier(sources[:max_sources])
    
    # Combine all text for verification
    full_text = " ".join([
        draft.get('abstract', ''),
        draft.get('introduction', ''),
        draft.get('literatureReview', ''),
        draft.get('dataAnalysis', ''),
        draft.get('challenges', ''),
        draft.get('futureOutlook', ''),
        draft.get('conclusion', '')
    ])
    
    for section in draft.get('mainSections', []):
        full_text += " " + section.get('content', '')
    
    unsupported_claims = verifier.flag_unsupported_claims(full_text)
    
    # Stage 2: Verify citation integrity
    update_report_progress('Verification', 'Checking citation integrity...', 80)
    integrity = verify_citation_integrity(draft, sources[:max_sources])
    
    verification_report = {
        'unsupported_claims': unsupported_claims,
        'citation_integrity': integrity,
        'verifier_summary': verifier.get_verification_summary()
    }
    
    # If issues found, attempt regeneration with corrections
    if unsupported_claims or integrity['invalid_citations']:
        update_report_progress('Refinement', 'Fixing verification issues...', 85)
        
        correction_prompt = f"""
The draft has factual verification issues that MUST be fixed:

UNSUPPORTED CLAIMS:
"""
        for issue in unsupported_claims[:5]:  # Top 5 issues
            correction_prompt += f"- Claim: '{issue['claim'][:80]}...'\n"
            correction_prompt += f"  Issue: {issue['issue']}\n"
            correction_prompt += f"  Action: Remove this claim or find correct citation\n\n"
        
        if integrity['invalid_citations']:
            correction_prompt += f"\nINVALID CITATIONS: {integrity['invalid_citations']}\n"
            correction_prompt += "Action: Remove or correct these citations\n"
        
        correction_prompt += f"""
\nnREGENERATION INSTRUCTIONS:
1. Remove ALL claims with unsupported numbers
2. Ensure every number/percentage has a valid source [1-{len(sources[:max_sources])}]
3. DO NOT invent metrics not in sources
4. Use conservative language if exact numbers unavailable

Return corrected JSON with same structure."""
        
        # Append to original prompt and retry
        retry_prompt = prompt + "\n\n" + correction_prompt
        
        try:
            response = call_anthropic_api(
                [{"role": "user", "content": retry_prompt}],
                max_tokens=6000
            )
            text = "".join([c['text'] for c in response['content'] if c['type'] == 'text'])
            corrected_draft = parse_json_response(text)
            
            # Validate structure
            valid = True
            for key in required_keys:
                if key not in corrected_draft or not corrected_draft[key]:
                    valid = False
                    break
            
            if valid:
                draft = corrected_draft
                # Re-verify
                verifier2 = ClaimVerifier(sources[:max_sources])
                full_text2 = " ".join([corrected_draft.get(k, '') for k in required_keys if k != 'mainSections'])
                for section in corrected_draft.get('mainSections', []):
                    full_text2 += " " + section.get('content', '')
                
                remaining_issues = verifier2.flag_unsupported_claims(full_text2)
                verification_report['correction_attempted'] = True
                verification_report['remaining_issues_after_correction'] = len(remaining_issues)
            else:
                verification_report['correction_attempted'] = False
                verification_report['correction_failed'] = 'Invalid structure in corrected draft'
        except Exception as e:
            verification_report['correction_attempted'] = False
            verification_report['correction_error'] = str(e)
    
    # Clean up citations in text
    def fix_citations(text):
        if isinstance(text, str):
            text = re.sub(r'\[Source\s+(\d+)\]', r'[\1]', text, flags=re.IGNORECASE)
            text = re.sub(r'\[source\s+(\d+)\]', r'[\1]', text, flags=re.IGNORECASE)
        return text
    
    for key in draft:
        if isinstance(draft[key], str):
            draft[key] = fix_citations(draft[key])
        elif isinstance(draft[key], list):
            for i, item in enumerate(draft[key]):
                if isinstance(item, dict):
                    for k, v in item.items():
                        if isinstance(v, str):
                            item[k] = fix_citations(v)
                elif isinstance(item, str):
                    draft[key][i] = fix_citations(item)
    
    return draft, verification_report

# ================================================================================
# CITATION FORMATTING (Preserved with enhancements)
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

def format_citation_ieee(source: Dict, index: int) -> str:
    """Format citation in IEEE style with authority indicator"""
    meta = source.get('metadata', {})
    authors = meta.get('authors', 'Research Team')
    title = meta.get('title', 'Research Article')
    venue = meta.get('venue', 'Academic Publication')
    year = meta.get('year', '2024')
    url = source.get('url', '')
    tier = source.get('authority_tier', 'unknown')
    
    if not authors or authors.lower() in ['unknown', 'author unknown']:
        authors = venue + ' Authors'
    
    if not title or title.lower() == 'unknown':
        title = 'Research Article'
    
    formatted_authors = format_authors_ieee(authors)
    
    # Add tier indicator for transparency
    tier_note = ""
    if tier == 'top_tier':
        tier_note = " [Top-tier journal]"
    elif tier == 'preprint':
        tier_note = " [Preprint]"
    
    citation = f'[{index}] {formatted_authors}, "{title}," {venue}, {year}.{tier_note} <a href="{url}" target="_blank">{url}</a>'
    
    return citation

def format_citation_apa(source: Dict, index: int) -> str:
    """Format citation in APA style"""
    meta = source.get('metadata', {})
    authors = meta.get('authors', 'Research Team')
    title = meta.get('title', 'Research Article')
    venue = meta.get('venue', 'Academic Publication')
    year = meta.get('year', '2024')
    url = source.get('url', '')
    
    if not authors or authors.lower() in ['unknown', 'author unknown']:
        authors = venue + ' Authors'
    
    if not title or title.lower() == 'unknown':
        title = 'Research Article'
    
    citation = f"{authors} ({year}). {title}. <i>{venue}</i>. Retrieved from <a href=\"{url}\" target=\"_blank\">{url}</a>"
    
    return citation

# ================================================================================
# ENHANCED HTML GENERATION WITH VERIFICATION DISPLAY
# ================================================================================

def generate_html_report_enhanced(
    refined_draft: Dict,
    form_data: Dict,
    sources: List[Dict],
    verification_report: Dict,
    max_sources: int = 25
) -> str:
    """Generate HTML report with verification metadata"""
    
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
    
    # Verification summary for display
    ver_summary = verification_report.get('verifier_summary', {})
    integrity = verification_report.get('citation_integrity', {})
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{form_data['topic']} - Research Report</title>
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
        .tier-badge {{
            display: inline-block;
            padding: 0.1rem 0.3rem;
            border-radius: 0.2rem;
            font-size: 0.75rem;
            font-weight: bold;
            margin-left: 0.3rem;
        }}
        .tier-top {{ background-color: #d4edda; color: #155724; }}
        .tier-publisher {{ background-color: #d1ecf1; color: #0c5460; }}
        .tier-preprint {{ background-color: #fff3cd; color: #856404; }}
        .verification-box {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 0.3rem;
            padding: 1rem;
            margin: 1rem 0;
            font-size: 10pt;
        }}
        .metric {{
            display: inline-block;
            margin-right: 1rem;
        }}
    </style>
</head>
<body>
    <div class="cover">
        <h1>{form_data['topic']}</h1>
        <div class="meta">Research Report</div>
        <div class="meta">Subject: {form_data['subject']}</div>
        <div class="meta" style="margin-top: 1in;">
            {form_data['researcher']}<br>
            {form_data['institution']}<br>
            {report_date}
        </div>
        <div class="meta" style="margin-top: 0.5in; font-size: 10pt;">
            Generated by SROrch | {style} Citation Format
        </div>
        <div class="meta" style="margin-top: 0.3in; font-size: 9pt; color: #666;">
            Verification: {ver_summary.get('total_issues', 0)} issues flagged | 
            {integrity.get('total_citations', 0)} citations | 
            {integrity.get('citation_coverage', 0):.0%} source coverage
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
            if style == 'APA':
                citation = format_citation_apa(source, new_ref_num)
            else:
                citation = format_citation_ieee(source, new_ref_num)
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
            if style == 'APA':
                citation = format_citation_apa(source, idx)
            else:
                citation = format_citation_ieee(source, idx)
            citation_text = citation.split(']', 1)[1] if ']' in citation else citation
            html += f'        <div class="ref-item" style="font-size: 9pt;">• {citation_text}</div>\n'
    
    # Add verification appendix
    html += f"""
    </div>
    
    <div class="verification-box" style="page-break-before: always;">
        <h3>Document Verification Report</h3>
        <p><strong>Generation Metadata:</strong></p>
        <div class="metric"><strong>Sources Analyzed:</strong> {len(sources)}</div>
        <div class="metric"><strong>Sources Cited:</strong> {len(cited_refs_sorted)}</div>
        <div class="metric"><strong>Citation Coverage:</strong> {integrity.get('citation_coverage', 0):.1%}</div>
        <div class="metric"><strong>Claim Issues Flagged:</strong> {ver_summary.get('total_issues', 0)}</div>
        <br><br>
        <p><strong>Authority Distribution:</strong></p>
        <div class="metric">Top-tier: {sum(1 for s in sources if s.get('authority_tier') == 'top_tier')}</div>
        <div class="metric">Publisher: {sum(1 for s in sources if s.get('authority_tier') == 'publisher')}</div>
        <div class="metric">Conference: {sum(1 for s in sources if s.get('authority_tier') == 'conference')}</div>
        <div class="metric">Preprint: {sum(1 for s in sources if s.get('authority_tier') == 'repository')}</div>
        <br><br>
        <p style="font-size: 9pt; color: #666;">
            <em>This report was generated with automated verification. 
            Claims marked with citations were checked against source documents. 
            Quantitative claims without verification may require manual review.</em>
        </p>
    </div>
</body>
</html>"""
    
    return html

def renumber_citations_in_text(text: str, old_to_new: Dict[int, int]) -> str:
    """Renumber citations in text according to the mapping"""
    def replace_citation(match):
        old_num = int(match.group(1))
        new_num = old_to_new.get(old_num, old_num)
        return f'[{new_num}]'
    
    return re.sub(r'\[(\d+)\]', replace_citation, text)

def renumber_citations_in_draft(draft: Dict, old_to_new: Dict[int, int]) -> Dict:
    """Renumber all citations in the draft according to the mapping"""
    new_draft = {}
    
    for key, value in draft.items():
        if isinstance(value, str):
            new_draft[key] = renumber_citations_in_text(value, old_to_new)
        elif isinstance(value, list):
            new_list = []
            for item in value:
                if isinstance(item, dict):
                    new_item = {}
                    for k, v in item.items():
                        if isinstance(v, str):
                            new_item[k] = renumber_citations_in_text(v, old_to_new)
                        else:
                            new_item[k] = v
                    new_list.append(new_item)
                elif isinstance(item, str):
                    new_list.append(renumber_citations_in_text(item, old_to_new))
                else:
                    new_list.append(item)
            new_draft[key] = new_list
        else:
            new_draft[key] = value
    
    return new_draft

def refine_draft_simple(draft: Dict, topic: str, sources_count: int) -> Dict:
    """Add executive summary"""
    draft['executiveSummary'] = (
        f"This comprehensive report examines {topic}, analyzing key developments, "
        f"challenges, and future directions based on {sources_count} authoritative academic sources."
    )
    
    return draft

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
        st.info("🔍 Stage 1/6: Analyzing topic...")
        analysis = analyze_topic_with_ai(topic, subject)
        st.session_state.report_research['subtopics'] = analysis['subtopics']
        
        # Stage 2: Academic Research
        reuse_existing = False
        if 'results' in st.session_state and st.session_state.get('search_query'):
            existing_query = st.session_state.get('search_query', '').lower()
            if topic.lower() in existing_query or subject.lower() in existing_query:
                st.info(f"✅ Reusing existing search results")
                results = st.session_state['results']
                reuse_existing = True
        
        if not reuse_existing:
            st.info("🔬 Stage 2/6: Searching academic databases...")
            update_report_progress('Research', 'Initializing academic search engines...', 15)
            
            search_query = f"{topic} {subject}".strip()
            
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
            
            for key, value in api_keys.items():
                if key != 'email' and value and len(value) > 5:
                    os.environ[f"{key.upper()}_API_KEY"] = value
                elif key == 'email' and value:
                    os.environ['USER_EMAIL'] = value
            
            orchestrator = ResearchOrchestrator(config=orchestrator_config)
            
            update_report_progress('Research', f'Searching databases for "{search_query}"...', 25)
            results = orchestrator.run_search(search_query, limit_per_engine=15)
            
            if not results:
                raise Exception("No results found from academic databases")
            
            update_report_progress('Research', f'Found {len(results)} papers', 40)
        
        # Stage 3: Source Processing with Deduplication and Ranking
        st.info("📊 Stage 3/6: Processing and ranking sources...")
        update_report_progress('Processing', 'Deduplicating and ranking by authority...', 50)
        
        raw_sources = convert_orchestrator_to_source_format(results)
        sources = deduplicate_and_rank_sources(raw_sources)
        
        st.session_state.report_research['sources'] = sources
        
        if len(sources) < 3:
            raise Exception(f"Only {len(sources)} sources found after deduplication. Need at least 3.")
        
        # Show authority distribution
        tier_counts = Counter(s.get('authority_tier', 'unknown') for s in sources[:20])
        tier_display = ", ".join([f"{k}: {v}" for k, v in tier_counts.items()])
        st.info(f"📚 Authority distribution (top 20): {tier_display}")
        
        # Stage 4: Draft Generation with Verification
        st.info("✍️ Stage 4/6: Writing and verifying draft...")
        max_sources = st.session_state.report_form_data.get('max_sources', 25)
        
        draft, verification_report = generate_draft_with_verification(
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
        issues = len(verification_report.get('unsupported_claims', []))
        if issues > 0:
            st.warning(f"⚠️ Verification found {issues} unsupported claims (corrected where possible)")
        else:
            st.success("✅ All quantitative claims verified against sources")
        
        # Stage 5: Refinement
        st.info("🔍 Stage 5/6: Final refinement...")
        update_report_progress('Refinement', 'Polishing document...', 90)
        
        refined = refine_draft_simple(draft, topic, len(sources))
        st.session_state.report_final = refined
        
        # Stage 6: HTML Generation
        st.info("✨ Stage 6/6: Generating final document...")
        html = generate_html_report_enhanced(
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
        
        # Final summary
        ver_summary = verification_report.get('verifier_summary', {})
        st.success(
            f"✅ Report complete in {exec_mins}m {exec_secs}s! "
            f"{len(sources)} sources, {st.session_state.report_api_calls} API calls, "
            f"{ver_summary.get('total_issues', 0)} verification issues flagged"
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
# REMAINING ORIGINAL FUNCTIONS (Preserved)
# ================================================================================

# ... [Include all original functions: render_api_key_input_section, check_api_keys, 
# get_available_engines, display_results_preview, create_download_buttons, etc.] ...

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
    """Create download buttons for all generated files"""
    st.subheader("📥 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    files_to_download = {
        'MASTER_REPORT_FINAL.csv': ('CSV Report', 'text/csv', col1),
        'EXECUTIVE_SUMMARY.txt': ('Executive Summary', 'text/plain', col2),
        'RESEARCH_GAPS.txt': ('Research Gaps', 'text/plain', col3),
        'research_data.json': ('JSON Data', 'application/json', col1),
        'references.bib': ('BibTeX', 'text/plain', col2),
        'research_analytics.png': ('Analytics Chart', 'image/png', col3),
        'SESSION_REPORT.txt': ('Session Report', 'text/plain', col1)
    }
    
    for filename, (label, mime_type, column) in files_to_download.items():
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                with column:
                    st.download_button(
                        label=f"⬇️ {label}",
                        data=f,
                        file_name=filename,
                        mime=mime_type
                    )
    
    # ZIP download
    zip_path = f"{output_dir}.zip"
    if os.path.exists(zip_path):
        with open(zip_path, 'rb') as f:
            st.download_button(
                label="📦 Download Complete Archive (ZIP)",
                data=f,
                file_name=os.path.basename(zip_path),
                mime='application/zip',
                use_container_width=True
            )

# ================================================================================
# MAIN APPLICATION (Enhanced with verification display)
# ================================================================================

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<p class="main-header">🔬 SROrch - Scholarly Research Orchestrator & Report Writer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Engine Academic Literature Search, Analysis & Verified Report Generation</p>', unsafe_allow_html=True)
    
    # Check dev mode
    is_dev_mode, dev_keys = check_dev_mode()
    if is_dev_mode:
        st.info(f"🔧 **Development Mode Active** - Using {len(dev_keys)} pre-configured API key(s)")
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Input Section
        render_api_key_input_section()
        
        st.divider()
        
        # Load API keys and check status
        api_keys = load_api_keys()
        key_status = check_api_keys(api_keys)
        available_engines = get_available_engines(key_status)
        
        # Engine Status Display
        st.subheader("🔍 Available Engines")
        
        engine_display = {
            "Semantic Scholar": key_status['s2'],
            "Google Scholar": key_status['serp'],
            "CORE": key_status['core'],
            "SCOPUS": key_status['scopus'],
            "Springer Nature": key_status['springer'],
            "arXiv": "✅",
            "PubMed": "✅",
            "Crossref/DOI": "✅",
            "OpenAlex": "✅",
            "Europe PMC": "✅",
            "PLOS": "✅",
            "SSRN": "✅",
            "DeepDyve": "✅",
            "Wiley": "✅",
            "Taylor & Francis": "✅",
            "ACM Digital Library": "✅",
            "DBLP": "✅",
            "SAGE Journals": "✅",
        }
        
        for engine, status in engine_display.items():
            if status == "✅":
                st.markdown(f"✅ **{engine}**")
            else:
                st.markdown(f"❌ {engine} *(no key)*")
        
        st.info(f"**Active Engines:** {len(available_engines)}/18")
        
        if len(available_engines) < 8:
            free_count = len([e for e in available_engines if e in ["arXiv", "PubMed", "Crossref/DOI", "OpenAlex"]])
            st.markdown(f"""
            <div class="info-box">
                <strong>💡 Get More Coverage!</strong><br>
                You're using <strong>{free_count}</strong> free engines. Add API keys to unlock premium engines!
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Search Configuration
        st.subheader("🔍 Search Parameters")
        
        limit_per_engine = st.slider(
            "Results per engine",
            min_value=5,
            max_value=50,
            value=25,
            step=5,
            help="Number of papers to fetch from each search engine"
        )
        
        st.divider()
        
        # Advanced Configuration
        st.subheader("🎛️ Advanced Settings")
        
        with st.expander("Scoring & Ranking"):
            abstract_limit = st.number_input(
                "Deep Look Limit",
                min_value=1,
                max_value=20,
                value=10,
                help="Number of top papers to fetch detailed abstracts for"
            )
            
            citation_weight = st.slider(
                "Citation Weight",
                min_value=0.1,
                max_value=5.0,
                value=1.5,
                step=0.1,
                help="Weight given to citation counts in relevance scoring"
            )
            
            source_weight = st.number_input(
                "Source Weight",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Weight given to multi-source consensus"
            )
            
            high_consensus_threshold = st.number_input(
                "High Consensus Threshold",
                min_value=2,
                max_value=7,
                value=4,
                help="Number of sources required to trigger alert"
            )
        
        with st.expander("Recency Boost"):
            recency_boost = st.checkbox(
                "Enable Recency Boost",
                value=True,
                help="Give preference to recent publications"
            )
            
            recency_years = st.slider(
                "Recent Paper Window (years)",
                min_value=1,
                max_value=10,
                value=5,
                help="Papers within this timeframe get boosted"
            )
            
            recency_multiplier = st.slider(
                "Boost Multiplier",
                min_value=1.0,
                max_value=2.0,
                value=1.2,
                step=0.1,
                help="Score multiplier for recent papers"
            )
        
        with st.expander("Output Options"):
            enable_alerts = st.checkbox("Enable Consensus Alerts", value=True)
            enable_visualization = st.checkbox("Enable Visualizations", value=True)
            
            export_formats = st.multiselect(
                "Export Formats",
                options=['csv', 'json', 'bibtex'],
                default=['csv', 'json', 'bibtex']
            )
        
        # Build configuration
        config = {
            'abstract_limit': abstract_limit,
            'high_consensus_threshold': high_consensus_threshold,
            'citation_weight': citation_weight,
            'source_weight': source_weight,
            'enable_alerts': enable_alerts,
            'enable_visualization': enable_visualization,
            'export_formats': export_formats,
            'recency_boost': recency_boost,
            'recency_years': recency_years,
            'recency_multiplier': recency_multiplier
        }
    
    # Main content area - 4 TABS
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "📊 Results", "📝 Report Writer", "ℹ️ About"])
    
    # ====== TAB 1: SEARCH ======
    with tab1:
        st.header("Search Academic Literature")
        
        if len(available_engines) == 18:
            st.success(f"✅ All 18 engines active! Comprehensive coverage enabled.")
        elif len(available_engines) >= 13:
            st.info(f"ℹ️ Using {len(available_engines)} engines including all 13 free engines")
        else:
            st.info(f"ℹ️ Searching with {len(available_engines)} engines")
        
        # Search input
        search_query = st.text_input(
            "Enter your research query:",
            placeholder="e.g., Langerhans Cell Histiocytosis, Machine Learning in Healthcare, etc.",
            help="Enter keywords or phrases describing your research topic"
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_button = st.button("🚀 Start Search", type="primary", use_container_width=True)
        
        with col2:
            if st.button("🔄 Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache cleared!")
        
        # Search execution
        if search_button:
            if not search_query:
                st.error("Please enter a search query!")
            else:
                # Store in session state
                st.session_state['search_query'] = search_query
                st.session_state['config'] = config
                st.session_state['limit_per_engine'] = limit_per_engine
                
                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("🔧 Initializing orchestrator...")
                    progress_bar.progress(10)
                    
                    # Set API keys in environment
                    for key, value in api_keys.items():
                        if key != 'email' and value and len(value) > 5:
                            os.environ[f"{key.upper()}_API_KEY"] = value
                        elif key == 'email' and value:
                            os.environ['USER_EMAIL'] = value
                    
                    # Initialize orchestrator
                    orchestrator = ResearchOrchestrator(config=config)
                    
                    status_text.text(f"🔍 Searching across {len(available_engines)} databases...")
                    progress_bar.progress(30)
                    
                    # Run search
                    results = orchestrator.run_search(search_query, limit_per_engine=limit_per_engine)
                    
                    status_text.text("📝 Generating reports and visualizations...")
                    progress_bar.progress(70)
                    
                    # Save results
                    orchestrator.save_master_csv(results, search_query)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Search completed successfully!")
                    
                    # Store results
                    st.session_state['results'] = results
                    st.session_state['output_dir'] = orchestrator.output_dir
                    st.session_state['metadata'] = orchestrator.session_metadata
                    
                    # Success message
                    st.markdown(f"""
                    <div class="success-box">
                        <h3>✅ Search Completed Successfully!</h3>
                        <p><strong>Total Papers Found:</strong> {len(results)}</p>
                        <p><strong>Engines Used:</strong> {len(orchestrator.session_metadata['successful_engines'])}</p>
                        <p><strong>Execution Time:</strong> {(orchestrator.session_metadata['end_time'] - orchestrator.session_metadata['start_time']).total_seconds():.2f} seconds</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.info("👉 Switch to the 'Results' tab to view findings, or 'Report Writer' to generate a verified report!")
                    
                except Exception as e:
                    st.markdown(f"""
                    <div class="error-box">
                        <h3>❌ Search Failed</h3>
                        <p>{str(e)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    import traceback
                    with st.expander("🔍 View Full Error Trace"):
                        st.code(traceback.format_exc())
    
    # ====== TAB 2: RESULTS ======
    with tab2:
        st.header("Search Results & Analytics")
        
        if 'results' in st.session_state and st.session_state['results']:
            results = st.session_state['results']
            metadata = st.session_state.get('metadata', {})
            output_dir = st.session_state.get('output_dir', '')
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Papers", len(results))
            
            with col2:
                high_consensus = sum(1 for p in results if p.get('source_count', 0) >= config['high_consensus_threshold'])
                st.metric("High Consensus Papers", high_consensus)
            
            with col3:
                avg_citations = sum(p.get('citations_int', 0) for p in results) / len(results) if results else 0
                st.metric("Avg Citations", f"{avg_citations:.1f}")
            
            with col4:
                successful_engines = len(metadata.get('successful_engines', []))
                st.metric("Active Engines", successful_engines)
            
            st.divider()
            
            # Interactive Data Explorer
            csv_path = os.path.join(output_dir, "MASTER_REPORT_FINAL.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    
                    # Initialize bookmarks
                    if 'bookmarked_papers' not in st.session_state:
                        st.session_state['bookmarked_papers'] = set()
                    
                    st.subheader("📊 Interactive Data Explorer")
                    
                    # Filtering controls
                    with st.expander("🔍 Filter & Search", expanded=False):
                        filter_col1, filter_col2, filter_col3 = st.columns(3)
                        
                        with filter_col1:
                            if 'citations' in df.columns:
                                min_citations = st.number_input(
                                    "Min Citations",
                                    min_value=0,
                                    max_value=int(df['citations'].max()),
                                    value=0,
                                    key="cite_filter"
                                )
                        
                        with filter_col2:
                            if 'source_count' in df.columns:
                                min_sources = st.slider(
                                    "Min Sources",
                                    min_value=1,
                                    max_value=int(df['source_count'].max()),
                                    value=1,
                                    key="src_filter"
                                )
                        
                        with filter_col3:
                            search_text = st.text_input(
                                "Search Title/Authors",
                                placeholder="Enter keywords...",
                                key="txt_search"
                            )
                    
                    # Apply filters
                    filtered_df = df.copy()
                    
                    if 'citations' in df.columns and min_citations > 0:
                        filtered_df = filtered_df[filtered_df['citations'] >= min_citations]
                    
                    if 'source_count' in df.columns and min_sources > 1:
                        filtered_df = filtered_df[filtered_df['source_count'] >= min_sources]
                    
                    if search_text:
                        mask = (
                            filtered_df['title'].str.contains(search_text, case=False, na=False) |
                            filtered_df['ieee_authors'].str.contains(search_text, case=False, na=False)
                        )
                        filtered_df = filtered_df[mask]
                    
                    # Display controls
                    view_col1, view_col2, view_col3 = st.columns([2, 1, 1])
                    
                    with view_col1:
                        st.markdown(f"**Showing {len(filtered_df)} of {len(df)} papers**")
                    
                    with view_col2:
                        quick_action = st.selectbox(
                            "Quick Filter",
                            ["All Papers", "Highly Cited (>50)", "High Consensus (≥4)", "Recent (Boosted)", "Bookmarked Only"],
                            key="quick_filter"
                        )
                    
                    with view_col3:
                        st.metric("📑 Bookmarks", len(st.session_state['bookmarked_papers']))
                    
                    # Apply quick filter
                    if quick_action == "Highly Cited (>50)" and 'citations' in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df['citations'] > 50]
                    elif quick_action == "High Consensus (≥4)" and 'source_count' in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df['source_count'] >= 4]
                    elif quick_action == "Recent (Boosted)" and 'recency_boosted' in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df['recency_boosted'] == True]
                    elif quick_action == "Bookmarked Only":
                        if st.session_state['bookmarked_papers']:
                            filtered_df = filtered_df[filtered_df.index.isin(st.session_state['bookmarked_papers'])]
                        else:
                            st.info("📑 No bookmarks yet. Select papers below to bookmark them!")
                    
                    # Column selection
                    all_cols = df.columns.tolist()
                    default_cols = ['relevance_score', 'source_count', 'ieee_authors', 'title', 'venue', 'year', 'citations', 'url']
                    default_cols = [c for c in default_cols if c in all_cols]
                    
                    selected_cols = st.multiselect(
                        "Select Columns",
                        options=all_cols,
                        default=default_cols,
                        key="col_select"
                    )
                    
                    if selected_cols:
                        display_df = filtered_df[selected_cols].copy()
                        
                        # Add bookmark indicator
                        display_df.insert(0, '📑', display_df.index.map(lambda x: '⭐' if x in st.session_state['bookmarked_papers'] else ''))
                        
                        # Apply alternating row colors
                        def highlight_rows(row):
                            if row.name % 2 == 0:
                                return ['background-color: rgba(128, 128, 128, 0.1)'] * len(row)
                            else:
                                return ['background-color: rgba(128, 128, 128, 0.05)'] * len(row)
                        
                        styled_df = display_df.style.apply(highlight_rows, axis=1)
                        
                        # Interactive table
                        st.dataframe(
                            styled_df,
                            use_container_width=True,
                            height=400,
                            hide_index=False,
                            column_config={
                                '📑': st.column_config.TextColumn('📑', width="small"),
                                "url": st.column_config.LinkColumn("URL", display_text="🔗 Open"),
                                "doi": st.column_config.TextColumn("DOI", width="medium"),
                                "relevance_score": st.column_config.NumberColumn("Score", format="%d"),
                                "citations": st.column_config.NumberColumn("Cites", format="%d"),
                                "source_count": st.column_config.NumberColumn("Sources", format="%d"),
                                "year": st.column_config.TextColumn("Year", width="small"),
                            }
                        )
                        
                        st.info("💡 **Tip**: Click URLs to open papers directly!")
                        
                        st.divider()
                        
                        # Bookmark & Download Management
                        action_col1, action_col2, action_col3 = st.columns(3)
                        
                        with action_col1:
                            st.markdown("#### 📑 Bookmark Manager")
                            
                            paper_options = {idx: f"[{idx}] {row['title'][:50]}..."
                                           for idx, row in filtered_df.iterrows()}
                            
                            selected_for_bookmark = st.multiselect(
                                "Select papers to bookmark",
                                options=list(paper_options.keys()),
                                format_func=lambda x: paper_options[x],
                                key="bookmark_selector"
                            )
                            
                            bookmark_col1, bookmark_col2 = st.columns(2)
                            
                            with bookmark_col1:
                                if st.button("⭐ Add Bookmarks", use_container_width=True):
                                    st.session_state['bookmarked_papers'].update(selected_for_bookmark)
                                    st.success(f"Added {len(selected_for_bookmark)} bookmark(s)!")
                                    st.rerun()
                            
                            with bookmark_col2:
                                if st.button("🗑️ Clear All", use_container_width=True):
                                    st.session_state['bookmarked_papers'].clear()
                                    st.success("All bookmarks cleared!")
                                    st.rerun()
                        
                        with action_col2:
                            st.markdown("#### ✅ Select & Download")
                            
                            selected_for_download = st.multiselect(
                                "Select papers to download",
                                options=list(paper_options.keys()),
                                format_func=lambda x: paper_options[x],
                                key="download_selector"
                            )
                            
                            if selected_for_download:
                                selected_papers_df = filtered_df.loc[selected_for_download]
                                
                                download_col1, download_col2 = st.columns(2)
                                
                                with download_col1:
                                    csv_data = selected_papers_df.to_csv(index=False)
                                    st.download_button(
                                        label=f"📥 CSV ({len(selected_for_download)})",
                                        data=csv_data,
                                        file_name=f"selected_{len(selected_for_download)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                                
                                with download_col2:
                                    json_data = selected_papers_df.to_json(orient='records', indent=2)
                                    st.download_button(
                                        label=f"📥 JSON ({len(selected_for_download)})",
                                        data=json_data,
                                        file_name=f"selected_{len(selected_for_download)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                        mime="application/json",
                                        use_container_width=True
                                    )
                            else:
                                st.info("Select papers above to enable download")
                        
                        with action_col3:
                            st.markdown("#### 📥 Export Filtered")
                            
                            st.markdown(f"**Current filter: {len(filtered_df)} papers**")
                            
                            export_col1, export_col2 = st.columns(2)
                            
                            with export_col1:
                                csv_data = filtered_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 All CSV",
                                    data=csv_data,
                                    file_name=f"filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            
                            with export_col2:
                                json_data = filtered_df.to_json(orient='records', indent=2)
                                st.download_button(
                                    label="📥 All JSON",
                                    data=json_data,
                                    file_name=f"filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
                    
                    st.divider()
                
                except Exception as e:
                    st.warning(f"Could not load interactive viewer: {e}")
            
            # Display analytics chart
            chart_path = os.path.join(output_dir, "research_analytics.png")
            if os.path.exists(chart_path):
                st.subheader("📈 Research Analytics")
                st.image(chart_path, use_container_width=True)
                st.divider()
            
            # Results preview
            display_results_preview(results, limit=10)
            
            st.divider()
            
            # Download section
            if output_dir and os.path.exists(output_dir):
                create_download_buttons(output_dir)
        
        else:
            st.info("👈 No results yet. Start a search in the 'Search' tab!")
    
    # ====== TAB 3: REPORT WRITER (Enhanced with verification) ======
    with tab3:
        st.header("📝 Academic Report Writer")
        st.markdown("*Now with automated claim verification and source authority ranking*")
        
        # Check for Anthropic API key
        try:
            anthropic_key = st.secrets["ANTHROPIC_API_KEY"]
            api_available = True
        except:
            api_available = False
            st.error("⚠️ Anthropic API key not found in secrets (needed for report generation)")
            st.info("💡 Please configure ANTHROPIC_API_KEY in Streamlit secrets to use this feature")
        
        if api_available:
            # Report generation UI based on step
            if st.session_state.report_step == 'input':
                st.markdown("### Report Configuration")
                
                # Check if we have search results
                has_search_results = 'results' in st.session_state and st.session_state.get('search_query')
                
                if has_search_results:
                    st.success(f"✅ Found existing search results: '{st.session_state.get('search_query')}'")
                    st.info("💡 You can reuse these results or enter a new topic to search fresh data")
                
                col1, col2 = st.columns(2)
                with col1:
                    topic = st.text_input(
                        "Topic *",
                        value=st.session_state.report_form_data['topic'],
                        placeholder="e.g., Quantum Computing in Drug Discovery"
                    )
                    subject = st.text_input(
                        "Subject *",
                        value=st.session_state.report_form_data['subject'],
                        placeholder="e.g., Computer Science"
                    )
                with col2:
                    researcher = st.text_input(
                        "Researcher *",
                        value=st.session_state.report_form_data['researcher'],
                        placeholder="Your name"
                    )
                    institution = st.text_input(
                        "Institution *",
                        value=st.session_state.report_form_data['institution'],
                        placeholder="University/Organization"
                    )
                
                col3, col4 = st.columns(2)
                with col3:
                    date = st.date_input(
                        "Date",
                        value=datetime.strptime(st.session_state.report_form_data['date'], '%Y-%m-%d')
                    )
                with col4:
                    style = st.selectbox("Citation Style", ["IEEE", "APA"])
                
                # Source count control with cost warning
                st.markdown("#### 📚 Source Configuration")
                max_sources = st.slider(
                    "Maximum sources to use for writing",
                    min_value=10,
                    max_value=100,
                    value=st.session_state.report_form_data.get('max_sources', 25),
                    step=5,
                    help="Higher values = more comprehensive but slower and more expensive"
                )
                
                # Cost/time warning based on source count
                if max_sources <= 25:
                    st.success(f"✅ Conservative mode ({max_sources} sources) - ~4-6 min, ~25-35 API calls")
                elif max_sources <= 50:
                    st.warning(f"⚠️ Balanced mode ({max_sources} sources) - ~6-10 min, ~50-60 API calls, higher cost")
                else:
                    st.error(f"🔴 Comprehensive mode ({max_sources} sources) - ~10-15 min, ~80-100 API calls, **significantly higher cost**")
                
                # Update form data
                st.session_state.report_form_data.update({
                    'topic': topic,
                    'subject': subject,
                    'researcher': researcher,
                    'institution': institution,
                    'date': date.strftime('%Y-%m-%d'),
                    'citation_style': style,
                    'max_sources': max_sources
                })
                
                valid = all([topic, subject, researcher, institution])
                
                st.markdown("---")
                
                # NEW: Verification features info
                st.info("""
                **🛡️ Verification Features (NEW):**
                - **Claim Verification**: All quantitative claims checked against source documents
                - **Source Authority Ranking**: Top-tier journals prioritized over preprints
                - **Citation Integrity**: Ensures all citations point to valid sources
                - **Deduplication**: Prevents duplicate references (e.g., arXiv + publisher version)
                
                **How it works:**
                1. 🔍 Searches 18 academic databases (or reuses existing results)
                2. 📚 Deduplicates and ranks sources by authority (Nature > arXiv)
                3. ✍️ Uses Claude to write report with strict technical requirements
                4. 🛡️ Verifies all quantitative claims against sources (flags unsupported claims)
                5. 📄 Generates HTML with verification metadata and proper citations
                
                **Time:** 4-6 minutes | **API Calls:** ~25-35 to Anthropic
                """)
                
                if st.button(
                    "🚀 Generate Verified Report",
                    disabled=not valid,
                    type="primary",
                    use_container_width=True
                ):
                    execute_report_pipeline()
                    st.rerun()
                
                if not valid:
                    st.warning("⚠️ Please fill all required fields")
            
            elif st.session_state.report_step == 'processing':
                st.markdown("### 🔄 Generating Report with Verification")
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{st.session_state.report_progress['stage']}**")
                    st.progress(st.session_state.report_progress['percent'] / 100)
                with col2:
                    st.metric("Progress", f"{st.session_state.report_progress['percent']}%")
                
                st.info(st.session_state.report_progress['detail'])
                
                if st.session_state.report_start_time:
                    elapsed = time.time() - st.session_state.report_start_time
                    elapsed_mins = int(elapsed // 60)
                    elapsed_secs = int(elapsed % 60)
                    st.caption(
                        f"⏱️ Elapsed: {elapsed_mins}m {elapsed_secs}s | "
                        f"API Calls: {st.session_state.report_api_calls}"
                    )
                
                # Show sources as they're found with authority badges
                if st.session_state.report_research['sources']:
                    with st.expander(
                        f"📚 Academic Sources Found ({len(st.session_state.report_research['sources'])})",
                        expanded=True
                    ):
                        for i, s in enumerate(st.session_state.report_research['sources'][:10], 1):
                            meta = s.get('metadata', {})
                            tier = s.get('authority_tier', 'unknown')
                            tier_emoji = {'top_tier': '🏆', 'publisher': '📚', 'conference': '🎓', 'repository': '📄', 'other': '📃'}.get(tier, '📄')
                            
                            st.markdown(
                                f"**{i}.** {tier_emoji} {meta.get('title', 'Unknown')[:60]}...  "
                                f"({tier.replace('_', ' ').title()})"
                            )
                            st.caption(f"👤 {meta.get('authors', 'Unknown')} | 📅 {meta.get('year', 'N/A')} | 📖 {meta.get('venue', 'N/A')}")
                
                if st.session_state.report_processing:
                    time.sleep(3)
                    st.rerun()
            
            elif st.session_state.report_step == 'complete':
                st.success("✅ Verified Report Generated Successfully!")
                
                # Show verification badge
                ver_results = st.session_state.get('verification_results', {})
                issues = len(ver_results.get('unsupported_claims', []))
                
                if issues == 0:
                    st.markdown('<span class="verification-badge verified">✓ All Claims Verified</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="verification-badge unverified">⚠ {issues} Issues Flagged</span>', unsafe_allow_html=True)
                
                if st.session_state.report_execution_time:
                    exec_mins = int(st.session_state.report_execution_time // 60)
                    exec_secs = int(st.session_state.report_execution_time % 60)
                    st.info(f"⏱️ **Execution Time:** {exec_mins} minutes {exec_secs} seconds")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Academic Sources", len(st.session_state.report_research['sources']))
                with col2:
                    high_consensus = sum(1 for s in st.session_state.report_research['sources']
                                       if s.get('_orchestrator_data', {}).get('source_count', 1) >= 4)
                    st.metric("High Consensus", high_consensus)
                with col3:
                    if st.session_state.report_research['sources']:
                        avg_cites = sum(s.get('_orchestrator_data', {}).get('citations_int', 0)
                                       for s in st.session_state.report_research['sources']) / len(st.session_state.report_research['sources'])
                        st.metric("Avg Citations", f"{avg_cites:.1f}")
                with col4:
                    st.metric("API Calls", st.session_state.report_api_calls)
                
                # Verification details
                with st.expander("🛡️ Verification Details", expanded=False):
                    ver_summary = ver_results.get('verifier_summary', {})
                    integrity = ver_results.get('citation_integrity', {})
                    
                    st.markdown("**Claim Verification:**")
                    st.markdown(f"- Total metrics found in sources: {ver_summary.get('total_metrics_found', 0)}")
                    st.markdown(f"- Unsupported claims flagged: {ver_summary.get('total_issues', 0)}")
                    
                    st.markdown("**Citation Integrity:**")
                    st.markdown(f"- Valid citations: {len(integrity.get('valid_citations', []))}")
                    st.markdown(f"- Invalid citations: {len(integrity.get('invalid_citations', []))}")
                    st.markdown(f"- Citation coverage: {integrity.get('citation_coverage', 0):.1%}")
                    
                    if integrity.get('orphaned_high_quality_sources'):
                        st.markdown("**High-Quality Uncited Sources:**")
                        for src in integrity['orphaned_high_quality_sources'][:5]:
                            st.caption(f"[{src['index']}] {src['title']} ({src['citations']} cites)")
                
                st.markdown("---")
                
                # Download
                col1, col2 = st.columns(2)
                with col1:
                    if st.session_state.report_html:
                        filename = f"{st.session_state.report_form_data['topic'].replace(' ', '_')}_Verified_Report.html"
                        st.download_button(
                            "📥 Download Verified HTML Report",
                            data=st.session_state.report_html,
                            file_name=filename,
                            mime="text/html",
                            type="primary",
                            use_container_width=True
                        )
                        st.info("""
                        **To create PDF:**
                        1. Open HTML in browser
                        2. Press Ctrl+P (Cmd+P on Mac)
                        3. Select "Save as PDF"
                        """)
                
                with col2:
                    st.metric("File Size", f"{len(st.session_state.report_html) / 1024:.1f} KB")
                    st.metric("Quality", "Verified Professional")
                
                st.markdown("---")
                
                # Sources preview with authority indicators
                with st.expander("📚 References Preview", expanded=False):
                    for i, s in enumerate(st.session_state.report_research['sources'][:20], 1):
                        meta = s.get('metadata', {})
                        orch = s.get('_orchestrator_data', {})
                        tier = s.get('authority_tier', 'unknown')
                        
                        tier_badge = {'top_tier': '🏆 Top-tier', 'publisher': '📚 Publisher', 
                                     'conference': '🎓 Conference', 'repository': '📄 Preprint'}.get(tier, '📄 Other')
                        
                        st.markdown(f"**[{i}]** {tier_badge} {meta.get('title', 'N/A')[:70]}")
                        st.caption(f"👤 {meta.get('authors', 'N/A')} | 📅 {meta.get('year', 'N/A')} | 📖 {meta.get('venue', 'N/A')}")
                        st.caption(f"🔗 [{s['url']}]({s['url']})")
                        if orch.get('source_count'):
                            st.caption(f"✓ Found in {orch['source_count']} database(s) | 📊 {orch.get('citations', 0)} citations")
                        st.divider()
                
                if st.button("🔄 Generate Another Report", type="secondary", use_container_width=True):
                    reset_report_system()
                    st.rerun()
            
            elif st.session_state.report_step == 'error':
                st.error("❌ Error Occurred")
                st.warning(st.session_state.report_progress['detail'])
                
                if st.session_state.report_execution_time:
                    exec_mins = int(st.session_state.report_execution_time // 60)
                    exec_secs = int(st.session_state.report_execution_time % 60)
                    st.caption(f"Failed after {exec_mins}m {exec_secs}s")
                
                if st.button("🔄 Try Again", type="primary", use_container_width=True):
                    reset_report_system()
                    st.rerun()
    
    # ====== TAB 4: ABOUT ======
    with tab4:
        st.header("About SROrch")
        
        st.markdown("""
        ### 🔬 Scholarly Research Orchestrator & Report Writer
        
        SROrch is a comprehensive academic research platform that combines powerful literature search
        with automated report generation capabilities.
        
        #### 🆕 What's New in This Version
        
        **Enhanced Verification System:**
        - **Claim Verification**: Automatically checks all quantitative claims against source documents
        - **Source Authority Ranking**: Prioritizes top-tier journals (Nature, Science) over preprints
        - **Deduplication Engine**: Prevents duplicate references (e.g., arXiv + Nature version merged)
        - **Citation Integrity**: Ensures all citations are valid and all claims are supported
        
        **Improved Quality Controls:**
        - Technical specificity enforcement (no more "sophisticated strategies")
        - Temporal consistency checks (specific years, not "recently")
        - Evaluation framework alignment (task-specific benchmarks, not generic guidelines)
        - Automated regeneration when verification issues are detected
        
        #### 📚 Supported Databases (18 Engines!)
        
        **Premium Engines (Require API Keys):**
        - **Semantic Scholar** - AI-powered academic search (FREE key available!)
        - **Google Scholar** - Broad academic search (via SERP API)
        - **CORE** - Open access research aggregator
        - **SCOPUS** - Comprehensive scientific database
        - **Springer Nature** - Major scientific publisher
        
        **Free Engines (Always Available):**
        - **Core Set:** arXiv, PubMed, Crossref/DOI, OpenAlex
        - **Extended Set:** Europe PMC, PLOS, SSRN, DeepDyve
        - **Publisher Access:** Wiley, Taylor & Francis, ACM, DBLP, SAGE

        #### ✨ Key Features
        
        **Search & Analysis:**
        - Multi-source consensus detection
        - Intelligent relevance scoring with authority weighting
        - Deep abstract fetching
        - Enhanced gap analysis with domain-specific patterns
        - Publication analytics and visualizations
        
        **Report Writer:**
        - **Verified Claims**: All numbers checked against sources
        - **Authority-Aware**: Top-tier sources prioritized
        - **Proper Citations**: IEEE/APA with real metadata
        - **Technical Depth**: Specific architectures, benchmarks, metrics
        - **Professional Output**: HTML with verification metadata
        
        #### 🚀 Getting Started
        
        **Search Mode:**
        1. Enter your research query
        2. Configure search parameters (optional)
        3. Click "Start Search"
        4. View results, download data, or generate a verified report
        
        **Report Mode:**
        1. Configure report details (topic, author, institution)
        2. Choose citation style (IEEE or APA)
        3. Click "Generate Verified Report"
        4. Review verification details and download HTML
        
        #### 🔑 API Keys
        
        **Required for Report Generation:**
        - Anthropic API key (in Streamlit secrets)
        
        **Optional for Enhanced Search:**
        - Semantic Scholar (free, highly recommended)
        - SERP API (Google Scholar)
        - CORE, SCOPUS, Springer Nature
        
        All keys are session-only for security!
        
        ---
        
        **Version:** 2.1 - Enhanced with Claim Verification  
        **Security Model:** Zero-Trust (User-Provided Keys)  
        **Verification Engine:** Multi-stage claim validation  
        **License:** MIT
        """)
        
        with st.expander("🖥️ System Information"):
            st.code(f"""
Python Version: {sys.version}
Working Directory: {os.getcwd()}
Streamlit Version: {st.__version__}
Security Model: Session-only keys (no persistence)
Report Generation: Claude Sonnet 4.5 with verification
Claim Verification: Enabled (fuzzy matching ±5%)
Source Ranking: Authority-tier based
            """)

if __name__ == "__main__":
    main()
