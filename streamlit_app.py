# ==============================================
# streamlit_app.py
# SROrch Streamlit Interface
# ==============================================

"""
COMPLETE VERSION with Strict Verification
A comprehensive web interface for Scholarly Research Orchestrator with 
integrated academic report generation and factual integrity controls
"""

import streamlit as st
import os
import sys
import json
import shutil
import base64
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

# Import fixes
from srorch_critical_fixes import (
    SourceQualityFilter,
    normalize_publication_year,
    create_source_boundary_prompt,
    AlignedClaimVerifier,
    integrate_fixes_into_pipeline,
)

# Page configuration
st.set_page_config(
    page_title="SROrch - Research Orchestrator & Reviewer",
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

# ==============================================
# CONFIGURATION & CONSTANTS
# ==============================================

MODEL_PRIMARY = "claude-sonnet-4-20250514"
MODEL_FALLBACK = "claude-haiku-3-5-20241022"
MIN_API_DELAY = 3.0
RETRY_DELAYS = [10, 20, 40]

# CRITICAL: Forbidden words that must not appear without specific metrics
FORBIDDEN_GENERIC_TERMS = [
    'advanced', 'sophisticated', 'state-of-the-art', 'state of the art',
    'enhanced', 'novel', 'cutting-edge', 'cutting edge', 'innovative',
    'powerful', 'robust', 'seamless', 'efficient', 'effective',
    'comprehensive', 'holistic', 'groundbreaking', 'revolutionary'
]
NSOURCES = 20

# ==============================================
# ENHANCED CLAIM VERIFICATION SYSTEM 
# (STRICT MODE)
# ==============================================

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
                'years': [],
                'raw_numbers': []  # ✅ ADD THIS LINE
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
                source_metrics['raw_numbers'].append(match.group(1))  # ✅ ADD THIS LINE
            
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
                source_metrics['raw_numbers'].append(match.group(2))  # ✅ ADD THIS LINE
            
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
            all_violations.extend(self.verify_exact_numbers(content, section_name))  # ✅ ADD THIS
            all_violations.extend(self.check_forbidden_language(content, section_name))
            all_violations.extend(self.verify_technical_specificity(content, section_name))
            all_violations.extend(self.verify_citation_support(content, section_name))

        
        for section in draft.get('mainSections', []):
            content = section.get('content', '')
            title = section.get('title', 'Section')
            all_violations.extend(self.verify_exact_numbers(content, f"mainSection:{title}"))  # ✅ ADD THIS
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

    def verify_exact_numbers(self, text: str, section: str) -> List[Dict]:
        """Verify that all numbers in text appear exactly in cited sources"""
        violations = []
        
        # Find all number + citation pairs
        pattern = r'(\d+(?:\.\d+)?%?)[^[]{0,100}\[(\d+)\]'
        
        for match in re.finditer(pattern, text):
            number = match.group(1).replace('%', '')
            citation = match.group(2)
            
            if citation not in self.source_metrics:
                violations.append({
                    'type': 'invalid_citation',
                    'number': number,
                    'citation': citation,
                    'section': section,
                    'severity': 'CRITICAL',
                    'issue': f'Source [{citation}] not found'
                })
                continue
            
            # Check if exact number exists in source
            source_data = self.source_metrics[citation]
            raw_numbers = source_data.get('raw_numbers', [])
            
            if number not in raw_numbers:
                violations.append({
                    'type': 'hallucinated_number',
                    'number': number,
                    'citation': citation,
                    'section': section,
                    'severity': 'CRITICAL',
                    'issue': f'Number {number} NOT in source [{citation}]',
                    'suggestion': f'Available numbers: {raw_numbers[:5]}'
                })
        
        return violations

# ==============================================
# FIXED SOURCE AUTHORITY CLASSIFICATION
# ==============================================

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

def deduplicate_and_rank_sources_strict(sources: List[Dict]) -> List[Dict]:
    """Enhanced deduplication with DOI + arXiv ID + title matching"""

    def normalize_doi(doi: str) -> str:
        """Normalize DOI for comparison"""
        if not doi or doi.upper() in ('N/A', 'NA', 'UNKNOWN', 'NONE'):
            return ''
        doi = re.sub(r'^(doi:|https?://doi.org/|https?://dx.doi.org/)', '', doi.lower().strip())
        doi = re.sub(r'\s+', '', doi)
        return doi if len(doi) > 5 else ''
    
    def extract_arxiv_id(url: str) -> str:
        """Extract arXiv ID from URL"""
        match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)', url.lower())
        return match.group(1) if match else ''
    
    def normalize_title(title: str) -> str:
        """Normalize title for comparison"""
        if not title:
            return ''
        normalized = re.sub(r'[^\w\s]', '', title.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    

    TIER_ORDER = {
        'top_tier_journal': 0,
        'publisher_journal': 1,
        'conference': 2,
        'other': 3,
        'preprint': 4
    }
    
    seen = {}
    doi_to_key = {}
    arxiv_to_key = {}
    title_to_key = {}
    
    for source in sources:
        metadata = source.get('metadata', {})
        
        # Extract identifiers
        doi = normalize_doi(metadata.get('doi', ''))
        arxiv_id = extract_arxiv_id(source.get('url', ''))
        title = normalize_title(metadata.get('title', ''))
        
        # Get authority tier
        venue = metadata.get('venue', '')
        url = source.get('url', '')
        tier = get_authority_tier_fixed(venue, url)
        source['authority_tier'] = tier
        
        # Determine unique key with cross-linking
        key = None
        
        # Check if DOI already seen
        if doi:
            if doi in doi_to_key:
                key = doi_to_key[doi]
            else:
                key = f"doi:{doi}"
                doi_to_key[doi] = key
        
        # Check if arXiv ID already seen and link to DOI
        if not key and arxiv_id:
            if arxiv_id in arxiv_to_key:
                key = arxiv_to_key[arxiv_id]
            else:
                if doi:  # Link arXiv to DOI
                    arxiv_to_key[arxiv_id] = doi_to_key[doi]
                    key = doi_to_key[doi]
                else:
                    key = f"arxiv:{arxiv_id}"
                    arxiv_to_key[arxiv_id] = key
        
        # Fallback to title matching
        if not key and title and len(title) > 30:
            if title in title_to_key:
                key = title_to_key[title]
            else:
                # Link title to existing DOI/arXiv if available
                if doi and doi in doi_to_key:
                    title_to_key[title] = doi_to_key[doi]
                    key = doi_to_key[doi]
                elif arxiv_id and arxiv_id in arxiv_to_key:
                    title_to_key[title] = arxiv_to_key[arxiv_id]
                    key = arxiv_to_key[arxiv_id]
                else:
                    key = f"title:{title[:50]}"
                    title_to_key[title] = key
        
        if not key:
            continue
        
        # Merge or add
        if key in seen:
            existing = seen[key]
            existing['source_count'] = existing.get('source_count', 1) + 1
            
            # Update to maximum citations
            existing_cites = safe_int(existing.get('metadata', {}).get('citations', 0))
            new_cites = safe_int(metadata.get('citations', 0))
            if new_cites > existing_cites:
                existing['metadata']['citations'] = new_cites
            
            # Upgrade to better tier
            existing_tier_rank = TIER_ORDER.get(existing.get('authority_tier'), 5)
            new_tier_rank = TIER_ORDER.get(tier, 5)
            
            if new_tier_rank < existing_tier_rank:
                existing['authority_tier'] = tier
                existing['metadata']['venue'] = venue
                existing['url'] = url
                if doi and not normalize_doi(existing['metadata'].get('doi', '')):
                    existing['metadata']['doi'] = metadata.get('doi', 'N/A')
        else:
            source['source_count'] = 1
            seen[key] = source
    
    ranked = sorted(
        seen.values(),
        key=lambda x: (
            TIER_ORDER.get(x.get('authority_tier'), 5),
            -safe_int(x.get('metadata', {}).get('year', 0)),
            -safe_int(x.get('metadata', {}).get('citations', 0)),
            -x.get('source_count', 1)
        )
    )
    
    return ranked


# ==============================================
# TECHNICAL SPECIFICATION EXTRACTION
# ==============================================

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

# ==============================================
# CITATION COVERAGE ENFORCEMENT
# ==============================================

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

# ==============================================
# ORIGINAL UTILITY FUNCTIONS (Preserved)
# ==============================================

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
    """Initialize session state with empty API keys and reviewer state"""
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
    
    # Reviewer state
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
        if st.button("✅ Apply Keys (This Session Only)", key="apply_keys", width='stretch'):
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

# ==============================================
# REPORT GENERATION FUNCTIONS
# ==============================================

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

def call_anthropic_api(
    messages: List[Dict], 
    max_tokens: int = 1000, 
    use_fallback: bool = False,
    system: str = None  # NEW PARAMETER
) -> Dict:
    """
    FIXED: Call Anthropic API with proper system parameter handling
    
    Args:
        messages: List of message dicts with 'role' (user/assistant only) and 'content'
        max_tokens: Maximum tokens to generate
        use_fallback: Whether to use fallback model
        system: Optional system prompt (passed as top-level parameter, NOT in messages)
    """
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
    
    # Build request data
    data = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages  # ✅ Must only contain user/assistant roles
    }
    
    # ✅ FIX: Add system parameter separately (not as a message role)
    if system:
        data["system"] = system
    
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
            
            # ✅ NEW: Better error reporting for 400 errors
            if response.status_code == 400:
                error_detail = response.json()
                raise Exception(f"400 Bad Request: {error_detail.get('error', {}).get('message', 'Unknown error')}")
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            st.warning(f"⚠️ API error (attempt {attempt+1}/3): {str(e)[:100]}")
            if attempt == 2:
                if not use_fallback:
                    st.info("🔄 Trying fallback model...")
                    return call_anthropic_api(messages, max_tokens, use_fallback=True, system=system)
                raise
            time.sleep(RETRY_DELAYS[attempt])
    
    if not use_fallback:
        st.info("🔄 Primary model failed. Trying fallback model...")
        return call_anthropic_api(messages, max_tokens, use_fallback=True, system=system)
    
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
    """Convert ResearchOrchestrator output to report reviewer source format"""
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

# ==============================================
# ENHANCED DRAFT GENERATION WITH STRICT CONTROLS
# ==============================================

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
    max_sources: int = 25,
    boundary_prompt: str = ""  # CRITICAL FIX: Added parameter
) -> Tuple[Dict, Dict]:
    """Generate draft with strict technical specificity and source boundaries"""
    
    update_report_progress('Drafting', 'Writing with strict technical requirements...', 65)
    
    if not sources:
        raise Exception("No sources available")
    
    # Aggregate technical specifications from sources
    all_specs = aggregate_technical_specs(sources[:max_sources])
    
    # Prepare source list with technical annotations
    source_list = []
    for i, s in enumerate(sources[:max_sources], 1):
        meta = s.get('metadata', {})
        tier = s.get('authority_tier', 'unknown')
        
        # Extract specs for this source
        source_text = ' '.join([meta.get('title', ''), s.get('content', '')])
        source_specs = extract_technical_specifications(source_text)
        spec_summary = ', '.join([f"{k}: {', '.join(v[:3])}" for k, v in source_specs.items() if v])
        
        tier_emoji = {
            'top_tier_journal': '🏆',
            'publisher_journal': '📚',
            'conference': '🎓',
            'preprint': '📄',
            'other': '📃'
        }.get(tier, '📄')
        
        source_list.append(f"""[{i}] {tier_emoji} [{tier.upper()}] {meta.get('title', 'Unknown')} ({meta.get('year', 'N/A')})
Authors: {meta.get('authors', 'Unknown')}
Venue: {meta.get('venue', 'Unknown')} | Citations: {meta.get('citations', 0)}
Technical: {spec_summary if spec_summary else 'General content'}
URL: {s['url'][:70]}""")
    
    sources_text = "\n\n".join(source_list)
    
    # CRITICAL FIX: Build system prompt with boundary enforcement
    base_system_prompt = f"""You are a PRECISE technical report generator. ABSOLUTE RULES:

CRITICAL RULE - EXACT NUMBERS ONLY:
Every quantitative claim MUST use the EXACT number from sources.
✓ "OpenScholar-8B outperforms GPT-4o by 5% in correctness [16]" 
✗ "approximately 15-20% improvement" (FORBIDDEN - use exact numbers)
✗ "significant improvement" (FORBIDDEN - vague)

If exact number not in source: Write "Not specified in sources" or omit claim.
NEVER round, estimate, approximate, or invent numbers.

VERIFICATION REQUIREMENT:
Before writing ANY number:
1. Check if EXACT number appears in source text
2. If yes: Use verbatim with citation [X]
3. If no: Do NOT write that number

EXAMPLES OF VIOLATIONS:
❌ "15-20% improvement" (range not in source)
❌ "approximately 78% satisfaction" (approximation)
❌ "significant performance gains" (no number)

✅ "outperforms GPT-4o by 5%" (exact from source)
✅ "45M papers in datastore" (exact from source)
✅ "71.8% on ScholarQABench" (exact from source)
    
FORBIDDEN WORDS (without specific metrics):
{', '.join(FORBIDDEN_GENERIC_TERMS[:NSOURCES])}
Replace with exact specifications or remove.

MANDATORY SPECIFICITY:
Every claim needs ONE of:
- Exact benchmark + score: "ScholarQABench: 71.8%"
- Parameter count: "8B parameters"
- Dataset size: "45M papers, 237M embeddings"
- System name + version: "OpenScholar-8B"
- Specific year: "2024" not "recent"

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
- Cite AT LEAST {int(max_sources * 0.6)} different sources minimum
- Every paragraph must contain 2-4 citations [X]
- Never cite same source twice in one paragraph
- Use high-authority sources [1]-[10] preferentially

RULE 4 - IF UNSURE:
Write "Not specified in source [X]" or omit entirely.
NEVER invent numbers, benchmarks, or metrics.

EXTRACTED TECHNICAL DETAILS FROM SOURCES:
Benchmarks: {', '.join(all_specs.get('benchmarks', ['None found']))}
Models: {', '.join(all_specs.get('models', ['None found']))}
Parameter counts: {', '.join(all_specs.get('parameter_counts', ['None found']))}
Dataset sizes: {', '.join(all_specs.get('dataset_sizes', ['None found']))}
Architectures: {', '.join(all_specs.get('architectures', ['None found']))}

REMEMBER: Specificity is MANDATORY. Generic language is FORBIDDEN."""

    # CRITICAL FIX: Prepend boundary prompt to enforce source constraints
    full_system_prompt = boundary_prompt + "\n\n" + base_system_prompt if boundary_prompt else base_system_prompt

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
SOURCE BOUNDARY: You may ONLY cite [1]-[{len(sources[:max_sources])}].
DO NOT mention any paper not in this numbered list.
DO NOT invent authors or systems."""

    # Section-specific requirements
    section_requirements = """
SECTION REQUIREMENTS (All must be met):

Abstract (200 words):
- MUST name specific benchmark(s) with scores
- MUST include dataset size or parameter count
- MUST cite [5] (Asai et al. Nature 2025) if available

Introduction:
- MUST cite specific founding paper with year
- MUST quantify the problem (e.g., "X papers published annually")

Literature Review:
- MUST compare specific systems with metric differences
- NO generic "many studies have shown"

Main Sections (4 required):
1. Architecture: Name specific encoders, dimensions, parameters
2. Benchmarks: List ALL benchmarks found in sources with scores
3. Comparisons: System A vs System B with percentage differences
4. Applications: Specific domains with quantitative results

Data & Analysis:
- MUST include specific numbers for every claim
- Table-like comparisons preferred

Challenges:
- Specific technical limitations (e.g., "bi-encoder latency of Xms")
- NOT generic "challenges remain"

Future:
- Concrete capabilities with predicted metrics
- Specific technical directions

Conclusion:
- Summary of specific achievements with numbers
- NO new claims without citations"""

    user_prompt = f"""Write technical report about "{topic}" in {subject}.

{variations_text}

{eval_context}

{strict_boundary}

{section_requirements}

ACADEMIC SOURCES TO USE:
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

REMINDER: Every claim needs [X] citation. Every number needs source support. Generic terms are forbidden."""

    # Add temporal context
    user_prompt = add_temporal_context(user_prompt, sources[:max_sources])
    
    # First generation attempt with boundary-enforced prompt
    response = call_anthropic_api(
        [{"role": "user", "content": user_prompt}],
        max_tokens=6000,
        system=full_system_prompt  # CRITICAL: Uses combined prompt with boundaries
    )

    text = "".join([c['text'] for c in response['content'] if c['type'] == 'text'])
    draft = parse_json_response(text)
    
    # Ensure structure
    required_keys = ['abstract', 'introduction', 'literatureReview', 'mainSections',
                    'dataAnalysis', 'challenges', 'futureOutlook', 'conclusion']
    for key in required_keys:
        if key not in draft or not draft[key]:
            draft[key] = "Section content." if key != 'mainSections' else [{"title": "Section", "content": "Content."}]
    
    # CRITICAL FIX: Use AlignedClaimVerifier instead of StrictClaimVerifier
    update_report_progress('Verification', 'Running strict claim verification...', 75)
    verifier = AlignedClaimVerifier(sources[:max_sources])  # CHANGED: AlignedClaimVerifier
    verification = verifier.verify_draft(draft)  # CHANGED: New verify_draft method
    
    # Check source coverage
    coverage_check = {
        'coverage': verification.get('coverage', 0),
        'cited_count': verification.get('cited_count', 0),
        'total_sources': verification.get('total_sources', 0),
        'status': 'ok' if verification.get('coverage', 0) >= 0.5 else 'insufficient_coverage'
    }
    
    # If critical issues, attempt regeneration with corrections
    if verification.get('has_critical') or coverage_check['status'] != 'ok':
        update_report_progress('Refinement', 'Fixing critical issues...', 85)
        
        correction_prompt = f"""
CRITICAL ISSUES FOUND - MUST FIX:

{chr(10).join([f"- {v['type']}: {v.get('issue', '')}" for v in verification.get('violations', [])[:NSOURCES]])}

COVERAGE ISSUE: {coverage_check['message'] if 'message' in coverage_check else 'Insufficient source coverage'}

REGENERATION RULES:
1. Remove ALL forbidden generic terms
2. Add specific metrics (benchmarks, parameters, percentages) to every claim
3. Ensure minimum {int(max_sources * 0.5)} unique citations
4. If specific number unavailable, remove claim or write "Not specified in sources"

Return corrected JSON."""
        
        try:
            retry_response = call_anthropic_api(
                [{"role": "user", "content": user_prompt + "\n\n" + correction_prompt}],
                max_tokens=6000,
                system=full_system_prompt + "\n\nTHIS IS A CORRECTION ATTEMPT. FIX ALL ISSUES LISTED."
            )
            
            retry_text = "".join([c['text'] for c in retry_response['content'] if c['type'] == 'text'])
            corrected_draft = parse_json_response(retry_text)
            
            # Validate structure
            valid = all(k in corrected_draft and corrected_draft[k] for k in required_keys)
            if valid:
                draft = corrected_draft
                # Re-verify
                verification = verifier.verify_draft(draft)
                coverage_check = {
                    'coverage': verification.get('coverage', 0),
                    'cited_count': verification.get('cited_count', 0),
                    'total_sources': verification.get('total_sources', 0)
                }
                verification['correction_attempted'] = True
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

# ==============================================
# CITATION FORMATTING AND HTML GENERATION
# ==============================================

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
    """Format citation with correct tier badge, DOI, and URL"""
    meta = source.get('metadata', {})
    tier = source.get('authority_tier', 'unknown')
    url = source.get('url', '')
    doi = meta.get('doi', '')
    
    tier_labels = {
        'top_tier_journal': '[Nature/Science]',
        'publisher_journal': '[Journal]',
        'conference': '[Conference]',
        'preprint': '[Preprint]',
        'other': '[Other]'
    }
    
    if style == 'APA':
        citation = f"{meta.get('authors', 'Unknown')} ({meta.get('year', 'n.d.')}). {meta.get('title', 'Untitled')}. <i>{meta.get('venue', 'Unknown')}</i>. {tier_labels.get(tier, '')}"
    else:  # IEEE
        authors = format_authors_ieee(meta.get('authors', 'Unknown'))
        citation = f'[{index}] {authors}, "{meta.get("title", "Untitled")}," {meta.get("venue", "Unknown")}, {meta.get("year", "n.d.")}. {tier_labels.get(tier, "")}'
    
    # ✅ NEW: Add DOI if available (preferred for published papers)
    if doi and doi not in ('N/A', 'NA', 'Unknown', 'NONE', ''):
        # Clean DOI - remove URL prefixes if present
        clean_doi = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '').replace('doi:', '').strip()
        if clean_doi:  # Only add if not empty after cleaning
            citation += f' DOI: <a href="https://doi.org/{clean_doi}">{clean_doi}</a>'
    # ✅ NEW: Otherwise add URL (for preprints, web resources, etc.)
    elif url:
        citation += f' <a href="{url}">{url}</a>'
    
    return citation

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

    # Load logo from file (prefer SVG, fallback to JPG)
    logo_html = ''
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for logo_name, mime in [('logo.svg', 'image/svg+xml'), ('logo.jpg', 'image/jpeg')]:
        logo_path = os.path.join(script_dir, logo_name)
        if os.path.isfile(logo_path):
            with open(logo_path, 'rb') as f:
                logo_b64 = base64.b64encode(f.read()).decode('utf-8')
            logo_html = f'<img src="data:{mime};base64,{logo_b64}" alt="SROrch" style="height: 65px;">'
            break

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
        @page {{ size: A4; margin: 1cm 1cm 2cm 2cm; }}
        body {{
            font-family: 'Times New Roman', serif;
            font-size: 12pt;
            line-height: 1.6;
            color: #000;
            max-width: 180mm;
            margin: 0 auto;
            padding: 1cm 1cm 2cm 2cm;
        }}
        .cover {{
            text-align: center;
            padding-top: 0;
            page-break-after: always;
            page-break-inside: avoid;
        }}
        .cover-logo {{
            margin-bottom: 1.5cm;
        }}
        .cover-logo img {{
            height: 65px;
        }}
        .cover h1 {{
            font-size: 22pt;
            font-weight: bold;
            margin: 1.5cm 0 0.8cm 0;
            border-bottom: none;
            padding-bottom: 0;
        }}
        .cover .meta {{
            font-size: 14pt;
            margin: 0.2cm 0;
        }}
        .cover .meta-subtitle {{
            font-size: 13pt;
        }}
        .cover .meta-author {{
            font-size: 12pt;
            margin-top: 2cm;
        }}
        .cover .meta-footer {{
            font-size: 9pt;
            margin-top: 1cm;
            color: #555;
        }}
        h1 {{
            font-size: 18pt;
            margin-top: 0.5in;
            border-bottom: 2px solid #333;
            padding-bottom: 0.1in;
            page-break-after: avoid;
        }}
        h2 {{
            font-size: 14pt;
            margin-top: 0.3in;
            font-weight: bold;
            page-break-after: avoid;
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
            page-break-inside: avoid;
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
        <div class="cover-logo">
            {logo_html}
        </div>
        <h1>{form_data['topic']}</h1>
        <div class="meta meta-subtitle">Technical Research Report</div>
        <div class="meta meta-subtitle">Subject: {form_data['subject']}</div>
        <div class="meta meta-author">
            {form_data['researcher']}<br>
            {form_data['institution']}<br>
            {report_date}
        </div>
        <div class="meta meta-footer">
            Generated by SROrch | STRICT MODE
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
            source = sources[i - 1]
            # Apply off-topic filtering to uncited sources
            s_title = str(source.get('metadata', {}).get('title', '')).lower()
            s_content = str(source.get('content', '')).lower()
            s_combined = f"{s_title} {s_content}"
            off_topic = sum(1 for kw in SourceQualityFilter.OFFTOPIC_INDICATORS if kw in s_combined)
            if off_topic >= 1:
                continue
            uncited_sources.append((i, source))
    
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
        for v in ver['violations'][:NSOURCES]:
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

# ==============================================
# SEARCH TAB FUNCTIONS
# ==============================================

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
                width='stretch'
            )

# ==============================================
# ENHANCED PIPELINE EXECUTION
# ==============================================

def execute_report_pipeline():
    """Execute complete report generation pipeline with critical fixes integrated"""
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
        max_sources = st.session_state.report_form_data.get('max_sources', 25)
        api_keys = load_api_keys()
        
        # Stage 1: Topic Analysis
        st.info("🔍 Stage 1/6: Analyzing topic...")
        update_report_progress('Analysis', 'Generating research plan...', 10)
        analysis = analyze_topic_with_ai(topic, subject)
        st.session_state.report_research['subtopics'] = analysis['subtopics']
        
        # Stage 2: Retrieve or reuse sources
        st.info("🔬 Stage 2/6: Retrieving academic sources...")
        update_report_progress('Research', 'Searching databases...', 25)
        
        # Initialize report_research if not exists
        if 'report_research' not in st.session_state:
            st.session_state.report_research = {}
        
        reuse_existing = False
        raw_sources = []  # Will hold sources before filtering
        
        # Check for existing search results
        if 'results' in st.session_state and st.session_state.get('search_query'):
            existing_query = st.session_state.get('search_query', '').lower()
            current_topic = topic.lower()
            # Check if existing results match current topic
            if any(term in existing_query for term in current_topic.split()[:3]):
                st.info(f"✅ Reusing existing search results: '{st.session_state.get('search_query')}'")
                results = st.session_state['results']
                reuse_existing = True
        
        if not reuse_existing:
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
            
            # Set API keys
            for key, value in api_keys.items():
                if key != 'email' and value and len(value) > 5:
                    os.environ[f"{key.upper()}_API_KEY"] = value
                elif key == 'email' and value:
                    os.environ['USER_EMAIL'] = value
            
            orchestrator = ResearchOrchestrator(config=orchestrator_config)
            update_report_progress('Research', f'Searching for "{topic}"...', 35)
            
            search_query = f"{topic} {subject}".strip()
            results = orchestrator.run_search(search_query, limit_per_engine=15)
            
            if not results:
                raise Exception("No results found from academic databases")
            
            # Store results in session for potential reuse
            st.session_state['results'] = results
            st.session_state['search_query'] = search_query
            
            update_report_progress('Research', f'Found {len(results)} papers', 50)
        
        # CRITICAL: Convert results to source format and store
        raw_sources = convert_orchestrator_to_source_format(results)
        
        # Store in report_research for Stage 3 access
        st.session_state.report_research['sources'] = raw_sources
        st.session_state.report_research['raw_results'] = results
        
        # Stage 3: QUALITY FILTERING (CRITICAL FIX)
        st.info("🛡️ Stage 3/6: Filtering low-quality sources...")
        update_report_progress('Filtering', 'Removing irrelevant sources...', 55)
        
        # Now retrieve from session (guaranteed to exist)
        sources_to_filter = st.session_state.report_research.get('sources', [])
        
        if not sources_to_filter:
            raise Exception(f"No sources available for filtering. Raw sources count: {len(raw_sources)}")
        
        # Apply critical fixes pipeline
        sources, fix_metadata = integrate_fixes_into_pipeline(sources_to_filter, topic)
        
        # Show filtering results to user
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Sources", fix_metadata['original_count'])
        with col2:
            st.metric("After Filtering", fix_metadata['filtered_count'])
        with col3:
            st.metric("Year Corrections", fix_metadata.get('year_corrections', 0))
        
        # Show domain detection
        domain_display = {
            'medical': '🏥 Medical',
            'computer_science': '💻 Computer Science',
            'general': '📚 General'
        }.get(fix_metadata.get('domain', 'general'), '📚 General')
        
        st.info(f"Domain detected: {domain_display}")
        
        if fix_metadata['original_count'] != fix_metadata['filtered_count']:
            with st.expander(f"View {fix_metadata['original_count'] - fix_metadata['filtered_count']} filtered sources"):
                for reason in fix_metadata.get('rejection_reasons', [])[:NSOURCES]:
                    st.caption(f"• {reason}")
        
        if len(sources) < 5:
            st.warning(f"Only {len(sources)} sources after filtering. Using top 10 unfiltered.")
            sources = sources_to_filter[:NSOURCES]
        
        # Stage 4: TEMPORAL NORMALIZATION
        st.info("📅 Stage 4/6: Normalizing publication dates...")
        update_report_progress('Normalization', 'Correcting years from DOI data...', 60)
        
        # Show authority distribution
        tier_counts = Counter(s.get('authority_tier', 'unknown') for s in sources[:20])
        tier_display = ", ".join([f"{k.replace('_', ' ')}: {v}" for k, v in tier_counts.items()])
        st.info(f"📚 Authority distribution: {tier_display}")
        
        # Update session with cleaned sources
        st.session_state.report_research['sources'] = sources
        st.session_state.report_research['filter_metadata'] = fix_metadata
        
        # Stage 5: SOURCE-BOUNDED DRAFT GENERATION
        st.info("✍️ Stage 5/6: Writing with strict source boundaries...")
        update_report_progress('Drafting', 'Generating verified content...', 70)
        
        # Create source boundary prompt to prevent hallucinations
        boundary_prompt = create_source_boundary_prompt(sources, topic, max_sources)
        
        draft, verification_report = generate_draft_strict(
            topic=topic,
            subject=subject,
            subtopics=analysis['subtopics'],
            sources=sources,
            variations=st.session_state.report_research['phrase_variations'],
            evaluation_frameworks=analysis.get('evaluationFrameworks', []),
            max_sources=max_sources,
            boundary_prompt=boundary_prompt  # CRITICAL: Pass boundary prompt
        )
        
        st.session_state.report_draft = draft
        st.session_state.verification_results = verification_report
        
        # Display verification results
        ver = verification_report.get('verification', {})
        cov = verification_report.get('coverage', {})
        
        if ver.get('has_critical'):
            st.error(f"⚠️ {ver.get('total_violations', 0)} critical issues found and corrected")
        elif ver.get('total_violations', 0) > 0:
            st.warning(f"⚠️ {ver.get('total_violations', 0)} minor issues flagged")
        else:
            st.success("✅ All claims verified against sources")
        
        st.info(f"📊 Source coverage: {cov.get('coverage', 0):.1%} ({cov.get('cited_count', 0)}/{cov.get('total_sources', 0)})")
        
        # Stage 6: FINAL REFINEMENT & HTML GENERATION
        st.info("✨ Stage 6/6: Final refinement...")
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
            f"{ver.get('total_violations', 0)} verification issues flagged"
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

# ==============================================
# MAIN APPLICATION
# ==============================================

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<p class="main-header">🔬 SROrch - Scholarly Research Orchestrator & Reviewer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Engine Academic Literature Search, Analysis & Strictly Verified Report Generation</p>', unsafe_allow_html=True)
    
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
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "📊 Results", "📝 Reviewer", "ℹ️ About"])
    
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
            search_button = st.button("🚀 Start Search", type="primary", width='stretch')
        
        with col2:
            if st.button("🔄 Clear Cache", width='stretch'):
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
                    
                    st.info("👉 Switch to the 'Results' tab to view findings, or 'Reviewer' to generate a verified report!")
                    
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
                    
                    # ✅ FIX: Convert numeric columns to proper types
                    numeric_columns = ['citations', 'source_count', 'relevance_score', 'year']
                    for col in numeric_columns:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                    
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
                        # Adding a bookmark indicator column
                        display_df.insert(0, '⭐', display_df.index.map(lambda x: '⭐' if x in st.session_state['bookmarked_papers'] else ''))
                        #display_df.insert(0, ''�', display_df.index.map(lambda x: '⭐' if x in st.session_state['bookmarked_papers'] else ''))
                        
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
                            width='stretch',
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
                            paper_options = {}
                            for idx, row in filtered_df.iterrows():
                                try:
                                    title = str(row.get('title', 'Unknown'))[:50]
                                    paper_options[idx] = f"[{idx}] {title}..."
                                except Exception:
                                    paper_options[idx] = f"[{idx}] Paper"
                            
                            
                            selected_for_bookmark = st.multiselect(
                                "Select papers to bookmark",
                                options=list(paper_options.keys()),
                                format_func=lambda x: paper_options[x],
                                key="bookmark_selector"
                            )
                            
                            bookmark_col1, bookmark_col2 = st.columns(2)
                            
                            with bookmark_col1:
                                if st.button("⭐ Add Bookmarks", width='stretch'):
                                    st.session_state['bookmarked_papers'].update(selected_for_bookmark)
                                    st.success(f"Added {len(selected_for_bookmark)} bookmark(s)!")
                                    st.rerun()
                            
                            with bookmark_col2:
                                if st.button("🗑️ Clear All", width='stretch'):
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
                                        width='stretch'
                                    )
                                
                                with download_col2:
                                    json_data = selected_papers_df.to_json(orient='records', indent=2)
                                    st.download_button(
                                        label=f"📥 JSON ({len(selected_for_download)})",
                                        data=json_data,
                                        file_name=f"selected_{len(selected_for_download)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                        mime="application/json",
                                        width='stretch'
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
                                    width='stretch'
                                )
                            
                            with export_col2:
                                json_data = filtered_df.to_json(orient='records', indent=2)
                                st.download_button(
                                    label="📥 All JSON",
                                    data=json_data,
                                    file_name=f"filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                    width='stretch'
                                )
                    
                    st.divider()
                
                except Exception as e:
                    st.error(f"⚠️ Could not load interactive viewer: {str(e)}")
                    st.info("💡 The data is still available - you can download the CSV file below")
                    # Try to show basic view
                    try:
                        basic_df = pd.read_csv(csv_path)
                        st.dataframe(basic_df.head(20), width='stretch')
                    except:
                        pass
            
            # Display analytics chart
            chart_path = os.path.join(output_dir, "research_analytics.png")
            if os.path.exists(chart_path):
                st.subheader("📈 Research Analytics")
                st.image(chart_path, width='stretch')
                st.divider()
            
            # Results preview
            display_results_preview(results, limit=10)
            
            st.divider()
            
            # Download section
            if output_dir and os.path.exists(output_dir):
                create_download_buttons(output_dir)
        
        else:
            st.info("👈 No results yet. Start a search in the 'Search' tab!")
    
    # ====== TAB 3: REVIEWER ======
    with tab3:
        st.header("📝 Academic Reviewer (Strict Mode)")
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
                
                # Info box
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
                    "🚀 Generate Strictly Verified Report",
                    disabled=not valid,
                    type="primary",
                    width='stretch'
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
                        for i, s in enumerate(st.session_state.report_research['sources'][:NSOURCES], 1):
                            meta = s.get('metadata', {})
                            tier = s.get('authority_tier', 'unknown')
                            tier_emoji = {
                                'top_tier_journal': '🏆',
                                'publisher_journal': '📚',
                                'conference': '🎓',
                                'preprint': '📄',
                                'other': '📃'
                            }.get(tier, '📄')
                            
                            st.markdown(
                                f"**{i}.** {tier_emoji} {meta.get('title', 'Unknown')[:60]}...  "
                                f"({tier.replace('_', ' ').title()})"
                            )
                            st.caption(f"👤 {meta.get('authors', 'Unknown')} | 📅 {meta.get('year', 'N/A')} | 📖 {meta.get('venue', 'N/A')}")
                
                if st.session_state.report_processing:
                    time.sleep(3)
                    st.rerun()
            
            elif st.session_state.report_step == 'complete':
                st.success("✅ Strictly Verified Report Generated Successfully!")
                
                # Show verification badge
                ver_results = st.session_state.get('verification_results', {})
                issues = len(ver_results.get('verification', {}).get('violations', []))
                
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
                    ver_summary = ver_results.get('verification', {})
                    integrity = ver_results.get('coverage', {})
                    specs = ver_results.get('technical_specs_found', {})
                    
                    st.markdown("**Claim Verification:**")
                    st.markdown(f"- Total violations flagged: {ver_summary.get('total_violations', 0)}")
                    if ver_summary.get('by_type'):
                        st.markdown(f"- By type: {dict(ver_summary.get('by_type', {}))}")
                    
                    st.markdown("**Citation Coverage:**")
                    st.markdown(f"- Coverage: {integrity.get('coverage', 0):.1%}")
                    st.markdown(f"- Sources cited: {integrity.get('cited_count', 0)}/{integrity.get('total_sources', 0)}")
                    
                    st.markdown("**Technical Specs Extracted:**")
                    for key, values in specs.items():
                        st.markdown(f"- {key}: {', '.join(values[:5])}")
                    
                    if ver_summary.get('violations'):
                        st.markdown("**Sample Violations:**")
                        for v in ver_summary['violations'][:5]:
                            st.caption(f"[{v['type']}] {v.get('suggestion', v.get('issue', ''))}")

                st.markdown("---")
                
                # Download
                col1, col2 = st.columns(2)
                with col1:
                    if st.session_state.report_html:
                        filename = f"{st.session_state.report_form_data['topic'].replace(' ', '_')}_Strict_Report.html"
                        st.download_button(
                            "📥 Download Verified HTML Report",
                            data=st.session_state.report_html,
                            file_name=filename,
                            mime="text/html",
                            type="primary",
                            width='stretch'
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
                        
                        tier_badge = {
                            'top_tier_journal': '🏆 Nature/Science',
                            'publisher_journal': '📚 Journal',
                            'conference': '🎓 Conference',
                            'preprint': '📄 Preprint'
                        }.get(tier, '📄 Other')
                        
                        st.markdown(f"**[{i}]** {tier_badge} {meta.get('title', 'N/A')[:70]}")
                        st.caption(f"👤 {meta.get('authors', 'N/A')} | 📅 {meta.get('year', 'N/A')} | 📖 {meta.get('venue', 'N/A')}")
                        st.caption(f"🔗 [{s['url']}]({s['url']})")
                        if orch.get('source_count'):
                            st.caption(f"✓ Found in {orch['source_count']} database(s) | 📊 {orch.get('citations', 0)} citations")
                        st.divider()
                
                if st.button("🔄 Generate Another Report", type="secondary", width='stretch'):
                    reset_report_system()
                    st.rerun()
            
            elif st.session_state.report_step == 'error':
                st.error("❌ Error Occurred")
                st.warning(st.session_state.report_progress['detail'])
                
                if st.session_state.report_execution_time:
                    exec_mins = int(st.session_state.report_execution_time // 60)
                    exec_secs = int(st.session_state.report_execution_time % 60)
                    st.caption(f"Failed after {exec_mins}m {exec_secs}s")
                
                if st.button("🔄 Try Again", type="primary", width='stretch'):
                    reset_report_system()
                    st.rerun()
    
    # ====== TAB 4: ABOUT ======
    with tab4:
        st.header("About SROrch")
        
        st.markdown("""
        ### 🔬 Scholarly Research Orchestrator & Reviewer
        
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
        
        **Reviewer:**
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
        3. Click "Generate Strictly Verified Report"
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
        
        **Version:** 2.1 - Enhanced with Strict Claim Verification  
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
Report Generation: Claude Sonnet 4.5 with strict verification
Claim Verification: Enabled (fuzzy matching ±5%)
Source Ranking: Authority-tier based (Nature/Science > Conference > Preprint)
Forbidden Terms: {len(FORBIDDEN_GENERIC_TERMS)} generic words blocked
            """)

if __name__ == "__main__":
    main()
