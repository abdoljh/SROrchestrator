"""
SROrch Critical Fixes Module
Implements all improvements identified in critical analysis
"""

import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict


# ================================================================================
# FIX 1: SOURCE QUALITY FILTERING
# ================================================================================

class SourceQualityFilter:
    """
    Multi-tier filtering to eliminate irrelevant sources.
    Addresses: 'Greek Australian Literature', off-topic papers, future years.
    """
    
    # Blacklist: Venues irrelevant to CS/AI/ML/RAG
    BLACKLISTED_VENUES = {
        'journal of the australasian universities language and literature association',
        'language and literature: international journal of stylistics',
        'journal of language, literature and culture',
        'australasian universities language and literature association',
        'greek australian literature',
        'czech literature abroad',
        'parable',
        'professionalisation',
    }
    
    # Off-topic keywords (humanities/literature)
    OFFTOPIC_INDICATORS = [
        'literary analysis', 'poetry', 'shakespeare', 'novelistic',
        'cultural discourse', 'stylistics', 'multilingual australian literature',
        'book review', 'parable', 'professionalisation'
    ]
    
    # CS/AI indicators that override off-topic keywords
    CS_INDICATORS = [
        'retrieval', 'language model', 'neural', 'embedding', 'transformer',
        'synthesis', 'benchmark', 'dataset', 'rag', 'llm', 'generation',
        'scientific literature', 'citation', 'abstract'
    ]
    
    MIN_RELEVANCE_SCORE = 200
    
    @classmethod
    def filter_sources(cls, sources: List[Dict], topic: str = "") -> Tuple[List[Dict], List[str]]:
        """Apply comprehensive filtering."""
        filtered = []
        rejections = []
        
        for i, source in enumerate(sources):
            meta = source.get('metadata', {})
            venue = str(meta.get('venue', '')).lower()
            title = str(meta.get('title', '')).lower()
            abstract = str(source.get('content', '')).lower()
            relevance = source.get('relevance_score', 0)
            
            # Tier 1: Relevance threshold
            if relevance < cls.MIN_RELEVANCE_SCORE:
                rejections.append(f"[{i}] Low relevance ({relevance}): {title[:50]}...")
                continue
            
            # Tier 2: Blacklisted venue
            if any(blacklisted in venue for blacklisted in cls.BLACKLISTED_VENUES):
                rejections.append(f"[{i}] Blacklisted venue: {venue[:60]}")
                continue
            
            # Tier 3: Off-topic with CS override
            combined = f"{title} {abstract}"
            off_topic = sum(1 for kw in cls.OFFTOPIC_INDICATORS if kw in combined)
            cs_score = sum(1 for kw in cls.CS_INDICATORS if kw in combined)
            
            if off_topic > 0 and cs_score == 0:
                rejections.append(f"[{i}] Off-topic (score: {off_topic}): {title[:50]}...")
                continue
            
            # Tier 4: Temporal sanity
            year = str(meta.get('year', ''))
            if year.isdigit():
                year_int = int(year)
                current = datetime.now().year
                if year_int > current + 1:
                    url = source.get('url', '').lower()
                    if 'arxiv' not in url:
                        rejections.append(f"[{i}] Future year {year_int}: {title[:50]}...")
                        continue
            
            filtered.append(source)
        
        print(f"Source filtering: {len(sources)} → {len(filtered)} ({len(sources) - len(filtered)} removed)")
        if rejections:
            print(f"Sample rejections: {rejections[:3]}")
        
        return filtered, rejections


# ================================================================================
# FIX 2: TEMPORAL NORMALIZATION
# ================================================================================

def normalize_publication_year(source: Dict) -> str:
    """Normalize year using DOI priority. Fixes 2026→2025."""
    meta = source.get('metadata', {})
    current_year = str(meta.get('year', ''))
    doi = str(meta.get('doi', '')).strip()
    url = source.get('url', '').lower()
    
    # Priority 1: DOI extraction (Nature: 10.1038/s41586-025-...)
    if doi and doi.upper() not in ('N/A', 'NA', '', 'NONE'):
        # Nature/Science pattern
        nature_match = re.search(r'[-\.](\d{2})\d{2}[-\.]', doi)
        if nature_match:
            prefix = nature_match.group(1)
            century = '20' if int(prefix) < 50 else '19'
            doi_year = century + prefix
            
            if current_year.isdigit() and abs(int(current_year) - int(doi_year)) > 1:
                print(f"Year fix: {current_year} → {doi_year} (from DOI)")
                return doi_year
            return doi_year
        
        # arXiv DOI: 10.48550/arxiv.2411.14199
        arxiv_match = re.search(r'arxiv\.(\d{2})(\d{2})', doi)
        if arxiv_match:
            return '20' + arxiv_match.group(1)
    
    # Priority 2: arXiv URL
    if 'arxiv.org' in url:
        url_match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d{2})(\d{2})', url)
        if url_match:
            arxiv_year = '20' + url_match.group(1)
            if current_year.isdigit() and abs(int(current_year) - int(arxiv_year)) > 1:
                print(f"Year fix: {current_year} → {arxiv_year} (from arXiv)")
                return arxiv_year
            return arxiv_year
    
    # Priority 3: Sanity check current year
    if current_year.isdigit():
        year_int = int(current_year)
        now = datetime.now().year
        if year_int > now + 1:
            return str(now)
        return current_year
    
    return 'N/A'


def apply_year_normalization(sources: List[Dict]) -> List[Dict]:
    """Apply year normalization to all sources."""
    for source in sources:
        normalized = normalize_publication_year(source)
        if normalized != 'N/A':
            source['metadata']['year'] = normalized
    return sources


# ================================================================================
# FIX 3: SOURCE BOUNDARY PROMPT
# ================================================================================

def create_source_boundary_prompt(sources: List[Dict], topic: str, max_sources: int = 25) -> str:
    """Create strict boundary to prevent hallucinated citations."""
    entries = []
    
    for i, s in enumerate(sources[:max_sources], 1):
        meta = s.get('metadata', {})
        title = meta.get('title', 'Unknown')[:60]
        year = meta.get('year', 'N/A')
        venue = meta.get('venue', 'Unknown')[:30]
        
        # Extract metrics
        content = s.get('content', '')
        metrics = []
        pcts = re.findall(r'(\d+(?:\.\d+)?)%', content)
        if pcts:
            metrics.append(f"{pcts[0]}%")
        sizes = re.findall(r'(\d+(?:\.\d+)?)\s*(M|B)', content, re.I)
        if sizes:
            metrics.append(f"{sizes[0][0]}{sizes[0][1]}")
        
        metric_str = f" [{' | '.join(metrics)}]" if metrics else ""
        entries.append(f"[{i}] {title}... ({venue}, {year}){metric_str}")
    
    return f"""
╔══════════════════════════════════════════════════════════════════╗
║              STRICT SOURCE BOUNDARY - MANDATORY                  ║
╚══════════════════════════════════════════════════════════════════╝

Topic: "{topic}"

You may ONLY cite these {len(sources[:max_sources])} sources using [1]-[{len(sources[:max_sources])}]:
{chr(10).join(entries)}

RULES:
1. CITE ONLY sources above
2. DO NOT mention SciFact, MultiXScience, or benchmarks not listed
3. DO NOT cite by author name (use [number] only)
4. If info unavailable, write "Not specified in retrieved sources"

VIOLATIONS (will be rejected):
❌ "evaluated on SciFact [26]" → Not in list
❌ "Smith et al. (2023) show..." → Use [X] format

CORRECT:
✅ "achieves 71.8% [5]" → [5] is in list above
"""


# ================================================================================
# FIX 4: ENHANCED VERIFICATION
# ================================================================================

class AlignedClaimVerifier:
    """Verifier aligned with generation data to eliminate false negatives."""
    
    def __init__(self, sources: List[Dict]):
        self.sources = sources
        self.source_count = len(sources)
        self.metrics_cache = self._build_metrics_cache()
    
    def _build_metrics_cache(self) -> Dict:
        """Pre-extract metrics from all sources."""
        cache = {}
        for i, source in enumerate(self.sources, 1):
            text = ' '.join([
                str(source.get('metadata', {}).get('title', '')),
                str(source.get('content', '')),
                str(source.get('_orchestrator_data', {}).get('abstract', '')),
                str(source.get('_orchestrator_data', {}).get('tldr', ''))
            ])
            
            cache[str(i)] = {
                'numbers': re.findall(r'\b(\d+(?:\.\d+)?)\b', text),
                'percentages': re.findall(r'(\d+(?:\.\d+)?)%', text),
                'years': list(set(re.findall(r'\b(20[0-2]\d)\b', text))),
            }
        return cache
    
    def is_valid_citation(self, num: int) -> bool:
        return 1 <= num <= self.source_count
    
    def verify_draft(self, draft: Dict) -> Dict:
        """Comprehensive verification."""
        violations = []
        cited = set()
        
        sections = {
            'abstract': draft.get('abstract', ''),
            'introduction': draft.get('introduction', ''),
            'literatureReview': draft.get('literatureReview', ''),
            'dataAnalysis': draft.get('dataAnalysis', ''),
            'challenges': draft.get('challenges', ''),
            'futureOutlook': draft.get('futureOutlook', ''),
            'conclusion': draft.get('conclusion', '')
        }
        
        for section_name, content in sections.items():
            if not isinstance(content, str):
                continue
            
            for match in re.finditer(r'\[(\d+)\]', content):
                citation_num = match.group(1)
                int_num = int(citation_num)
                cited.add(int_num)
                
                # Check validity
                if not self.is_valid_citation(int_num):
                    violations.append({
                        'type': 'invalid_citation',
                        'severity': 'CRITICAL',
                        'section': section_name,
                        'citation': citation_num,
                        'issue': f'Citation [{citation_num}] not found (max: {self.source_count})'
                    })
                    continue
                
                # Check numbers in context
                context = content[max(0, match.start()-80):min(len(content), match.end()+80)]
                numbers = re.findall(r'(\d+(?:\.\d+)?)\s*(?:%|percent)?', context)
                source_data = self.metrics_cache.get(citation_num, {})
                source_nums = source_data.get('numbers', []) + source_data.get('percentages', [])
                
                for num in numbers:
                    if num not in source_nums and not self._is_acceptable(num):
                        violations.append({
                            'type': 'unsupported_number',
                            'severity': 'WARNING',
                            'section': section_name,
                            'citation': citation_num,
                            'number': num,
                            'issue': f'Number {num} not in source [{citation_num}]'
                        })
        
        return {
            'total_violations': len(violations),
            'violations': violations,
            'has_critical': any(v['severity'] == 'CRITICAL' for v in violations),
            'coverage': len(cited) / self.source_count if self.source_count else 0,
            'cited_count': len(cited),
            'total_sources': self.source_count
        }
    
    def _is_acceptable(self, num: str) -> bool:
        """Check if number needs no source (years, small ints)."""
        try:
            n = float(num)
            # Years 1950-2030 (publication years)
            if 1950 <= n <= 2030:
                return True
            # Small structural numbers
            if n <= 10:
                return True
            return False
        except:
            return False

    def _is_acceptable(self, num: str, context: str = "") -> bool:
        """Check if number needs no source verification."""
        try:
            n = float(num)
            
            # Publication years (common in citations)
            if 1950 <= n <= 2030:
                # Additional check: is this likely a year in context?
                if "year" in context.lower() or int(n) == n:  # Whole numbers likely years
                    return True
            
            # Small structural numbers (counts up to 10)
            if n <= 10 and n == int(n):
                return True
                
            return False
        except (ValueError, TypeError):
            return False
    
    def verify_draft(self, draft: Dict) -> Dict:
        """Comprehensive verification with context-aware exemptions."""
        violations = []
        cited = set()
        
        sections = {
            'abstract': draft.get('abstract', ''),
            'introduction': draft.get('introduction', ''),
            'literatureReview': draft.get('literatureReview', ''),
            'dataAnalysis': draft.get('dataAnalysis', ''),
            'challenges': draft.get('challenges', ''),
            'futureOutlook': draft.get('futureOutlook', ''),
            'conclusion': draft.get('conclusion', '')
        }
        
        for section_name, content in sections.items():
            if not isinstance(content, str):
                continue
            
            for match in re.finditer(r'\[(\d+)\]', content):
                citation_num = match.group(1)
                int_num = int(citation_num)
                cited.add(int_num)
                
                # Check validity
                if not self.is_valid_citation(int_num):
                    violations.append({
                        'type': 'invalid_citation',
                        'severity': 'CRITICAL',
                        'section': section_name,
                        'citation': citation_num,
                        'issue': f'Citation [{citation_num}] not found (max: {self.source_count})'
                    })
                    continue
                
                # Check numbers in context (with surrounding text for context)
                context_start = max(0, match.start() - 80)
                context_end = min(len(content), match.end() + 80)
                context = content[context_start:context_end]
                
                numbers = re.findall(r'(\d+(?:\.\d+)?)\s*(?:%|percent)?', context)
                source_data = self.metrics_cache.get(citation_num, {})
                source_nums = source_data.get('numbers', []) + source_data.get('percentages', [])
                
                for num in numbers:
                    # Skip if acceptable (years, small ints)
                    if self._is_acceptable(num, context):
                        continue
                        
                    if num not in source_nums:
                        violations.append({
                            'type': 'unsupported_number',
                            'severity': 'WARNING',
                            'section': section_name,
                            'citation': citation_num,
                            'number': num,
                            'issue': f'Number {num} not in source [{citation_num}]'
                        })
        
        return {
            'total_violations': len(violations),
            'violations': violations,
            'has_critical': any(v['severity'] == 'CRITICAL' for v in violations),
            'coverage': len(cited) / self.source_count if self.source_count else 0,
            'cited_count': len(cited),
            'total_sources': self.source_count
        }

# ================================================================================
# INTEGRATION HELPERS
# ================================================================================

def integrate_fixes_into_pipeline(raw_sources: List[Dict], topic: str) -> Tuple[List[Dict], Dict]:
    """
    Apply all fixes to source list. Returns (cleaned_sources, metadata).
    """
    metadata = {'original_count': len(raw_sources)}
    
    # Fix 1: Quality filtering
    filtered, rejections = SourceQualityFilter.filter_sources(raw_sources, topic)
    metadata['filtered_count'] = len(filtered)
    metadata['rejection_reasons'] = rejections[:10]
    
    # Fallback if too aggressive
    if len(filtered) < 5:
        filtered = raw_sources[:10]
        metadata['fallback_used'] = True
    
    # Fix 2: Year normalization
    filtered = apply_year_normalization(filtered)
    year_fixes = sum(1 for s in filtered 
                     if normalize_publication_year(s) != s.get('metadata', {}).get('year', ''))
    metadata['year_corrections'] = year_fixes
    
    return filtered, metadata
