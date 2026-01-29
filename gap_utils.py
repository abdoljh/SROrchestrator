import re
from collections import Counter

def analyze_research_gaps(results):
    """
    Analyzes top papers for explicit research gaps and emerging trends.
    Returns a dictionary of found gaps and keyword frequency.
    """
    # 1. Phrases that explicitly signal a research gap
    GAP_PATTERNS = [
        r"(?:further|future) research (?:is|will be) (?:needed|required|necessary)",
        r"(?:remains|is) poorly understood",
        r"little (?:is known|research has been done)",
        r"has (?:not|never) been (?:investigated|explored|studied)",
        r"(?:lack|dearth) of (?:evidence|studies|data)",
        r"conflicting (?:results|evidence|findings)",
        r"unresolved issues",
        r"the limitation(?:s)? of (?:this|the) study",
        r"future work should focus on",
        r"potential area for future investigation"
    ]

    found_gaps = []
    all_keywords = []
    
    # We only analyze papers that have abstracts/TLDRs
    papers_with_text = [p for p in results if p.get('abstract') or p.get('tldr')]
    
    for paper in papers_with_text:
        text = f"{paper.get('tldr', '')} {paper.get('abstract', '')}".lower()
        
        # Extract specific gap sentences
        sentences = re.split(r'(?<=[.!?]) +', text)
        for sentence in sentences:
            for pattern in GAP_PATTERNS:
                if re.search(pattern, sentence):
                    found_gaps.append({
                        'title': paper['title'],
                        'gap_statement': sentence.strip(),
                        'year': paper.get('year')
                    })
                    break # Don't record same sentence twice for different patterns

        # Track keywords for "Hot Topics"
        if paper.get('keywords'):
            all_keywords.extend([k.strip().lower() for k in paper['keywords'].split(',')])

    # 2. Identify emerging vs. dying topics (if multiple years exist)
    common_keywords = Counter(all_keywords).most_common(10)
    
    return {
        'total_gaps_found': len(found_gaps),
        'gap_list': found_gaps,
        'top_keywords': common_keywords
    }
