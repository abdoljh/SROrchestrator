import re
from collections import Counter

def get_ai_gap_patterns():
    """Specific patterns for Technical, AI, and Engineering gaps."""
    return [
        # Performance & Benchmark Gaps
        r"(?:remains|is) (?:a challenge|an open problem|unsolved)",
        r"(?:fails|struggles) to (?:generalize|capture|scale)",
        r"(?:performance|accuracy) (?:degrades|drops|plateaus) (?:when|at)",
        r"unable to (?:reach|achieve|match) (?:human-level|SOTA|state-of-the-art)",
        r"saturated (?:benchmarks|datasets)",

        # Computational & Efficiency Gaps
        r"computationally (?:expensive|prohibitive|demanding)",
        r"limited by (?:hardware|gpu|memory|vram)",
        r"high (?:inference|training) cost",
        r"lack of (?:energy-efficient|real-time) (?:implementation|processing)",

        # Data & Robustness Gaps
        r"data (?:scarcity|sparsity|imbalance)",
        r"dependence on (?:large-scale|labeled|curated) datasets",
        r"(?:vulnerable|susceptible) to (?:adversarial|noise|out-of-distribution)",
        r"black-box (?:nature|model)",
        r"lack of (?:interpretability|explainability|transparency)"
    ]

def analyze_research_gaps(results, query=""):
    """
    Analyzes top papers for explicit research gaps and emerging trends.
    Now supports conditional Technical/AI pattern injection.
    """
    # 1. Base patterns for clinical and academic gaps
    patterns = [
        r"(?:further|future) (?:research|studies|investigations) (?:is|are|will be) (?:needed|required|warranted|necessary)",
        r"(?:remains|is) (?:poorly|not fully|incompletely) (?:understood|defined|elucidated)",
        r"(?:lack|dearth|scarcity) of (?:consensus|evidence|long-term data|prospective studies)",
        r"optimal (?:management|treatment|therapy) (?:remains|is) (?:controversial|unknown|unclear)",
        r"future (?:work|directions) (?:should|will) focus on",
        r"area(?:s)? for (?:potential|future) (?:investigation|exploration)"
    ]

    # 2. Inject AI patterns if query is technical
    tech_terms = ["ai", "machine learning", "deep learning", "algorithm", "neural", "network", "model"]
    if any(term in query.lower() for term in tech_terms):
        patterns.extend(get_ai_gap_patterns())
        
    found_gaps = []
    all_keywords = []
    
    # Analyze papers that have abstracts or TLDRs
    papers_with_text = [p for p in results if p.get('abstract') or p.get('tldr')]
    
    for paper in papers_with_text:
        # Combine fields for analysis
        text = f"{paper.get('tldr', '')} {paper.get('abstract', '')}".lower()
        
        # Split into sentences to isolate the specific gap statement
        sentences = re.split(r'(?<=[.!?]) +', text)
        for sentence in sentences:
            for pattern in patterns:
                if re.search(pattern, sentence):
                    found_gaps.append({
                        'title': paper.get('title', 'Unknown Title'),
                        'gap_statement': sentence.strip(),
                        'year': paper.get('year', 'N/A')
                    })
                    break 

        if paper.get('keywords'):
            all_keywords.extend([k.strip().lower() for k in paper['keywords'].split(',')])

    common_keywords = Counter(all_keywords).most_common(10)
    
    return {
        'total_gaps_found': len(found_gaps),
        'gap_list': found_gaps,
        'top_keywords': common_keywords
    }

