import re
from collections import Counter

def analyze_research_gaps(results):
    """
    Analyzes top papers for explicit research gaps and emerging trends.
    Returns a dictionary of found gaps and keyword frequency.
    """
    # 1. Phrases that explicitly signal a research gap
    # Expanded patterns specifically for clinical and high-level academic gaps
    GAP_PATTERNS = [
        # Explicit Gaps
        r"(?:further|future) (?:research|studies|investigations) (?:is|are|will be) (?:needed|required|warranted|necessary)",
        r"(?:remains|is) (?:poorly|not fully|incompletely) (?:understood|defined|elucidated)",
        r"(?:lack|dearth|scarcity) of (?:consensus|evidence|long-term data|prospective studies)",
        
        # Clinical/Medical Specific Gaps
        r"optimal (?:management|treatment|therapy) (?:remains|is) (?:controversial|unknown|unclear)",
        r"(?:randomized|controlled) trials (?:are|have not been) (?:lacking|conducted)",
        r"standard (?:of care|protocols) (?:has|have) (?:not|yet) (?:to be|been) established",
        
        # Methodological Limitations
        r"(?:small|limited) sample size",
        r"retrospective (?:nature|design) (?:of the|of this) study",
        r"results (?:should|must) be (?:interpreted|treated) with caution",
        r"not (?:statistically significant|generalisable to|representative of)",
        
        # Direct Future Work
        r"future (?:work|directions) (?:should|will) focus on",
        r"area(?:s)? for (?:potential|future) (?:investigation|exploration)"

        # Performance & Benchmark Gaps
        r"(?:remains|is) (?:a challenge|an open problem|unsolved)",
        r"(?:fails|struggles) to (?:generalize|capture|scale)",
        r"(?:performance|accuracy) (?:degrades|drops|plateaus) (?:when|at)",
        r"unable to (?:reach|achieve|match) (?:human-level|SOTA|state-of-the-art)",
        r"saturated (?:benchmarks|datasets)",

        # Computational & Efficiency Gaps (Common in AI)
        r"computationally (?:expensive|prohibitive|demanding)",
        r"limited by (?:hardware|GPU|memory|VRAM)",
        r"high (?:inference|training) cost",
        r"lack of (?:energy-efficient|real-time) (?:implementation|processing)",

        # Data & Robustness Gaps
        r"data (?:scarcity|sparsity|imbalance)",
        r"dependence on (?:large-scale|labeled|curated) datasets",
        r"(?:vulnerable|susceptible) to (?:adversarial|noise|out-of-distribution)",
        r"black-box (?:nature|model)",
        r"lack of (?:interpretability|explainability|transparency)",

        # Societal & Ethical Gaps (SAGE/SSRN patterns)
        r"(?:inherent|algorithmic) bias",
        r"ethical (?:implications|concerns|dilemmas)",
        r"long-term (?:impacts|consequences) (?:remain|are) (?:unknown|unexplored)",
        r"lack of (?:policy|regulatory|governance) frameworks"

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
