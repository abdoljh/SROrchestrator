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

def get_clinical_gap_patterns():
    """Specific patterns for Clinical and Medical research gaps."""
    return [
        # Clinical Trial & Evidence Gaps
        r"(?:lack|absence) of (?:randomized|controlled|prospective) (?:trials|studies)",
        r"limited (?:clinical|real-world) (?:evidence|data|validation)",
        r"(?:small|limited|insufficient) (?:sample size|cohort|patient population)",
        r"(?:short|limited) (?:follow-up|observation) (?:period|duration)",
        r"heterogeneity (?:in|of) (?:patient|treatment|study) (?:populations|protocols)",
        
        # Treatment & Intervention Gaps
        r"optimal (?:dose|dosage|treatment|regimen) (?:remains|is) (?:unclear|undetermined|unknown)",
        r"(?:no|limited) (?:standardized|consensus|established) (?:guidelines|protocols|criteria)",
        r"efficacy (?:in|across) (?:different|diverse) (?:populations|settings) (?:is unclear|remains unknown)",
        r"long-term (?:safety|efficacy|outcomes) (?:not|remain) (?:established|evaluated|assessed)",
        
        # Diagnostic & Biomarker Gaps
        r"lack of (?:validated|reliable|specific) (?:biomarkers|diagnostic criteria)",
        r"(?:sensitivity|specificity) (?:needs|requires) (?:improvement|validation)",
        r"early (?:detection|diagnosis) (?:remains|is) (?:challenging|difficult)",
    ]

def get_general_gap_patterns():
    """General academic research gap patterns applicable across disciplines."""
    return [
        # Future Research Needs
        r"(?:further|future|additional) (?:research|studies|investigations|work) (?:is|are|will be|should be) (?:needed|required|warranted|necessary)",
        r"(?:more|additional) (?:research|studies|data) (?:is|are) (?:needed|required)",
        r"future (?:work|directions|studies) (?:should|will|could) (?:focus on|address|examine|explore|investigate)",
        r"(?:warrants|requires|merits) (?:further|additional|more) (?:investigation|exploration|research|study)",
        
        # Knowledge Gaps
        r"(?:remains|is) (?:poorly|not fully|incompletely|inadequately) (?:understood|characterized|defined|elucidated)",
        r"(?:little|limited) (?:is|has been) (?:known|understood|reported) (?:about|regarding|concerning)",
        r"(?:limited|scarce|insufficient) (?:evidence|data|information|knowledge) (?:exists|is available)",
        r"(?:unclear|unknown|uncertain) (?:whether|if|how)",
        
        # Unexplored Areas
        r"has (?:not|never|rarely) been (?:investigated|explored|studied|examined|addressed)",
        r"(?:no|few|limited) (?:studies|investigations|research) (?:have|has) (?:examined|investigated|explored|addressed)",
        r"(?:remains|is) (?:unexplored|unexamined|uninvestigated)",
        r"potential (?:area|avenue|direction) for (?:future|additional) (?:investigation|research|exploration)",
        
        # Evidence & Data Gaps
        r"(?:lack|dearth|absence|paucity) of (?:evidence|studies|data|research|literature)",
        r"(?:limited|sparse|insufficient) (?:evidence|data) (?:exists|is available|to support)",
        r"(?:no|limited) (?:empirical|experimental|quantitative) (?:evidence|data|studies)",
        r"data (?:scarcity|limitations|gaps|are lacking)",
        
        # Conflicting & Inconsistent Findings
        r"(?:conflicting|inconsistent|contradictory|mixed) (?:results|evidence|findings|reports)",
        r"(?:controversial|debated|disputed) (?:findings|results|evidence)",
        r"(?:lack|absence) of (?:consensus|agreement) (?:on|regarding|about)",
        
        # Methodological Gaps
        r"(?:methodological|measurement|analytical) (?:limitations|challenges|issues)",
        r"(?:lack|absence) of (?:standardized|validated|reliable) (?:methods|measures|tools|instruments)",
        r"generalizability (?:is|remains) (?:limited|uncertain|unclear)",
        
        # Study Limitations
        r"(?:limitation|constraint)s? of (?:this|the|our) (?:study|research|investigation|work)",
        r"(?:this|our) (?:study|research) (?:has|had) (?:several|some|certain) (?:limitations|constraints)",
        r"(?:caution|care) (?:should|must) be (?:exercised|taken) when (?:interpreting|generalizing)",
        
        # Open Questions
        r"(?:unresolved|open|outstanding) (?:issues|questions|problems|challenges)",
        r"(?:key|important|critical) (?:questions|issues) (?:remain|are) (?:unanswered|unresolved|open)",
        r"it (?:remains|is) (?:unclear|unknown|uncertain|to be determined)",
        
        # Gaps in Understanding
        r"(?:mechanisms|pathways|processes) (?:underlying|behind) .{1,50} (?:remain|are) (?:unclear|unknown|poorly understood)",
        r"(?:how|why|whether) .{1,50} (?:remains|is) (?:unclear|unknown|poorly understood)",
    ]

def analyze_research_gaps(results, query=""):
    """
    Enhanced analysis of research gaps with domain-specific pattern matching.
    Returns structured data about gaps, keywords, and categories.
    """
    # Combine all patterns
    patterns = get_general_gap_patterns()
    
    # Add domain-specific patterns based on query
    tech_terms = ["ai", "machine learning", "deep learning", "algorithm", "neural", "network", "model", "computational"]
    clinical_terms = ["patient", "clinical", "treatment", "therapy", "disease", "diagnosis", "medical", "hospital", "health"]
    
    query_lower = query.lower()
    pattern_types = []
    
    if any(term in query_lower for term in tech_terms):
        patterns.extend(get_ai_gap_patterns())
        pattern_types.append("Technical/AI")
    
    if any(term in query_lower for term in clinical_terms):
        patterns.extend(get_clinical_gap_patterns())
        pattern_types.append("Clinical/Medical")
    
    found_gaps = []
    all_keywords = []
    gap_categories = Counter()
    
    # Analyze papers that have abstracts or TLDRs
    papers_with_text = [p for p in results if p.get('abstract') or p.get('tldr')]
    
    for paper in papers_with_text:
        # Combine fields for analysis
        text = f"{paper.get('tldr', '')} {paper.get('abstract', '')}".lower()
        
        # Split into sentences to isolate specific gap statements
        sentences = re.split(r'(?<=[.!?]) +', text)
        
        for sentence in sentences:
            # Skip very short sentences (likely not meaningful gaps)
            if len(sentence.split()) < 5:
                continue
                
            for pattern in patterns:
                if re.search(pattern, sentence):
                    # Categorize the gap type
                    category = categorize_gap(sentence, pattern)
                    gap_categories[category] += 1
                    
                    found_gaps.append({
                        'title': paper.get('title', 'Unknown Title'),
                        'gap_statement': sentence.strip(),
                        'year': paper.get('year', 'N/A'),
                        'category': category,
                        'citations': paper.get('citations', 0)
                    })
                    break  # Don't record same sentence twice

        # Extract keywords for theme analysis
        if paper.get('keywords'):
            all_keywords.extend([k.strip().lower() for k in paper['keywords'].split(',')])

    # Keyword frequency analysis
    common_keywords = Counter(all_keywords).most_common(15)
    
    # Sort gaps by citation count (high-impact gaps)
    sorted_gaps = sorted(found_gaps, key=lambda x: int(x.get('citations', 0)), reverse=True)
    
    return {
        'total_gaps_found': len(found_gaps),
        'gap_list': sorted_gaps,
        'top_keywords': common_keywords,
        'gap_categories': dict(gap_categories),
        'pattern_types_used': pattern_types if pattern_types else ["General"],
        'papers_analyzed': len(papers_with_text)
    }

def categorize_gap(sentence, pattern):
    """Categorize a gap statement into a specific type."""
    sentence_lower = sentence.lower()
    
    # Future research category
    if any(term in sentence_lower for term in ['future', 'further', 'additional', 'more research', 'future work']):
        return "Future Research Needed"
    
    # Knowledge gap category
    if any(term in sentence_lower for term in ['poorly understood', 'unclear', 'unknown', 'not known', 'little is known']):
        return "Knowledge Gap"
    
    # Data/Evidence gap category
    if any(term in sentence_lower for term in ['lack of evidence', 'lack of data', 'limited data', 'insufficient evidence', 'data scarcity']):
        return "Data/Evidence Gap"
    
    # Methodological gap category
    if any(term in sentence_lower for term in ['limitation', 'limited by', 'methodological', 'measurement', 'sample size']):
        return "Methodological Limitation"
    
    # Clinical/Treatment gap category
    if any(term in sentence_lower for term in ['treatment', 'therapy', 'clinical', 'patient', 'trial', 'efficacy']):
        return "Clinical/Treatment Gap"
    
    # Technical/Performance gap category
    if any(term in sentence_lower for term in ['performance', 'accuracy', 'computational', 'efficiency', 'scalability']):
        return "Technical/Performance Gap"
    
    # Conflicting evidence category
    if any(term in sentence_lower for term in ['conflicting', 'inconsistent', 'contradictory', 'controversial']):
        return "Conflicting Evidence"
    
    # Unexplored area category
    if any(term in sentence_lower for term in ['not been investigated', 'unexplored', 'never been studied', 'rarely studied']):
        return "Unexplored Area"
    
    # Default category
    return "Other Gap"

def generate_gap_summary_stats(gap_data):
    """Generate statistical summary of research gaps for reporting."""
    stats = {
        'total_gaps': gap_data['total_gaps_found'],
        'papers_with_gaps': len(set(g['title'] for g in gap_data['gap_list'])),
        'avg_gaps_per_paper': round(gap_data['total_gaps_found'] / max(gap_data['papers_analyzed'], 1), 2),
        'categories': gap_data['gap_categories'],
        'high_impact_gaps': len([g for g in gap_data['gap_list'] if int(g.get('citations', 0)) >= 50]),
        'recent_gaps': len([g for g in gap_data['gap_list'] if str(g.get('year', '')) >= str(2020)])
    }
    return stats