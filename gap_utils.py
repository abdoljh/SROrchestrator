import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set
import math

# ==================================================
# 1. ENHANCED PATTERN LIBRARIES WITH CONTEXTUAL AWARENESS
# ==================================================

def get_ai_gap_patterns():
    """Enhanced patterns for Technical, AI, and Engineering gaps."""
    return {
        # Performance & Benchmark Gaps
        r"(?:remains|is) (?:a challenge|an open problem|unsolved)": "Performance Gap",
        r"(?:fails|struggles) to (?:generalize|capture|scale)": "Generalization Gap",
        r"(?:performance|accuracy) (?:degrades|drops|plateaus) (?:when|at)": "Performance Degradation",
        r"unable to (?:reach|achieve|match) (?:human-level|SOTA|state-of-the-art)": "SOTA Gap",
        r"saturated (?:benchmarks|datasets)": "Benchmark Saturation",
        r"(?:bottleneck|limitation) in (?:computational|processing|training) (?:efficiency|speed)": "Efficiency Bottleneck",
        
        # Computational & Efficiency Gaps
        r"computationally (?:expensive|prohibitive|demanding)": "Computational Cost",
        r"limited by (?:hardware|gpu|memory|vram|compute)": "Hardware Limitation",
        r"high (?:inference|training) cost": "Cost Barrier",
        r"lack of (?:energy-efficient|real-time) (?:implementation|processing)": "Energy/Real-time Gap",
        r"scalability (?:issues|challenges|concerns)": "Scalability Gap",
        
        # Data & Robustness Gaps
        r"data (?:scarcity|sparsity|imbalance|paucity)": "Data Scarcity",
        r"dependence on (?:large-scale|labeled|curated|annotated) datasets": "Data Dependency",
        r"(?:vulnerable|susceptible) to (?:adversarial|noise|out-of-distribution|ood)": "Robustness Gap",
        r"black-box (?:nature|model|behavior)": "Interpretability Gap",
        r"lack of (?:interpretability|explainability|transparency|understanding)": "Explainability Gap",
        r"domain (?:shift|adaptation|generalization) (?:remains|is)": "Domain Adaptation Gap",
        
        # Ethical & Bias Gaps
        r"(?:bias|fairness|ethical) (?:issues|concerns|challenges)": "Ethical/Bias Gap",
        r"(?:demographic|gender|racial) (?:bias|disparity)": "Demographic Bias",
        r"lack of (?:diversity|representation) in (?:data|training|datasets)": "Representation Gap"
    }

def get_clinical_gap_patterns():
    """Enhanced patterns for Clinical and Medical research gaps."""
    return {
        # Clinical Trial & Evidence Gaps
        r"(?:lack|absence) of (?:randomized|controlled|prospective|rct) (?:trials|studies)": "RCT Gap",
        r"limited (?:clinical|real-world) (?:evidence|data|validation)": "Real-world Evidence Gap",
        r"(?:small|limited|insufficient) (?:sample size|cohort|patient population|n\s*=\s*\d+)": "Sample Size Limitation",
        r"(?:short|limited) (?:follow-up|observation) (?:period|duration)": "Follow-up Gap",
        r"heterogeneity (?:in|of) (?:patient|treatment|study) (?:populations|protocols|designs)": "Heterogeneity Gap",
        r"lack of (?:longitudinal|long-term) (?:studies|data|outcomes)": "Longitudinal Data Gap",
        
        # Treatment & Intervention Gaps
        r"optimal (?:dose|dosage|treatment|regimen|protocol) (?:remains|is) (?:unclear|undetermined|unknown|not established)": "Optimal Treatment Unknown",
        r"(?:no|limited) (?:standardized|consensus|established) (?:guidelines|protocols|criteria|recommendations)": "Guideline Gap",
        r"efficacy (?:in|across) (?:different|diverse) (?:populations|settings|subgroups) (?:is unclear|remains unknown|not established)": "Population-specific Efficacy Gap",
        r"long-term (?:safety|efficacy|outcomes|side effects) (?:not|remain) (?:established|evaluated|assessed|known)": "Long-term Safety Gap",
        r"(?:adverse|side) effects? (?:profile|data) (?:limited|unknown|not well|insufficiently)": "Safety Profile Gap",
        
        # Diagnostic & Biomarker Gaps
        r"lack of (?:validated|reliable|specific|sensitive) (?:biomarkers|diagnostic criteria|diagnostic tools)": "Biomarker Gap",
        r"(?:sensitivity|specificity|ppv|npv) (?:needs|requires) (?:improvement|validation|optimization)": "Diagnostic Accuracy Gap",
        r"early (?:detection|diagnosis|screening) (?:remains|is) (?:challenging|difficult|suboptimal)": "Early Detection Gap",
        r"(?:differential|accurate) diagnosis (?:challenging|difficult|remains problematic)": "Diagnosis Challenge",
        
        # Mechanism & Pathway Gaps
        r"(?:mechanisms?|pathways?) (?:underlying|of action|driving) .{1,60} (?:remain|are) (?:unclear|unknown|poorly understood)": "Mechanism Gap",
        r"(?:etiology|pathophysiology) (?:of|underlying) .{1,40} (?:remains|is) (?:unclear|unknown)": "Etiology Gap"
    }

def get_general_gap_patterns():
    """Enhanced general academic research gap patterns."""
    return {
        # Future Research Needs
        r"(?:further|future|additional) (?:research|studies|investigations|work) (?:is|are|will be|should be|needed|required|warranted|necessary)": "Future Research Needed",
        r"(?:more|additional) (?:research|studies|data|evidence) (?:is|are) (?:needed|required|warranted)": "More Research Needed",
        r"future (?:work|directions|studies|research) (?:should|will|could|needs to) (?:focus on|address|examine|explore|investigate|consider)": "Future Direction",
        r"(?:warrants|requires|merits|calls for) (?:further|additional|more|systematic) (?:investigation|exploration|research|study|analysis)": "Calls for Investigation",
        r"urgent(?:ly)? (?:need|require|call) for .{1,50} (?:research|study|investigation)": "Urgent Research Need",
        
        # Knowledge Gaps
        r"(?:remains|is) (?:poorly|not fully|incompletely|inadequately|insufficiently) (?:understood|characterized|defined|elucidated|explained)": "Knowledge Gap",
        r"(?:little|limited|scant) (?:is|has been) (?:known|understood|reported|documented|studied) (?:about|regarding|concerning|on)": "Limited Knowledge",
        r"(?:limited|scarce|insufficient|inadequate) (?:evidence|data|information|knowledge) (?:exists|is available|to support|regarding)": "Evidence Scarcity",
        r"(?:unclear|unknown|uncertain|obscure|ambiguous) (?:whether|if|how|why|what|when|where)": "Uncertainty",
        r"(?:not|never|yet) (?:been|been fully|been adequately) (?:determined|established|demonstrated|proven|validated)": "Validation Gap",
        
        # Unexplored Areas
        r"has (?:not|never|rarely|seldom) been (?:investigated|explored|studied|examined|addressed|evaluated|considered)": "Unexplored Area",
        r"(?:no|few|limited|scant|insufficient) (?:studies|investigations|research|publications|literature) (?:have|has) (?:examined|investigated|explored|addressed|focused on)": "Literature Gap",
        r"(?:remains|is) (?:largely |mostly |relatively )?(?:unexplored|unexamined|uninvestigated|underexplored|understudied)": "Underexplored Area",
        r"potential (?:area|avenue|direction|opportunity) for (?:future|additional|further) (?:investigation|research|exploration|inquiry)": "Research Opportunity",
        r"(?:gap|void|lacuna) in (?:the|our|existing) (?:literature|knowledge|understanding|research)": "Explicit Gap Statement",
        
        # Evidence & Data Gaps
        r"(?:lack|dearth|absence|paucity|shortage) of (?:evidence|studies|data|research|literature|information)": "Evidence Gap",
        r"(?:limited|sparse|insufficient|scant|patchy) (?:evidence|data) (?:exists|is available|to support|for|regarding)": "Data Limitation",
        r"(?:no|limited|insufficient) (?:empirical|experimental|quantitative|qualitative|systematic) (?:evidence|data|studies|research)": "Methodological Evidence Gap",
        r"data (?:scarcity|limitations|gaps|deficits|are lacking)": "Data Scarcity",
        r"(?:high-quality|robust|reliable) (?:data|evidence) (?:is|are) (?:lacking|needed|required)": "Quality Evidence Gap",
        
        # Conflicting & Inconsistent Findings
        r"(?:conflicting|inconsistent|contradictory|mixed|discrepant|divergent) (?:results|evidence|findings|reports|conclusions|data)": "Conflicting Evidence",
        r"(?:controversial|debated|disputed|contentious) (?:findings|results|evidence|issues|questions)": "Controversy",
        r"(?:lack|absence) of (?:consensus|agreement|convergence|clarity) (?:on|regarding|about|concerning)": "Consensus Gap",
        r"(?:debate|controversy) (?:remains|continues|exists|persists) (?:regarding|about|concerning|over)": "Ongoing Debate",
        
        # Methodological Gaps
        r"(?:methodological|analytical|experimental|design) (?:limitations|challenges|issues|constraints|shortcomings)": "Methodological Limitation",
        r"(?:lack|absence) of (?:standardized|validated|reliable|robust|appropriate) (?:methods|measures|tools|instruments|protocols)": "Methodology Gap",
        r"generalizability (?:is|remains) (?:limited|uncertain|unclear|questionable|unknown)": "Generalizability Gap",
        r"(?:difficult|challenging|problematic) to (?:compare|replicate|reproduce|generalize)": "Reproducibility Challenge",
        r"(?:need|requirement) for (?:novel|improved|better|alternative) (?:methods|approaches|techniques|methodologies)": "Method Innovation Need",
        
        # Study Limitations & Scope Gaps
        r"(?:limitation|constraint|weakness|shortcoming)s? of (?:this|the|our|current|present) (?:study|research|investigation|work|analysis)": "Study Limitation",
        r"(?:this|our|the present) (?:study|research) (?:has|had|suffers from) (?:several|some|certain|important|significant) (?:limitations|constraints|shortcomings)": "Explicit Limitation",
        r"(?:caution|care|carefulness) (?:should|must|needs to) be (?:exercised|taken|applied) when (?:interpreting|generalizing|extrapolating|applying)": "Interpretation Caution",
        r"(?:results|findings) (?:may not|might not|do not) (?:generalize|apply|extend) to": "Generalization Boundary",
        
        # Open Questions
        r"(?:unresolved|open|outstanding|pressing|important) (?:issues|questions|problems|challenges|questions remain)": "Open Question",
        r"(?:key|important|critical|fundamental|essential) (?:questions|issues|aspects) (?:remain|are) (?:unanswered|unresolved|open|pending)": "Critical Open Question",
        r"it (?:remains|is) (?:unclear|unknown|uncertain|to be determined|an open question)": "Status Unknown",
        r"(?:whether|how|why|what|to what extent) .{1,60} (?:remains|is) (?:unclear|unknown|uncertain|to be seen)": "Specific Unknown",
        
        # Theoretical Gaps
        r"(?:theoretical|conceptual|framework) (?:gap|limitation|weakness|issue)": "Theoretical Gap",
        r"lack of (?:theoretical|conceptual) (?:framework|understanding|foundation|clarity)": "Theory Gap",
        r"(?:inadequate|insufficient) (?:theoretical|conceptual) (?:development|foundation|grounding)": "Theory Development Gap"
    }

def get_emerging_gap_patterns():
    """Patterns for emerging/trending gap types."""
    return {
        # Sustainability & Environmental Gaps
        r"(?:environmental|sustainability|climate|carbon) (?:impact|footprint|implications) (?:not|never|rarely) (?:considered|assessed|evaluated|studied)": "Sustainability Gap",
        r"(?:green|sustainable|eco-friendly) (?:alternatives|solutions|approaches) (?:needed|required|lacking)": "Environmental Solution Gap",
        
        # Equity & Inclusion Gaps
        r"(?:health|research|knowledge) (?:equity|disparity|inequity)": "Equity Gap",
        r"(?:underrepresented|marginalized|minority) (?:populations|groups|communities)": "Representation Gap",
        
        # Interdisciplinary Gaps
        r"(?:interdisciplinary|cross-disciplinary|multidisciplinary) (?:approach|perspective|collaboration) (?:needed|lacking|absent)": "Interdisciplinary Gap",
        r"integration of .{1,40} with .{1,40} (?:remains|is) (?:limited|underexplored)": "Integration Gap"
    }

# ==================================================
# 2. ADVANCED SENTENCE ANALYSIS WITH CONTEXT
# ==================================================

class GapSentenceAnalyzer:
    """Analyzes sentences for gap indicators with contextual awareness."""
    
    def __init__(self):
        self.negation_terms = {
            'not', 'no', 'never', 'neither', 'nor', 'none', 'nothing', 'nobody',
            'nowhere', 'hardly', 'scarcely', 'barely', 'don\'t', 'doesn\'t',
            'didn\'t', 'won\'t', 'wouldn\'t', 'shouldn\'t', 'couldn\'t', 'can\'t',
            'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'hasn\'t', 'haven\'t',
            'hadn\'t', 'lack', 'lacking', 'absence', 'absent', 'without'
        }
        
        self.uncertainty_terms = {
            'unclear', 'unknown', 'uncertain', 'ambiguous', 'obscure', 'vague',
            'puzzling', 'perplexing', 'enigmatic', 'mysterious', 'debatable',
            'questionable', 'doubtful', 'dubious', 'inconclusive', 'unresolved',
            'undetermined', 'unverified', 'unproven', 'speculative', 'tentative'
        }
        
        self.future_oriented = {
            'future', 'further', 'next', 'subsequent', 'forthcoming', 'upcoming',
            'prospective', 'planned', 'intended', 'proposed', 'recommended',
            'suggested', 'should', 'needs to', 'must', 'require', 'warrant'
        }
        
    def analyze_sentence(self, sentence: str) -> Dict:
        """Perform deep analysis of a sentence for gap characteristics."""
        sentence_lower = sentence.lower()
        words = set(re.findall(r'\b\w+\b', sentence_lower))
        
        analysis = {
            'has_negation': bool(words & self.negation_terms),
            'has_uncertainty': bool(words & self.uncertainty_terms),
            'is_future_oriented': bool(words & self.future_oriented),
            'sentence_length': len(sentence.split()),
            'word_count': len(words),
            'complexity_score': self._calculate_complexity(sentence),
            'certainty_score': self._calculate_certainty(sentence_lower),
            'is_recommendation': self._is_recommendation(sentence_lower),
            'is_limitation': self._is_limitation(sentence_lower),
            'confidence': 0.0
        }
        
        # Calculate overall confidence score
        analysis['confidence'] = self._calculate_confidence(analysis)
        return analysis
    
    def _calculate_complexity(self, sentence: str) -> float:
        """Calculate linguistic complexity score."""
        # Average word length
        words = sentence.split()
        if not words:
            return 0.0
        avg_word_length = sum(len(w) for w in words) / len(words)
        
        # Clause complexity (approximated by punctuation)
        clauses = len(re.findall(r'[,;:.]', sentence)) + 1
        
        # Normalize to 0-1 scale
        complexity = min(1.0, (avg_word_length / 10) * (clauses / 5))
        return round(complexity, 2)
    
    def _calculate_certainty(self, sentence_lower: str) -> float:
        """Calculate certainty level (0 = uncertain, 1 = certain)."""
        certainty_markers = {
            'definitely', 'certainly', 'clearly', 'obviously', 'demonstrably',
            'proven', 'established', 'confirmed', 'verified', 'conclusive'
        }
        uncertainty_count = sum(1 for term in self.uncertainty_terms if term in sentence_lower)
        certainty_count = sum(1 for term in certainty_markers if term in sentence_lower)
        
        # Score from 0 to 1
        score = 0.5 + (certainty_count - uncertainty_count) * 0.1
        return max(0.0, min(1.0, score))
    
    def _is_recommendation(self, sentence_lower: str) -> bool:
        """Detect if sentence is a recommendation."""
        recommendation_starters = [
            'we recommend', 'it is recommended', 'future work should',
            'researchers should', 'studies should', 'further research',
            'future studies', 'investigation should', 'attention should'
        ]
        return any(starter in sentence_lower for starter in recommendation_starters)
    
    def _is_limitation(self, sentence_lower: str) -> bool:
        """Detect if sentence describes a limitation."""
        limitation_indicators = [
            'limitation', 'limited by', 'constrained by', 'restricted by',
            'hindered by', 'hampered by', 'impeded by', 'bottleneck'
        ]
        return any(indicator in sentence_lower for indicator in limitation_indicators)
    
    def _calculate_confidence(self, analysis: Dict) -> float:
        """Calculate confidence score for gap detection."""
        score = 0.5
        
        # Boost for negation + uncertainty combination
        if analysis['has_negation'] and analysis['has_uncertainty']:
            score += 0.2
        
        # Boost for future orientation
        if analysis['is_future_oriented']:
            score += 0.15
        
        # Penalize very short sentences
        if analysis['sentence_length'] < 8:
            score -= 0.1
        
        # Boost for recommendation or limitation
        if analysis['is_recommendation']:
            score += 0.1
        if analysis['is_limitation']:
            score += 0.1
            
        return round(max(0.0, min(1.0, score)), 2)


# ==================================================
# 3. SEMANTIC SIMILARITY & CLUSTERING
# ==================================================

def calculate_semantic_similarity(gap1: str, gap2: str) -> float:
    """
    Calculate semantic similarity between two gap statements.
    Uses word overlap and n-gram similarity as proxy for semantic similarity.
    """
    def preprocess(text):
        # Remove stopwords and normalize
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                    'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                    'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                    'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                    'through', 'during', 'before', 'after', 'above', 'below',
                    'between', 'under', 'again', 'further', 'then', 'once', 'and',
                    'but', 'if', 'or', 'because', 'until', 'while', 'so', 'than',
                    'that', 'this', 'these', 'those', 'i', 'me', 'my', 'myself',
                    'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs'}
        
        words = re.findall(r'\b\w+\b', text.lower())
        return set(w for w in words if w not in stopwords and len(w) > 2)
    
    set1, set2 = preprocess(gap1), preprocess(gap2)
    
    if not set1 or not set2:
        return 0.0
    
    # Jaccard similarity
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    jaccard = intersection / union if union > 0 else 0
    
    # N-gram similarity (bigrams)
    def get_bigrams(text):
        words = re.findall(r'\b\w+\b', text.lower())
        return set(f"{words[i]} {words[i+1]}" for i in range(len(words)-1))
    
    bigrams1, bigrams2 = get_bigrams(gap1), get_bigrams(gap2)
    if bigrams1 and bigrams2:
        bigram_sim = len(bigrams1 & bigrams2) / max(len(bigrams1), len(bigrams2))
    else:
        bigram_sim = 0
    
    # Weighted combination
    return round(0.6 * jaccard + 0.4 * bigram_sim, 3)


def cluster_similar_gaps(gaps: List[Dict], similarity_threshold: float = 0.6) -> List[Dict]:
    """
    Cluster similar gap statements to avoid redundancy.
    Returns representative gaps from each cluster.
    """
    if not gaps:
        return []
    
    n = len(gaps)
    visited = [False] * n
    clusters = []
    
    for i in range(n):
        if visited[i]:
            continue
            
        # Start new cluster
        cluster = [gaps[i]]
        visited[i] = True
        
        # Find similar gaps
        for j in range(i + 1, n):
            if visited[j]:
                continue
                
            sim = calculate_semantic_similarity(
                gaps[i]['gap_statement'], 
                gaps[j]['gap_statement']
            )
            
            if sim >= similarity_threshold:
                cluster.append(gaps[j])
                visited[j] = True
        
        clusters.append(cluster)
    
    # Select representative from each cluster (highest citation + confidence)
    representatives = []
    for cluster in clusters:
        best = max(cluster, key=lambda x: (
            int(x.get('citations', 0)) + 
            x.get('analysis', {}).get('confidence', 0) * 100
        ))
        best['cluster_size'] = len(cluster)
        best['cluster_members'] = [c['gap_statement'][:100] + '...' for c in cluster]
        representatives.append(best)
    
    return representatives


# ==================================================
# 4. ENHANCED MAIN ANALYSIS FUNCTION
# ==================================================

def analyze_research_gaps(results, query="", min_confidence=0.3):
    """
    Enhanced analysis of research gaps with semantic clustering and confidence scoring.
    
    Args:
        results: List of paper dictionaries with abstracts/tldrs
        query: Search query for domain detection
        min_confidence: Minimum confidence threshold for gap inclusion
    
    Returns:
        Dict with structured gap data, clusters, and analytics
    """
    # Initialize analyzer
    analyzer = GapSentenceAnalyzer()
    
    # Combine all patterns with their categories
    all_patterns = {}
    all_patterns.update(get_general_gap_patterns())
    
    # Domain-specific pattern selection
    query_lower = query.lower()
    pattern_types = ["General"]
    domain_scores = {}
    
    # Detect domains with confidence scores
    tech_terms = ["ai", "artificial intelligence", "machine learning", "deep learning", 
                  "neural network", "algorithm", "computational", "model", "prediction",
                  "classification", "regression", "nlp", "computer vision"]
    clinical_terms = ["patient", "clinical", "treatment", "therapy", "disease", 
                     "diagnosis", "medical", "hospital", "health", "medicine",
                     "drug", "surgical", "symptom", "prognosis", "biomarker"]
    
    tech_score = sum(1 for term in tech_terms if term in query_lower)
    clinical_score = sum(1 for term in clinical_terms if term in query_lower)
    
    if tech_score > 0:
        all_patterns.update(get_ai_gap_patterns())
        pattern_types.append("Technical/AI")
        domain_scores['Technical/AI'] = tech_score
        
    if clinical_score > 0:
        all_patterns.update(get_clinical_gap_patterns())
        pattern_types.append("Clinical/Medical")
        domain_scores['Clinical/Medical'] = clinical_score
    
    # Always add emerging patterns
    all_patterns.update(get_emerging_gap_patterns())
    pattern_types.append("Emerging")
    
    # Compile regex patterns for efficiency
    compiled_patterns = {
        re.compile(pattern, re.IGNORECASE): category 
        for pattern, category in all_patterns.items()
    }
    
    found_gaps = []
    all_keywords = []
    gap_categories = Counter()
    temporal_distribution = Counter()
    author_gaps = defaultdict(list)
    
    # Analyze papers
    papers_with_text = [p for p in results if p.get('abstract') or p.get('tldr')]
    total_sentences = 0
    
    for paper in papers_with_text:
        paper_id = paper.get('doi', paper.get('title', 'unknown'))
        text = f"{paper.get('tldr', '')} {paper.get('abstract', '')}"
        sentences = re.split(r'(?<=[.!?])\s+', text)
        total_sentences += len(sentences)
        
        # Extract year for temporal analysis
        year = str(paper.get('year', 'unknown'))
        if year.isdigit():
            decade = f"{year[:3]}0s"
            temporal_distribution[decade] += 1
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) < 5 or len(sentence) > 500:
                continue
            
            # Check against all patterns
            for pattern, category in compiled_patterns.items():
                if pattern.search(sentence):
                    # Perform deep analysis
                    analysis = analyzer.analyze_sentence(sentence)
                    
                    # Skip low confidence gaps
                    if analysis['confidence'] < min_confidence:
                        continue
                    
                    # Extract keywords from context
                    context_keywords = extract_context_keywords(sentence, query)
                    
                    gap_entry = {
                        'title': paper.get('title', 'Unknown Title'),
                        'gap_statement': sentence,
                        'year': paper.get('year', 'N/A'),
                        'category': category,
                        'subcategory': categorize_gap_detailed(sentence, category),
                        'citations': paper.get('citations', 0),
                        'doi': paper.get('doi', 'N/A'),
                        'venue': paper.get('venue', 'N/A'),
                        'analysis': analysis,
                        'keywords': context_keywords,
                        'pattern_matched': pattern.pattern[:50] + '...' if len(pattern.pattern) > 50 else pattern.pattern
                    }
                    
                    found_gaps.append(gap_entry)
                    gap_categories[category] += 1
                    author_gaps[paper.get('ieee_authors', 'Unknown')].append(gap_entry)
                    break  # One gap per sentence
        
        # Collect keywords
        if paper.get('keywords'):
            all_keywords.extend([k.strip().lower() for k in paper['keywords'].split(',') if k.strip()])
    
    # Cluster similar gaps to remove redundancy
    clustered_gaps = cluster_similar_gaps(found_gaps, similarity_threshold=0.65)
    
    # Sort by composite score (citations + confidence + recency)
    def composite_score(gap):
        citations = int(gap.get('citations', 0))
        confidence = gap.get('analysis', {}).get('confidence', 0)
        year = gap.get('year', '0')
        recency = 0
        if year.isdigit() and int(year) >= 2020:
            recency = 50  # Boost recent gaps
        
        cluster_bonus = gap.get('cluster_size', 1) * 10
        
        return citations + (confidence * 100) + recency + cluster_bonus
    
    sorted_gaps = sorted(clustered_gaps, key=composite_score, reverse=True)
    
    # Keyword analysis with TF-like weighting
    keyword_counts = Counter(all_keywords)
    total_kw = sum(keyword_counts.values())
    top_keywords = [
        (kw, count, round(count/max(total_kw, 1), 3)) 
        for kw, count in keyword_counts.most_common(20)
    ]
    
    # Calculate gap novelty (how often mentioned)
    gap_frequency = Counter(g['gap_statement'][:100] for g in found_gaps)
    recurring_gaps = [(stmt, freq) for stmt, freq in gap_frequency.items() if freq > 1]
    
    return {
        'total_gaps_found': len(found_gaps),
        'unique_gaps_after_clustering': len(clustered_gaps),
        'gap_list': sorted_gaps,
        'top_keywords': top_keywords,
        'gap_categories': dict(gap_categories),
        'subcategories': extract_subcategories(sorted_gaps),
        'pattern_types_used': pattern_types,
        'domain_scores': domain_scores,
        'papers_analyzed': len(papers_with_text),
        'sentences_analyzed': total_sentences,
        'temporal_distribution': dict(temporal_distribution),
        'high_impact_gaps': len([g for g in sorted_gaps if int(g.get('citations', 0)) >= 100]),
        'high_confidence_gaps': len([g for g in sorted_gaps if g.get('analysis', {}).get('confidence', 0) >= 0.7]),
        'recurring_gaps': recurring_gaps[:10],  # Top recurring themes
        'authors_with_most_gaps': sorted(
            [(author, len(gaps)) for author, gaps in author_gaps.items()],
            key=lambda x: x[1], reverse=True
        )[:5]
    }


def extract_context_keywords(sentence: str, query: str) -> List[str]:
    """Extract domain-specific keywords from gap sentence."""
    # Technical terms
    tech_patterns = [
        r'\b(?:neural|deep|machine|artificial|reinforcement|supervised|unsupervised)\s+\w+',
        r'\b(?:algorithm|model|architecture|framework|approach|method)\w*\b',
        r'\b(?:dataset|benchmark|corpus)\w*\b',
        r'\b(?:classification|regression|clustering|generation|prediction)\w*\b'
    ]
    
    # Medical terms
    medical_patterns = [
        r'\b(?:patient|clinical|therapeutic|diagnostic|prognostic)\w*\b',
        r'\b(?:disease|syndrome|disorder|condition|pathology)\w*\b',
        r'\b(?:treatment|therapy|intervention|regimen|dosage)\w*\b',
        r'\b(?:biomarker|genetic|molecular|cellular)\w*\b'
    ]
    
    keywords = []
    all_patterns = tech_patterns + medical_patterns
    
    for pattern in all_patterns:
        matches = re.findall(pattern, sentence, re.IGNORECASE)
        keywords.extend(matches)
    
    # Add query-related terms
    query_terms = [t for t in query.lower().split() if len(t) > 3]
    for term in query_terms:
        if term in sentence.lower():
            keywords.append(term)
    
    return list(set(keywords))[:5]  # Return unique keywords, max 5


def categorize_gap_detailed(sentence: str, parent_category: str) -> str:
    """Provide more granular categorization."""
    sentence_lower = sentence.lower()
    
    # Temporal indicators
    if any(w in sentence_lower for w in ['long-term', 'longitudinal', 'future', 'prospective']):
        return f"{parent_category} - Temporal"
    
    # Population indicators
    if any(w in sentence_lower for w in ['population', 'patient', 'participant', 'subject', 'demographic']):
        return f"{parent_category} - Population-specific"
    
    # Methodological indicators
    if any(w in sentence_lower for w in ['method', 'approach', 'technique', 'methodology', 'design']):
        return f"{parent_category} - Methodological"
    
    # Scale indicators
    if any(w in sentence_lower for w in ['large-scale', 'small-scale', 'sample size', 'underpowered']):
        return f"{parent_category} - Scale-related"
    
    return parent_category


def extract_subcategories(gaps: List[Dict]) -> Dict[str, int]:
    """Extract and count subcategories."""
    subcat_counts = Counter()
    for gap in gaps:
        subcat = gap.get('subcategory', 'Unknown')
        subcat_counts[subcat] += 1
    return dict(subcat_counts)


# ==================================================
# 5. ENHANCED STATISTICS & REPORTING
# ==================================================

def generate_gap_summary_stats(gap_data: Dict) -> Dict:
    """Generate comprehensive statistical summary."""
    gaps = gap_data.get('gap_list', [])
    
    # Citation analysis
    citations = [int(g.get('citations', 0)) for g in gaps]
    avg_citations = sum(citations) / len(citations) if citations else 0
    
    # Confidence analysis
    confidences = [g.get('analysis', {}).get('confidence', 0) for g in gaps]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    # Temporal analysis
    years = [str(g.get('year', '')) for g in gaps if str(g.get('year', '')).isdigit()]
    year_dist = Counter(years)
    
    # Category diversity
    categories = gap_data.get('gap_categories', {})
    diversity_score = len(categories) / max(len(categories), 1)
    
    # Gap density (gaps per paper)
    density = gap_data['total_gaps_found'] / max(gap_data['papers_analyzed'], 1)
    
    return {
        'total_gaps': gap_data['total_gaps_found'],
        'unique_gaps': gap_data.get('unique_gaps_after_clustering', 0),
        'reduction_ratio': round(1 - gap_data.get('unique_gaps_after_clustering', 0) / max(gap_data['total_gaps_found'], 1), 2),
        'papers_with_gaps': len(set(g['title'] for g in gaps)),
        'avg_gaps_per_paper': round(density, 2),
        'citation_stats': {
            'average': round(avg_citations, 1),
            'median': sorted(citations)[len(citations)//2] if citations else 0,
            'max': max(citations) if citations else 0,
            'high_impact_count': len([c for c in citations if c >= 100])
        },
        'confidence_stats': {
            'average': round(avg_confidence, 2),
            'high_confidence_count': len([c for c in confidences if c >= 0.7]),
            'low_confidence_count': len([c for c in confidences if c < 0.4])
        },
        'category_diversity': round(diversity_score, 2),
        'top_categories': gap_data.get('gap_categories', {}),
        'year_distribution': dict(year_dist),
        'domain_relevance': gap_data.get('domain_scores', {})
    }


def identify_priority_gaps(gap_data: Dict, top_n: int = 5) -> List[Dict]:
    """Identify high-priority gaps based on multiple criteria."""
    gaps = gap_data.get('gap_list', [])
    
    priority_scores = []
    for gap in gaps:
        score = 0
        reasons = []
        
        # High citations = high impact potential
        cites = int(gap.get('citations', 0))
        if cites >= 100:
            score += 3
            reasons.append("High-impact paper")
        elif cites >= 50:
            score += 2
            reasons.append("Well-cited paper")
        
        # High confidence = reliable detection
        conf = gap.get('analysis', {}).get('confidence', 0)
        if conf >= 0.8:
            score += 2
            reasons.append("High confidence detection")
        
        # Recent = timely
        year = str(gap.get('year', ''))
        if year.isdigit() and int(year) >= 2022:
            score += 2
            reasons.append("Recent finding")
        
        # Recurring theme = important pattern
        if gap.get('cluster_size', 1) > 1:
            score += gap.get('cluster_size', 1)
            reasons.append(f"Recurring theme (cluster of {gap.get('cluster_size')})")
        
        # Explicit limitation statement
        if gap.get('analysis', {}).get('is_limitation'):
            score += 1
            reasons.append("Explicit limitation")
        
        priority_scores.append((gap, score, reasons))
    
    # Sort by score and return top N
    priority_scores.sort(key=lambda x: x[1], reverse=True)
    return [
        {
            **gap, 
            'priority_score': score,
            'priority_reasons': reasons
        } 
        for gap, score, reasons in priority_scores[:top_n]
    ]
