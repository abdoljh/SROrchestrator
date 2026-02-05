"""
SROrch Streamlit Interface - Enhanced with Report Writing
A comprehensive web interface for Scholarly Research Orchestrator with integrated academic report generation
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
# REPORT GENERATION FUNCTIONS (from streamlit_app_rep.py)
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

def analyze_topic_with_ai(topic: str, subject: str) -> Dict:
    """Analyze topic and generate research plan"""
    update_report_progress('Topic Analysis', 'Creating research plan...', 10)
    
    variations = generate_phrase_variations(topic)
    st.session_state.report_research['phrase_variations'] = variations
    
    prompt = f"""Research plan for "{topic}" in {subject}.

Create:
1. 5 specific subtopics about "{topic}"
2. 5 academic search queries for finding papers (2020-2025)

Target databases: arXiv, IEEE, ACM, PubMed, Semantic Scholar

Return ONLY JSON:
{{
  "subtopics": ["aspect 1", "aspect 2", ...],
  "researchQueries": ["query 1", "query 2", ...]
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
    except:
        pass
    
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
        ]
    }

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
    """Format citation in IEEE style"""
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
    
    formatted_authors = format_authors_ieee(authors)
    citation = f'[{index}] {formatted_authors}, "{title}," {venue}, {year}. <a href="{url}" target="_blank">{url}</a>'
    
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

def extract_cited_references(draft: Dict) -> set:
    """Extract all citation numbers used in the draft"""
    cited = set()
    
    text_parts = []
    for key, value in draft.items():
        if isinstance(value, str):
            text_parts.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    for k, v in item.items():
                        if isinstance(v, str):
                            text_parts.append(v)
                elif isinstance(item, str):
                    text_parts.append(item)
    
    full_text = ' '.join(text_parts)
    matches = re.findall(r'\[(\d+)\]', full_text)
    
    for match in matches:
        cited.add(int(match))
    
    return cited

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

def generate_draft_optimized(
    topic: str,
    subject: str,
    subtopics: List[str],
    sources: List[Dict],
    variations: List[str],
    max_sources: int = 25
) -> Dict:
    """Generate report draft using academic sources"""
    update_report_progress('Drafting', 'Writing report...', 70)
    
    if not sources:
        raise Exception("No sources available")
    
    # Prepare source list for prompt (limited to max_sources)
    source_list = []
    for i, s in enumerate(sources[:max_sources], 1):
        meta = s.get('metadata', {})
        source_list.append(f"""[{i}] {meta.get('title', 'Unknown')} ({meta.get('year', 'N/A')})
Authors: {meta.get('authors', 'Unknown')}
Venue: {meta.get('venue', 'Unknown')}
{s['url'][:70]}
Abstract: {s.get('content', '')[:200]}""")
    
    sources_text = "\n\n".join(source_list)
    
    variations_text = f"""CRITICAL INSTRUCTION - PHRASE VARIATION:
You MUST use these variations to avoid repetition:
- "{topic}" - USE THIS SPARINGLY (maximum 5 times)
- "{variations[1]}" - PREFER THIS
- "{variations[2]}" - USE THIS OFTEN
- "this domain" - USE THIS
- "this research area" - USE THIS

DO NOT repeat "{topic}" more than 5 times total."""
    
    prompt = f"""Write academic report about "{topic}" in {subject}.

{variations_text}

REQUIREMENTS:
- Use ONLY provided academic sources below
- Cite sources as [1], [2], [3] etc. - just the number in brackets
- Include specific data, statistics, and years from sources
- VARY your phrasing - avoid repetition

SUBTOPICS: {', '.join(subtopics)}

ACADEMIC SOURCES:
{sources_text}

Write these sections:
1. Abstract (150-250 words)
2. Introduction
3. Literature Review
4. 3-4 Main Sections covering subtopics
5. Data & Analysis
6. Challenges
7. Future Outlook
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
                draft[key] = f"Section about the topic."
    
    # Fix citations
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
    
    return draft

def refine_draft_simple(draft: Dict, topic: str, sources_count: int) -> Dict:
    """Add executive summary"""
    update_report_progress('Refinement', 'Final polish...', 92)
    
    draft['executiveSummary'] = (
        f"This comprehensive report examines {topic}, analyzing key developments, "
        f"challenges, and future directions based on {sources_count} authoritative academic sources."
    )
    
    return draft

def generate_html_report_optimized(
    refined_draft: Dict,
    form_data: Dict,
    sources: List[Dict],
    max_sources: int = 25
) -> str:
    """Generate HTML report with cited and further references"""
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
    cited_refs = extract_cited_references(refined_draft)
    cited_refs_sorted = sorted(cited_refs)
    
    old_to_new = {}
    for new_num, old_num in enumerate(cited_refs_sorted, 1):
        old_to_new[old_num] = new_num
    
    renumbered_draft = renumber_citations_in_draft(refined_draft, old_to_new)
    
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
    
    # Generate references
    for old_ref_num in cited_refs_sorted:
        new_ref_num = old_to_new[old_ref_num]
        if old_ref_num <= len(sources):
            source = sources[old_ref_num - 1]
            if style == 'APA':
                citation = format_citation_apa(source, new_ref_num)
            else:
                citation = format_citation_ieee(source, new_ref_num)
            html += f'        <div class="ref-item">{citation}</div>\n'
    
    # Fallback: if no citations, include first 10 sources
    if len(cited_refs_sorted) == 0:
        for i, source in enumerate(sources[:10], 1):
            if style == 'APA':
                citation = format_citation_apa(source, i)
            else:
                citation = format_citation_ieee(source, i)
            html += f'        <div class="ref-item">{citation}</div>\n'
    
    # Add "Further References" section for uncited but relevant sources
    # Only include sources that were part of the writing process but not cited
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
        for idx, source in uncited_sources[:20]:  # Limit to 20 further refs
            # Don't renumber - just list them
            if style == 'APA':
                citation = format_citation_apa(source, idx)
            else:
                citation = format_citation_ieee(source, idx)
            # Remove numbering from citation since these are supplementary
            citation_text = citation.split(']', 1)[1] if ']' in citation else citation
            html += f'        <div class="ref-item" style="font-size: 9pt;">• {citation_text}</div>\n'
    
    html += """
    </div>
</body>
</html>"""
    
    return html

def execute_report_pipeline():
    """Execute complete report generation pipeline"""
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
            # Check if existing results are relevant
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
        
        # Convert to source format
        sources = convert_orchestrator_to_source_format(results)
        st.session_state.report_research['sources'] = sources
        
        if len(sources) < 3:
            raise Exception(f"Only {len(sources)} sources found. Need at least 3.")
        
        # Stage 3: Draft Generation
        st.info("✍️ Stage 3/5: Writing report...")
        max_sources = st.session_state.report_form_data.get('max_sources', 25)
        draft = generate_draft_optimized(
            topic,
            subject,
            analysis['subtopics'],
            sources,
            st.session_state.report_research['phrase_variations'],
            max_sources=max_sources
        )
        st.session_state.report_draft = draft
        
        # Stage 4: Quality Check (simplified)
        st.info("🔍 Stage 4/5: Quality check...")
        update_report_progress('Review', 'Quality check...', 85)
        
        # Stage 5: Refinement & HTML Generation
        st.info("✨ Stage 5/5: Final refinement...")
        refined = refine_draft_simple(draft, topic, len(sources))
        st.session_state.report_final = refined
        
        html = generate_html_report_optimized(
            refined,
            st.session_state.report_form_data,
            sources,
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
            f"{len(sources)} sources, {st.session_state.report_api_calls} API calls"
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
    st.session_state.report_research = {
        'subtopics': [],
        'sources': [],
        'phrase_variations': []
    }

# ================================================================================
# SEARCH TAB FUNCTIONS (from original streamlit_app_orc.py)
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
# MAIN APPLICATION
# ================================================================================

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<p class="main-header">🔬 SROrch - Scholarly Research Orchestrator & Report Writer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Engine Academic Literature Search, Analysis & Report Generation</p>', unsafe_allow_html=True)
    
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
    
    # Main content area - 4 TABS NOW!
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
                    
                    st.info("👉 Switch to the 'Results' tab to view findings, or 'Report Writer' to generate a report!")
                    
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
    
    # ====== TAB 3: REPORT WRITER ======
    with tab3:
        st.header("📝 Academic Report Writer")
        
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
                    st.success(f"✅ Conservative mode ({max_sources} sources) - ~3-5 min, ~20-30 API calls")
                elif max_sources <= 50:
                    st.warning(f"⚠️ Balanced mode ({max_sources} sources) - ~5-8 min, ~40-50 API calls, higher cost")
                else:
                    st.error(f"🔴 Comprehensive mode ({max_sources} sources) - ~8-12 min, ~60-80 API calls, **significantly higher cost**")
                
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
                **How it works:**
                1. 🔍 Searches 18 academic databases (or reuses existing results)
                2. 📚 Extracts real metadata (authors, venues, years)
                3. ✍️ Uses Claude to write comprehensive academic report
                4. 📄 Generates HTML with proper IEEE/APA citations
                
                **Time:** 3-5 minutes | **API Calls:** ~20-30 to Anthropic
                """)
                
                if st.button(
                    "🚀 Generate Report",
                    disabled=not valid,
                    type="primary",
                    use_container_width=True
                ):
                    execute_report_pipeline()
                    st.rerun()
                
                if not valid:
                    st.warning("⚠️ Please fill all required fields")
            
            elif st.session_state.report_step == 'processing':
                st.markdown("### 🔄 Generating Report")
                
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
                
                # Show sources as they're found
                if st.session_state.report_research['sources']:
                    with st.expander(
                        f"📚 Academic Sources Found ({len(st.session_state.report_research['sources'])})",
                        expanded=True
                    ):
                        for i, s in enumerate(st.session_state.report_research['sources'][:10], 1):
                            meta = s.get('metadata', {})
                            st.markdown(
                                f"**{i}.** {meta.get('title', 'Unknown')[:80]}...  "
                                f"👤 {meta.get('authors', 'Unknown')} | "
                                f"📊 {s.get('credibilityScore', 0)}%"
                            )
                
                if st.session_state.report_processing:
                    time.sleep(3)
                    st.rerun()
            
            elif st.session_state.report_step == 'complete':
                st.success("✅ Report Generated Successfully!")
                
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
                
                st.markdown("---")
                
                # Download
                col1, col2 = st.columns(2)
                with col1:
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
                        st.info("""
                        **To create PDF:**
                        1. Open HTML in browser
                        2. Press Ctrl+P (Cmd+P on Mac)
                        3. Select "Save as PDF"
                        """)
                
                with col2:
                    st.metric("File Size", f"{len(st.session_state.report_html) / 1024:.1f} KB")
                    st.metric("Quality", "Professional")
                
                st.markdown("---")
                
                # Sources preview
                with st.expander("📚 References Preview", expanded=False):
                    for i, s in enumerate(st.session_state.report_research['sources'][:20], 1):
                        meta = s.get('metadata', {})
                        orch = s.get('_orchestrator_data', {})
                        
                        st.markdown(f"**[{i}]** {meta.get('title', 'N/A')}")
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
        - Intelligent relevance scoring
        - Deep abstract fetching
        - Enhanced gap analysis with domain-specific patterns
        - Publication analytics and visualizations
        - Multiple export formats (CSV, JSON, BibTeX)
        
        **Report Writer (NEW!):**
        - Automated academic report generation
        - Proper IEEE/APA citations with real metadata
        - Integration with 18 search engines
        - Professional HTML output (convert to PDF)
        - Reuses existing search results when relevant
        
        #### 🚀 Getting Started
        
        **Search Mode:**
        1. Enter your research query
        2. Configure search parameters (optional)
        3. Click "Start Search"
        4. View results, download data, or generate a report
        
        **Report Mode:**
        1. Configure report details (topic, author, institution)
        2. Choose citation style (IEEE or APA)
        3. Click "Generate Report"
        4. Download HTML report (convertible to PDF)
        
        #### 🔑 API Keys
        
        **Required for Report Generation:**
        - Anthropic API key (in Streamlit secrets)
        
        **Optional for Enhanced Search:**
        - Semantic Scholar (free, highly recommended)
        - SERP API (Google Scholar)
        - CORE, SCOPUS, Springer Nature
        
        All keys are session-only for security!
        
        #### 💡 Tips
        
        - **Reuse Results**: Report Writer can reuse existing search results
        - **Citation Quality**: Papers in 4+ databases are highly reliable
        - **Export Options**: Use CSV for analysis, BibTeX for citations
        - **PDF Creation**: Open HTML in browser and print to PDF
        
        ---
        
        **Version:** 2.0 - Integrated Search & Report Generation  
        **Security Model:** Zero-Trust (User-Provided Keys)  
        **License:** MIT
        """)
        
        with st.expander("🖥️ System Information"):
            st.code(f"""
Python Version: {sys.version}
Working Directory: {os.getcwd()}
Streamlit Version: {st.__version__}
Security Model: Session-only keys (no persistence)
Report Generation: Claude Sonnet 4.5
            """)

if __name__ == "__main__":
    main()
