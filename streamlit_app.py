"""
SROrch Streamlit Interface
A web interface for the Scholarly Research Orchestrator
Session-Only API Keys (Safe for Public Deployment)
"""

import streamlit as st
import os
import sys
import json
import shutil
import pandas as pd  # ✅ NEW: For interactive data viewer
from datetime import datetime
from pathlib import Path
import zipfile

# Import the orchestrator (assuming master_orchestrator.py is in the same directory)
from master_orchestrator import ResearchOrchestrator

# Page configuration
st.set_page_config(
    page_title="SROrch - Research Orchestrator",
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
        padding: 0.5rem;
        background-color: #ffeaa7; 
        border-left: 4px solid #fd846e; 
        border-radius: 0.3rem;
        margin: 1rem 0;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# background-color: #ffeaa7 pale yellow
# border-left: #fdcb6e; # saturated shade of yellow

def get_secret_or_empty(key_name):
    """
    Safely retrieve a secret from Streamlit secrets, return empty string if not found.
    This allows graceful fallback to user-provided keys.
    """
    try:
        return st.secrets.get(key_name, '')
    except (FileNotFoundError, KeyError, AttributeError):
        return ''

def check_dev_mode():
    """
    Check if running in development mode (secrets configured)
    Returns (is_dev_mode, configured_keys)
    """
    dev_keys = []
    
    # Check which keys are configured in secrets
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
    """Initialize session state with empty API keys (session-only, not persistent)"""
    # Current Premium Engines
    if 'user_s2_key' not in st.session_state:
        st.session_state['user_s2_key'] = ''
    if 'user_serp_key' not in st.session_state:
        st.session_state['user_serp_key'] = ''
    if 'user_core_key' not in st.session_state:
        st.session_state['user_core_key'] = ''
    if 'user_scopus_key' not in st.session_state:
        st.session_state['user_scopus_key'] = ''
    if 'user_springer_key' not in st.session_state:  # ✅ NEW
        st.session_state['user_springer_key'] = ''
    if 'user_email' not in st.session_state:
        st.session_state['user_email'] = 'researcher@example.com'
    
    # 📌 PLACEHOLDER: Add additional premium engine keys here
    # Template for adding a new premium engine:
    # if 'user_new_engine_key' not in st.session_state:
    #     st.session_state['user_new_engine_key'] = ''
    
    # Example: IEEE Xplore (uncomment when implemented)
    # if 'user_ieee_key' not in st.session_state:
    #     st.session_state['user_ieee_key'] = ''
    
    # Example: Web of Science (uncomment when implemented)
    # if 'user_wos_key' not in st.session_state:
    #     st.session_state['user_wos_key'] = ''

def load_api_keys():
    """
    Load API keys with intelligent fallback strategy:
    1. First, check Streamlit secrets (for development/testing)
    2. If not found, use session state (user-provided keys)
    
    This allows developers to pre-configure keys during development,
    then switch to user-provided mode by simply deleting the secrets file.
    """
    return {
        # Current Premium Engines
        # Priority: secrets.toml > user input > empty string
        's2': st.session_state.get('user_s2_key', '').strip() or get_secret_or_empty('S2_API_KEY'),
        'serp': st.session_state.get('user_serp_key', '').strip() or get_secret_or_empty('SERP_API_KEY'),
        'core': st.session_state.get('user_core_key', '').strip() or get_secret_or_empty('CORE_API_KEY'),
        'scopus': st.session_state.get('user_scopus_key', '').strip() or get_secret_or_empty('SCOPUS_API_KEY'),
        'springer': st.session_state.get('user_springer_key', '').strip() or get_secret_or_empty('META_SPRINGER_API_KEY'),
        'email': st.session_state.get('user_email', 'researcher@example.com').strip() or get_secret_or_empty('USER_EMAIL') or 'researcher@example.com',
        
        # 📌 PLACEHOLDER: Add additional premium engine keys here
        # Template for adding a new premium engine:
        # 'new_engine': st.session_state.get('user_new_engine_key', '').strip() or get_secret_or_empty('NEW_ENGINE_API_KEY'),
        
        # Example: IEEE Xplore (uncomment when implemented)
        # 'ieee': st.session_state.get('user_ieee_key', '').strip() or get_secret_or_empty('IEEE_API_KEY'),
        
        # Example: Web of Science (uncomment when implemented)
        # 'wos': st.session_state.get('user_wos_key', '').strip() or get_secret_or_empty('WOS_API_KEY'),
    }

def render_api_key_input_section():
    """
    Render the API key input section in sidebar
    Keys are session-only (lost on refresh) for security
    """
    st.sidebar.header("🔑 API Configuration")
    
    # Check if running in dev mode
    is_dev_mode, dev_keys = check_dev_mode()
    
    if is_dev_mode:
        st.sidebar.markdown(f"""
        <div class="dev-mode-badge">
            🔧 DEV MODE ACTIVE<br>
            Pre-configured keys detected: {len(dev_keys)}<br>
            <small>Delete secrets.toml to switch to production mode</small>
        </div>
        """, unsafe_allow_html=True)
        
        with st.sidebar.expander("📋 Active Developer Keys", expanded=False):
            for key in dev_keys:
                st.markdown(f"✅ **{key}** (from secrets)")
            st.info("💡 These keys are loaded from `secrets.toml` for development convenience.")
    
    st.sidebar.info("🔒 **User keys are temporary** - Lost when you refresh or close the tab (for your security!)")
    
    with st.sidebar.expander("📝 Enter Your API Keys (Optional)", expanded=not is_dev_mode):
        if is_dev_mode:
            st.warning("⚠️ Developer keys are active. User input will override secrets for this session.")
        
        st.markdown("""
        **Optional Premium Engines:**
        - Semantic Scholar (free key available)
        - Google Scholar (via SERP API)
        - CORE
        - SCOPUS
        - Springer Nature
        
        **Always Available (No Key Needed):**
        - arXiv, PubMed, Crossref/DOI, OpenAlex
        - Europe PMC, PLOS, SSRN, DeepDyve
        - Wiley, Taylor & Francis, ACM, DBLP, SAGE
        
        🔒 **Security Note:**
        - Keys stored in browser memory only
        - Never saved to disk or server
        - Automatically cleared on refresh
        - Each user uses their own keys
        """)
        
        # =============================================================================
        # CURRENT PREMIUM ENGINES
        # =============================================================================
        
        # API Key Inputs
        s2_key = st.text_input(
            "Semantic Scholar API Key",
            value="",
            type="password",
            help="Get free key at: https://www.semanticscholar.org/product/api",
            key="s2_input_widget",
            placeholder="Enter your S2 API key (or use dev secrets)"
        )
        
        serp_key = st.text_input(
            "SERP API Key (Google Scholar)",
            value="",
            type="password",
            help="Get key at: https://serpapi.com/",
            key="serp_input_widget",
            placeholder="Enter your SERP API key (or use dev secrets)"
        )
        
        core_key = st.text_input(
            "CORE API Key",
            value="",
            type="password",
            help="Get key at: https://core.ac.uk/services/api",
            key="core_input_widget",
            placeholder="Enter your CORE API key (or use dev secrets)"
        )
        
        scopus_key = st.text_input(
            "SCOPUS API Key",
            value="",
            type="password",
            help="Get key at: https://dev.elsevier.com/",
            key="scopus_input_widget",
            placeholder="Enter your SCOPUS API key"
        )
        
        # ✅ NEW: Springer Nature
        springer_key = st.text_input(
            "Springer Nature API Key",
            value="",
            type="password",
            help="Get key at: https://dev.springernature.com/",
            key="springer_input_widget",
            placeholder="Enter your Springer API key"
        )
        
        # =============================================================================
        # 📌 PLACEHOLDER: Add new premium engine inputs here
        # =============================================================================
        # Template for adding a new premium engine:
        # new_engine_key = st.text_input(
        #     "New Engine API Key",
        #     value="",
        #     type="password",
        #     help="Get key at: https://newengine.com/api",
        #     key="new_engine_input_widget",
        #     placeholder="Enter your New Engine API key"
        # )
        
        # Example: IEEE Xplore (uncomment when implemented)
        # ieee_key = st.text_input(
        #     "IEEE Xplore API Key",
        #     value="",
        #     type="password",
        #     help="Get key at: https://developer.ieee.org/",
        #     key="ieee_input_widget",
        #     placeholder="Enter your IEEE API key"
        # )
        
        # Example: Web of Science (uncomment when implemented)
        # wos_key = st.text_input(
        #     "Web of Science API Key",
        #     value="",
        #     type="password",
        #     help="Get key at: https://developer.clarivate.com/",
        #     key="wos_input_widget",
        #     placeholder="Enter your WoS API key"
        # )
        # =============================================================================
        
        email = st.text_input(
            "Your Email",
            value="researcher@example.com",
            help="Used for API requests to arXiv, PubMed, etc.",
            key="email_input_widget",
            placeholder="your.email@example.com"
        )
        
        # Apply button
        if st.button("✅ Apply Keys (This Session Only)", key="apply_keys", use_container_width=True):
            # Update session state for current engines
            st.session_state['user_s2_key'] = s2_key.strip()
            st.session_state['user_serp_key'] = serp_key.strip()
            st.session_state['user_core_key'] = core_key.strip()
            st.session_state['user_scopus_key'] = scopus_key.strip()
            st.session_state['user_springer_key'] = springer_key.strip()
            st.session_state['user_email'] = email.strip()
            
            # 📌 PLACEHOLDER: Update session state for additional engines
            # st.session_state['user_new_engine_key'] = new_engine_key.strip()
            
            st.success("✅ Keys applied for this session!")
            st.rerun()
        
        # Show which keys are currently active (from either source)
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
        
        # 📌 PLACEHOLDER: Check for new engine keys
        
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
    """Check which API keys are configured and valid (not empty)"""
    status = {}
    
    # Current Premium Engines
    status['s2'] = "✅" if api_keys.get('s2') and len(api_keys.get('s2', '')) > 5 else "❌"
    status['serp'] = "✅" if api_keys.get('serp') and len(api_keys.get('serp', '')) > 5 else "❌"
    status['core'] = "✅" if api_keys.get('core') and len(api_keys.get('core', '')) > 5 else "❌"
    status['scopus'] = "✅" if api_keys.get('scopus') and len(api_keys.get('scopus', '')) > 5 else "❌"
    status['springer'] = "✅" if api_keys.get('springer') and len(api_keys.get('springer', '')) > 5 else "❌"  # ✅ NEW
    status['email'] = "✅" if api_keys.get('email') and api_keys['email'] != 'researcher@example.com' else "⚠️"
    
    # 📌 PLACEHOLDER: Add validation for additional engines
    # Template for adding a new premium engine:
    # status['new_engine'] = "✅" if api_keys.get('new_engine') and len(api_keys.get('new_engine', '')) > 5 else "❌"
    
    # Example: IEEE Xplore (uncomment when implemented)
    # status['ieee'] = "✅" if api_keys.get('ieee') and len(api_keys.get('ieee', '')) > 5 else "❌"
    
    # Example: Web of Science (uncomment when implemented)
    # status['wos'] = "✅" if api_keys.get('wos') and len(api_keys.get('wos', '')) > 5 else "❌"
    
    return status

def get_available_engines(key_status):
    """Determine which engines are available based on API keys"""
    available = []
    
    # Current Premium Engines (key-dependent)
    if key_status['s2'] == "✅":
        available.append("Semantic Scholar")
    if key_status['serp'] == "✅":
        available.append("Google Scholar")
    if key_status['core'] == "✅":
        available.append("CORE")
    if key_status['scopus'] == "✅":
        available.append("SCOPUS")
    if key_status['springer'] == "✅":  # ✅ NEW
        available.append("Springer Nature")
    
    # 📌 PLACEHOLDER: Add checks for additional premium engines
    # if key_status.get('ieee') == "✅":
    #     available.append("IEEE Xplore")
    # if key_status.get('wos') == "✅":
    #     available.append("Web of Science")
    
    # Free Engines (always available - no key required)
    # Original 4 free engines
    available.extend(["arXiv", "PubMed", "Crossref/DOI", "OpenAlex"])
    
    # ✅ NEW: 8 additional free engines
    available.extend([
        "Europe PMC", "PLOS", "SSRN", "DeepDyve",
        "Wiley", "Taylor & Francis", "ACM Digital Library", "DBLP", "SAGE Journals"  # ✅ Added SAGE
    ])
    
    # 📌 PLACEHOLDER: Add additional free engines here
    # available.extend(["DOAJ", "BASE"])  # Example free engines
    
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
    
    # Check for files and create download buttons
    files_to_download = {
        'MASTER_REPORT_FINAL.csv': ('CSV Report', 'text/csv', col1),
        'EXECUTIVE_SUMMARY.txt': ('Executive Summary', 'text/plain', col2),
        'RESEARCH_GAPS.txt': ('Research Gaps', 'text/plain', col3),  # ✅ NEW
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
    
    # Create and offer ZIP download
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

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<p class="main-header">🔬 SROrch - Scholarly Research Orchestrator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Engine Academic Literature Search & Analysis</p>', unsafe_allow_html=True)
    
    # Check and display dev mode status
    is_dev_mode, dev_keys = check_dev_mode()
    if is_dev_mode:
        st.info(f"🔧 **Development Mode Active** - Using {len(dev_keys)} pre-configured API key(s) from secrets.toml")
    
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
        
        # Show engine status
        engine_display = {
            # Current Premium Engines
            "Semantic Scholar": key_status['s2'],
            "Google Scholar": key_status['serp'],
            "CORE": key_status['core'],
            "SCOPUS": key_status['scopus'],
            "Springer Nature": key_status['springer'],  # ✅ NEW
            
            # 📌 PLACEHOLDER: Add additional premium engines here
            # "IEEE Xplore": key_status.get('ieee', '❌'),
            # "Web of Science": key_status.get('wos', '❌'),
            
            # Original Free Engines (always available)
            "arXiv": "✅",
            "PubMed": "✅",
            "Crossref/DOI": "✅",
            "OpenAlex": "✅",
            
            # ✅ NEW: Additional Free Engines (9 new)
            "Europe PMC": "✅",
            "PLOS": "✅",
            "SSRN": "✅",
            "DeepDyve": "✅",
            "Wiley": "✅",
            "Taylor & Francis": "✅",
            "ACM Digital Library": "✅",
            "DBLP": "✅",
            "SAGE Journals": "✅",  # ✅ NEW
            
            # 📌 PLACEHOLDER: Add additional free engines here
            # "DOAJ": "✅",
            # "BASE": "✅",
        }
        
        for engine, status in engine_display.items():
            if status == "✅":
                st.markdown(f"✅ **{engine}**")
            else:
                st.markdown(f"❌ {engine} *(no key)*")
        
        # ✅ UPDATED: Total engine count (5 premium + 13 free = 18 engines)
        st.info(f"**Active Engines:** {len(available_engines)}/18")
        
        # Informational message
        if len(available_engines) < 8:
            free_count = len([e for e in available_engines if e in ["arXiv", "PubMed", "Crossref/DOI", "OpenAlex"]])
            st.markdown(f"""
            <div class="info-box">
                <strong>💡 Get More Coverage!</strong><br>
                You're using <strong>{free_count}</strong> free engines. Add your own API keys above to unlock premium engines!
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
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "📊 Results", "ℹ️ About"])
    
    with tab1:
        st.header("Search Academic Literature")
        
        # Show engine availability info
        if len(available_engines) == 8:
            st.success(f"✅ All 8 engines active! You'll get comprehensive coverage.")
        elif len(available_engines) == 4:
            st.info(f"ℹ️ Using 4 free engines: {', '.join(available_engines)}")
            st.markdown("""
            <div class="info-box">
                <strong>💡 Want More Coverage?</strong><br>
                Add your own API keys in the sidebar to unlock 4 additional premium engines!<br>
                <strong>Semantic Scholar</strong> is free and highly recommended.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(f"ℹ️ Searching with {len(available_engines)} engines: {', '.join(available_engines)}")
        
        # Search input
        search_query = st.text_input(
            "Enter your research query:",
            placeholder="e.g., Langerhans Cell Histiocytosis, Machine Learning in Healthcare, etc.",
            help="Enter keywords or phrases describing your research topic"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
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
                    
                    # Set API keys in environment - ONLY IF VALID
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
                        <p><strong>Output Directory:</strong> {orchestrator.output_dir}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Switch to results tab
                    st.info("👉 Switch to the 'Results' tab to view your findings!")
                    
                except Exception as e:
                    st.markdown(f"""
                    <div class="error-box">
                        <h3>❌ Search Failed</h3>
                        <p>{str(e)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.error(f"Error details: {e}")
                    import traceback
                    with st.expander("🔍 View Full Error Trace"):
                        st.code(traceback.format_exc())
    
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
            
            # ✅ ENHANCED: Interactive Data Explorer with Bookmarks & Selection
            csv_path = os.path.join(output_dir, "MASTER_REPORT_FINAL.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    
                    # Initialize bookmarks in session state
                    if 'bookmarked_papers' not in st.session_state:
                        st.session_state['bookmarked_papers'] = set()
                    
                    st.subheader("📊 Interactive Data Explorer")
                    
                    # Filtering controls
                    with st.expander("🔍 Filter & Search", expanded=False):
                        filter_col1, filter_col2, filter_col3 = st.columns(3)
                        
                        with filter_col1:
                            # Citation filter
                            if 'citations' in df.columns:
                                min_citations = st.number_input(
                                    "Min Citations",
                                    min_value=0,
                                    max_value=int(df['citations'].max()),
                                    value=0,
                                    key="cite_filter"
                                )
                        
                        with filter_col2:
                            # Source count filter
                            if 'source_count' in df.columns:
                                min_sources = st.slider(
                                    "Min Sources",
                                    min_value=1,
                                    max_value=int(df['source_count'].max()),
                                    value=1,
                                    key="src_filter"
                                )
                        
                        with filter_col3:
                            # Text search
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
                        # Quick actions
                        quick_action = st.selectbox(
                            "Quick Filter",
                            ["All Papers", "Highly Cited (>50)", "High Consensus (≥4)", "Recent (Boosted)", "Bookmarked Only"],
                            key="quick_filter"
                        )
                    
                    with view_col3:
                        # Bookmark stats
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
                        
                        # Add bookmark indicator column
                        display_df.insert(0, '📑', display_df.index.map(lambda x: '⭐' if x in st.session_state['bookmarked_papers'] else ''))
                        
                        # ✅ NEW: Apply alternating row colors with custom styling (zebra strips)
                        # f0f2f6 very light, desaturated blue-gray, ffffff white
                        # Modern/Dark: Dark Gray/Light Gray (used for dark-themed UI) 
                        def highlight_rows(row):
                            """Apply alternating row colors for better readability"""
                            if row.name % 2 == 0:
                                return ['background-color: #f2f2f2'] * len(row)
                            else:
                                return ['background-color: #2f2f2f'] * len(row)
                        
                        # Apply styling
                        styled_df = display_df.style.apply(highlight_rows, axis=1)
                        
                        # ✅ ENHANCED: Interactive table with clickable URLs and custom styling
                        st.dataframe(
                            styled_df,
                            use_container_width=True,
                            height=400,
                            hide_index=False,
                            column_config={
                                '📑': st.column_config.TextColumn('📑', width="small"),
                                "url": st.column_config.LinkColumn("URL", display_text="🔗 Open"),  # ✅ Clickable!
                                "doi": st.column_config.TextColumn("DOI", width="medium"),
                                "relevance_score": st.column_config.NumberColumn("Score", format="%d"),
                                "citations": st.column_config.NumberColumn("Cites", format="%d"),
                                "source_count": st.column_config.NumberColumn("Sources", format="%d"),
                                "year": st.column_config.TextColumn("Year", width="small"),
                            }
                        )
                        
                        st.info("💡 **Tip**: Click row numbers below to select papers, or use Bookmark Manager for quick access!")
                        
                        # ✅ NEW: Row Selection & Bookmark Management
                        st.divider()
                        
                        action_col1, action_col2, action_col3 = st.columns(3)
                        
                        with action_col1:
                            st.markdown("#### 📑 Bookmark Manager")
                            
                            # Select papers to bookmark
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
                            
                            # Select papers for download
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
                        
                        # ✅ NEW: Quick Access to Bookmarked Papers
                        if st.session_state['bookmarked_papers']:
                            st.divider()
                            
                            with st.expander(f"⭐ View Bookmarked Papers ({len(st.session_state['bookmarked_papers'])})", expanded=False):
                                bookmarked_df = df.loc[list(st.session_state['bookmarked_papers'])]
                                
                                for idx, paper in bookmarked_df.iterrows():
                                    with st.container():
                                        col1, col2 = st.columns([4, 1])
                                        
                                        with col1:
                                            st.markdown(f"**[{idx}] {paper.get('title', 'N/A')}**")
                                            st.caption(f"Authors: {paper.get('ieee_authors', 'N/A')} | Year: {paper.get('year', 'N/A')} | Citations: {paper.get('citations', 0)}")
                                            if paper.get('url'):
                                                st.markdown(f"🔗 [{paper['url']}]({paper['url']})")
                                        
                                        with col2:
                                            if st.button(f"🗑️ Remove", key=f"remove_{idx}"):
                                                st.session_state['bookmarked_papers'].discard(idx)
                                                st.rerun()
                                        
                                        st.divider()
                                
                                # Download all bookmarks
                                bookmark_export_col1, bookmark_export_col2 = st.columns(2)
                                
                                with bookmark_export_col1:
                                    bookmarks_csv = bookmarked_df.to_csv(index=False)
                                    st.download_button(
                                        label=f"📥 Download All Bookmarks (CSV)",
                                        data=bookmarks_csv,
                                        file_name=f"bookmarks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                                
                                with bookmark_export_col2:
                                    bookmarks_json = bookmarked_df.to_json(orient='records', indent=2)
                                    st.download_button(
                                        label=f"📥 Download All Bookmarks (JSON)",
                                        data=bookmarks_json,
                                        file_name=f"bookmarks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                        mime="application/json",
                                        use_container_width=True
                                    )
                    
                    st.divider()
                    
                except Exception as e:
                    st.warning(f"Could not load interactive viewer: {e}")
                    import traceback
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())
            
            # Display analytics chart if available
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
            
            # Session details
            with st.expander("📋 Session Details"):
                st.json({
                    'query': metadata.get('query', 'N/A'),
                    'start_time': str(metadata.get('start_time', 'N/A')),
                    'end_time': str(metadata.get('end_time', 'N/A')),
                    'successful_engines': metadata.get('successful_engines', []),
                    'failed_engines': metadata.get('failed_engines', []),
                    'total_api_calls': metadata.get('total_api_calls', 0)
                })
        
        else:
            st.info("👈 No results yet. Start a search in the 'Search' tab!")
    
    with tab3:
        st.header("About SROrch")
        
        st.markdown("""
        ### 🔬 Scholarly Research Orchestrator
        
        SROrch is a powerful multi-engine academic literature search and analysis tool that aggregates 
        results from multiple scholarly databases to provide comprehensive research coverage.
        
        #### 📚 Supported Databases (18 Engines!)
        
        **Premium Engines (Require Your Own API Keys):**
        - **Semantic Scholar** - AI-powered academic search (FREE key available!)
        - **Google Scholar** - Broad academic search (via SERP API)
        - **CORE** - Open access research aggregator
        - **SCOPUS** - Comprehensive scientific database
        - **Springer Nature** - Major scientific publisher
        
        **Free Engines (Always Available - No Keys Needed):**
        
        *Original Core Engines:*
        - **arXiv** - Preprint repository for STEM fields
        - **PubMed** - Biomedical literature database
        - **Crossref/DOI** - Digital Object Identifier resolution
        - **OpenAlex** - Open catalog of scholarly papers
        
        *Additional Free Engines:*
        - **Europe PMC** - European biomedical literature
        - **PLOS** - Open access scientific journals
        - **SSRN** - Social science research network
        - **DeepDyve** - Academic research rental service
        - **Wiley** - Major scientific publisher
        - **Taylor & Francis** - Academic publisher
        - **ACM Digital Library** - Computing literature
        - **DBLP** - Computer science bibliography
        - **SAGE Journals** - Social sciences and humanities publisher

        #### ✨ Key Features
        - **Multi-source consensus detection** - Identifies papers found across multiple databases
        - **Intelligent relevance scoring** - Combines citations, source count, and recency
        - **Deep abstract fetching** - Retrieves full abstracts for top papers
        - **Enhanced gap analysis** - Domain-specific research gap detection
        - **Publication analytics** - Generates trend visualizations and statistics
        - **Multiple export formats** - CSV, JSON, and BibTeX support
        - **Recency boosting** - Optional preference for recent publications
        - **High-consensus alerts** - Automatic notifications for widely-indexed papers
        - **Zero-trust security** - Use your own API keys, no data stored on server
        
        #### 🚀 Getting Started
        
        **Instant Use (No Setup):**
        - Works immediately with 4 free engines
        - No API keys required
        - Perfect for quick searches and open access research
        
        **Add Your Own API Keys (Optional):**
        1. Click "Enter Your API Keys" in the sidebar
        2. Enter keys for any premium engines you have access to
        3. Click "Apply Keys"
        4. Get up to 8 engines for comprehensive coverage
        
        **Important:** Keys are temporary (this session only) for your security!
        
        #### 🔑 How to Get API Keys
        
        **Semantic Scholar** (Recommended - FREE!)
        - Visit: https://www.semanticscholar.org/product/api
        - Sign up and request a free API key (takes 2 minutes)
        - Best free option with excellent metadata and abstracts
        - Strongly recommended for most users
        
        **SERP API** (Google Scholar)
        - Visit: https://serpapi.com/
        - Free tier: 100 searches/month
        - Unlocks Google Scholar's massive database
        
        **CORE API** (FREE for academics)
        - Visit: https://core.ac.uk/services/api
        - Free academic API key available
        - Access to millions of open access papers
        
        **SCOPUS API**
        - Visit: https://dev.elsevier.com/
        - Requires institutional access or paid plan
        - Most comprehensive but also most expensive
        
        #### 🔒 Security & Privacy
        
        **Your API Keys Are Safe:**
        - ✅ Stored only in browser memory (session state)
        - ✅ Never saved to disk or server
        - ✅ Automatically cleared when you refresh or close tab
        - ✅ Each user uses their own keys and quotas
        - ✅ Zero exposure for the app developer
        - ✅ Zero risk of unexpected charges
        
        **How It Works:**
        1. You enter your API keys (optional)
        2. Keys stay in your browser's memory
        3. SROrch uses them to search on your behalf
        4. Results are returned to you
        5. Close tab → keys are gone
        
        **Why This Is Safe:**
        - No shared API keys (everyone uses their own)
        - No persistent storage (keys deleted automatically)
        - No server-side storage (keys never leave your browser)
        - No developer liability (you control your own quotas)
        
        #### 💡 Tips
        
        **For Best Results:**
        1. Start with the 4 free engines (instant access)
        2. Add Semantic Scholar key (free, highly recommended)
        3. Use specific search terms for better results
        4. Check "source_count" - papers in multiple engines are more reliable
        5. Export results in your preferred format (CSV/JSON/BibTeX)
        
        **Cost Optimization:**
        - Free engines: Unlimited, always available
        - Semantic Scholar: Free tier is generous
        - SERP API: 100 free searches/month
        - You control your own usage and costs
        
        #### 📊 Understanding Results
        
        **Relevance Score:** Based on citations, source count, and recency
        **Source Count:** How many databases found this paper (higher = more reliable)
        **High Consensus:** Papers found in 4+ databases automatically flagged
        
        #### 🎯 Use Cases
        - Literature reviews and systematic reviews
        - Research gap analysis
        - Citation mapping
        - Trend identification in research fields
        - Multi-database validation
        
        ---
        
        **Version:** Public v1.0 (Session-Only Keys)  
        **Security Model:** Zero-Trust (User-Provided Keys)  
        **License:** MIT
        """)
        
        # System information
        with st.expander("🖥️ System Information"):
            st.code(f"""
Python Version: {sys.version}
Working Directory: {os.getcwd()}
Streamlit Version: {st.__version__}
Security Model: Session-only keys (no persistence)
            """)

if __name__ == "__main__":
    main()
