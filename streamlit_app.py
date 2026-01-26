"""
SROrch Streamlit Interface
A web interface for the Scholarly Research Orchestrator
Enhanced with User-Provided API Keys Support
"""

import streamlit as st
import os
import sys
import json
import shutil
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
    </style>
""", unsafe_allow_html=True)

def load_api_keys():
    """
    Load API keys with flexibility:
    1. Try to load from Streamlit secrets (admin mode)
    2. Fall back to user-provided keys via session state
    """
    # Check if admin mode is enabled in secrets
    admin_mode = False
    try:
        admin_mode = st.secrets.get("ADMIN_MODE", False)
    except:
        pass
    
    if admin_mode:
        # Admin mode: Use secrets
        try:
            return {
                's2': st.secrets.get("S2_API_KEY", ""),
                'serp': st.secrets.get("SERP_API_KEY", ""),
                'core': st.secrets.get("CORE_API_KEY", ""),
                'scopus': st.secrets.get("SCOPUS_API_KEY", ""),
                'email': st.secrets.get("USER_EMAIL", "researcher@example.com")
            }, True  # Return admin_mode flag
        except Exception as e:
            st.error(f"Error loading admin secrets: {e}")
            admin_mode = False
    
    # User mode: Get keys from session state (populated by sidebar inputs)
    return {
        's2': st.session_state.get('user_s2_key', ''),
        'serp': st.session_state.get('user_serp_key', ''),
        'core': st.session_state.get('user_core_key', ''),
        'scopus': st.session_state.get('user_scopus_key', ''),
        'email': st.session_state.get('user_email', 'researcher@example.com')
    }, False  # Not admin mode

def render_api_key_input_section():
    """
    Render the API key input section in sidebar for user-provided keys
    """
    st.sidebar.header("🔑 API Configuration")
    
    # Check if admin mode
    admin_mode = False
    try:
        admin_mode = st.secrets.get("ADMIN_MODE", False)
    except:
        pass
    
    if admin_mode:
        st.sidebar.success("🔒 Admin Mode Active - Using Configured Keys")
        return
    
    # User mode: Allow key input
    st.sidebar.info("💡 **Tip:** Enter your API keys below or use free engines (arXiv, PubMed, Crossref, OpenAlex)")
    
    with st.sidebar.expander("📝 Enter Your API Keys (Optional)", expanded=False):
        st.markdown("""
        **Optional Premium Engines:**
        - Semantic Scholar (recommended)
        - Google Scholar (via SERP API)
        - CORE
        - SCOPUS
        
        **Always Available (No Key Needed):**
        - arXiv
        - PubMed
        - Crossref/DOI
        - OpenAlex
        """)
        
        # API Key Inputs
        s2_key = st.text_input(
            "Semantic Scholar API Key",
            value=st.session_state.get('user_s2_key', ''),
            type="password",
            help="Get free key at: https://www.semanticscholar.org/product/api",
            key="s2_input"
        )
        
        serp_key = st.text_input(
            "SERP API Key (Google Scholar)",
            value=st.session_state.get('user_serp_key', ''),
            type="password",
            help="Get key at: https://serpapi.com/",
            key="serp_input"
        )
        
        core_key = st.text_input(
            "CORE API Key",
            value=st.session_state.get('user_core_key', ''),
            type="password",
            help="Get key at: https://core.ac.uk/services/api",
            key="core_input"
        )
        
        scopus_key = st.text_input(
            "SCOPUS API Key",
            value=st.session_state.get('user_scopus_key', ''),
            type="password",
            help="Get key at: https://dev.elsevier.com/",
            key="scopus_input"
        )
        
        email = st.text_input(
            "Your Email",
            value=st.session_state.get('user_email', 'researcher@example.com'),
            help="Used for API requests to arXiv, PubMed, etc.",
            key="email_input"
        )
        
        # Update session state
        if st.button("💾 Save API Keys", key="save_keys"):
            st.session_state['user_s2_key'] = s2_key
            st.session_state['user_serp_key'] = serp_key
            st.session_state['user_core_key'] = core_key
            st.session_state['user_scopus_key'] = scopus_key
            st.session_state['user_email'] = email
            st.success("✅ API keys saved for this session!")
            st.rerun()

def check_api_keys(api_keys):
    """Check which API keys are configured"""
    status = {}
    status['s2'] = "✅" if api_keys.get('s2') else "❌"
    status['serp'] = "✅" if api_keys.get('serp') else "❌"
    status['core'] = "✅" if api_keys.get('core') else "❌"
    status['scopus'] = "✅" if api_keys.get('scopus') else "❌"
    status['email'] = "✅" if api_keys.get('email') and api_keys['email'] != 'researcher@example.com' else "⚠️"
    return status

def get_available_engines(key_status, api_keys):
    """Determine which engines are available based on API keys"""
    available = []
    
    # Key-dependent engines
    if key_status['s2'] == "✅":
        available.append("Semantic Scholar")
    if key_status['serp'] == "✅":
        available.append("Google Scholar")
    if key_status['core'] == "✅":
        available.append("CORE")
    if key_status['scopus'] == "✅":
        available.append("SCOPUS")
    
    # Always available engines (no key required)
    available.extend(["arXiv", "PubMed", "Crossref/DOI", "OpenAlex"])
    
    return list(set(available))

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
        'research_data.json': ('JSON Data', 'application/json', col3),
        'references.bib': ('BibTeX', 'text/plain', col1),
        'research_analytics.png': ('Analytics Chart', 'image/png', col2),
        'SESSION_REPORT.txt': ('Session Report', 'text/plain', col3)
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
    # Header
    st.markdown('<p class="main-header">🔬 SROrch - Scholarly Research Orchestrator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Engine Academic Literature Search & Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Input Section (NEW)
        render_api_key_input_section()
        
        st.divider()
        
        # Load API keys and check status
        api_keys, is_admin = load_api_keys()
        key_status = check_api_keys(api_keys)
        available_engines = get_available_engines(key_status, api_keys)
        
        # Engine Status Display
        st.subheader("🔍 Available Engines")
        
        # Show engine status
        engine_display = {
            "Semantic Scholar": key_status['s2'],
            "Google Scholar": key_status['serp'],
            "CORE": key_status['core'],
            "SCOPUS": key_status['scopus'],
            "arXiv": "✅",
            "PubMed": "✅",
            "Crossref/DOI": "✅",
            "OpenAlex": "✅"
        }
        
        for engine, status in engine_display.items():
            if status == "✅":
                st.markdown(f"✅ **{engine}**")
            else:
                st.markdown(f"❌ {engine} *(no key)*")
        
        st.info(f"**Active Engines:** {len(available_engines)}/8")
        
        # Informational message
        if len(available_engines) < 8:
            st.markdown("""
            <div class="info-box">
                <strong>💡 Get More Coverage!</strong><br>
                You're using <strong>{}</strong> free engines. Add API keys above to unlock premium engines for better results!
            </div>
            """.format(len([e for e in available_engines if e in ["arXiv", "PubMed", "Crossref/DOI", "OpenAlex"]])), unsafe_allow_html=True)
        
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
            elif len(available_engines) < 4:
                st.warning("⚠️ You have fewer than 4 engines available. Results may be limited. Consider adding more API keys for better coverage.")
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
                        if key != 'email' and value:  # Only set if not empty
                            os.environ[f"{key.upper()}_API_KEY"] = value
                        elif key == 'email':
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
        
        #### 📚 Supported Databases
        
        **Premium Engines (Require API Keys):**
        - **Semantic Scholar** - AI-powered academic search (free key available)
        - **Google Scholar** - Broad academic search (via SERP API)
        - **CORE** - Open access research aggregator
        - **SCOPUS** - Comprehensive scientific database
        
        **Free Engines (Always Available):**
        - **arXiv** - Preprint repository for STEM fields
        - **PubMed** - Biomedical literature database
        - **Crossref/DOI** - Digital Object Identifier resolution
        - **OpenAlex** - Open catalog of scholarly papers

        #### ✨ Key Features
        - **Multi-source consensus detection** - Identifies papers found across multiple databases
        - **Intelligent relevance scoring** - Combines citations, source count, and recency
        - **Deep abstract fetching** - Retrieves full abstracts for top papers
        - **Publication analytics** - Generates trend visualizations and statistics
        - **Multiple export formats** - CSV, JSON, and BibTeX support
        - **Recency boosting** - Optional preference for recent publications
        - **High-consensus alerts** - Automatic notifications for widely-indexed papers
        - **Flexible API key management** - Use your own keys or free engines
        
        #### 🎯 Use Cases
        - Literature reviews and systematic reviews
        - Research gap analysis
        - Citation mapping
        - Trend identification in research fields
        - Multi-database validation
        
        #### 🔧 Getting Started
        
        **Option 1: Use Free Engines Only**
        - No API keys required
        - Access to 4 engines: arXiv, PubMed, Crossref, OpenAlex
        - Perfect for quick searches and open access research
        
        **Option 2: Add Your API Keys**
        - Click "Enter Your API Keys" in the sidebar
        - Add keys for premium engines (Semantic Scholar recommended)
        - Get up to 8 engines for comprehensive coverage
        
        #### 🔑 How to Get API Keys
        
        **Semantic Scholar** (Recommended - Free)
        - Visit: https://www.semanticscholar.org/product/api
        - Sign up and request a free API key
        - Provides excellent metadata and abstracts
        
        **SERP API** (Google Scholar)
        - Visit: https://serpapi.com/
        - Free tier available with 100 searches/month
        - Unlocks Google Scholar integration
        
        **CORE API**
        - Visit: https://core.ac.uk/services/api
        - Free academic API key available
        - Access to millions of open access papers
        
        **SCOPUS API**
        - Visit: https://dev.elsevier.com/
        - Requires institutional access or paid plan
        - Premium scientific database
        
        #### 📖 How to Use
        1. (Optional) Add your API keys in the sidebar
        2. Enter your research query in the Search tab
        3. Adjust settings in the sidebar as needed
        4. Click "Start Search" and wait for results
        5. View analytics and download reports in the Results tab
        
        #### 📝 Output Files
        - `MASTER_REPORT_FINAL.csv` - Complete results table
        - `EXECUTIVE_SUMMARY.txt` - Top papers with abstracts
        - `research_data.json` - Structured JSON export
        - `references.bib` - BibTeX citations
        - `research_analytics.png` - Visualization dashboard
        - `SESSION_REPORT.txt` - Search session metadata
        
        #### 🔒 Privacy & Security
        - Your API keys are only stored in your browser session
        - Keys are never saved to disk or shared
        - Each user provides their own keys
        - No data is collected or logged
        
        ---
        
        **Version:** Enhanced v2.1 (User-Provided Keys)  
        **Author:** Research Tools Team  
        **License:** MIT
        """)
        
        # System information
        with st.expander("🖥️ System Information"):
            st.code(f"""
Python Version: {sys.version}
Working Directory: {os.getcwd()}
Streamlit Version: {st.__version__}
            """)

if __name__ == "__main__":
    main()
