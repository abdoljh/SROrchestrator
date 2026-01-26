"""
SROrch Streamlit Interface
A web interface for the Scholarly Research Orchestrator
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
# #dad4ed Soft Lavender
# #d4edda Silky Mint Green
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
        background-color: #dad4ed; 
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
    </style>
""", unsafe_allow_html=True)

def load_api_keys():
    """Load API keys from Streamlit secrets"""
    try:
        return {
            's2': st.secrets.get("S2_API_KEY", ""),
            'serp': st.secrets.get("SERP_API_KEY", ""),
            'core': st.secrets.get("CORE_API_KEY", ""),
            'scopus': st.secrets.get("SCOPUS_API_KEY", ""),
            'email': st.secrets.get("USER_EMAIL", "researcher@example.com")
        }
    except Exception as e:
        st.error(f"Error loading secrets: {e}")
        return {
            's2': "",
            'serp': "",
            'core': "",
            'scopus': "",
            'email': "researcher@example.com"
        }

def check_api_keys(api_keys):
    """Check which API keys are configured"""
    status = {}
    status['s2'] = "✅" if api_keys.get('s2') else "❌"
    status['serp'] = "✅" if api_keys.get('serp') else "❌"
    status['core'] = "✅" if api_keys.get('core') else "❌"
    status['scopus'] = "✅" if api_keys.get('scopus') else "❌"
    return status

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
                width="stretch"
            )

def main():
    # Header
    st.markdown('<p class="main-header">🔬 SROrch - Scholarly Research Orchestrator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Engine Academic Literature Search & Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Status
        st.subheader("🔑 API Key Status")
        api_keys = load_api_keys()
        key_status = check_api_keys(api_keys)
        
        st.write(f"Semantic Scholar: {key_status['s2']}")
        st.write(f"SERP API (Google Scholar): {key_status['serp']}")
        st.write(f"CORE API: {key_status['core']}")
        st.write(f"SCOPUS API: {key_status['scopus']}")
        st.write(f"Email (arXiv/PubMed/etc): {'✅' if api_keys.get('email') and api_keys['email'] != 'researcher@example.com' else '⚠️'}")
        
        # Show available engines count
        available_engines = []
        if key_status['s2'] == "✅":
            available_engines.append("Semantic Scholar")
        if key_status['serp'] == "✅":
            available_engines.append("Google Scholar")
        if key_status['core'] == "✅":
            available_engines.append("CORE")
        if key_status['scopus'] == "✅":
            available_engines.append("SCOPUS")
        if api_keys.get('email'):
            available_engines.extend(["arXiv", "PubMed", "Crossref", "OpenAlex"])
        
        st.info(f"**Available Engines:** {len(set(available_engines))}/8")
        
        # Warning for missing critical keys
        if not any([key_status['s2'] == "✅", key_status['serp'] == "✅", key_status['core'] == "✅", key_status['scopus'] == "✅", api_keys.get('email')]):
            st.error("⚠️ No API keys configured! Application may not work properly.")
        elif len(set(available_engines)) < 4:
            st.warning(f"⚠️ Only {len(set(available_engines))} engines available. Configure more API keys for better coverage.")
        
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
        
        # Search input
        search_query = st.text_input(
            "Enter your research query:",
            placeholder="e.g., Langerhans Cell Histiocytosis, Machine Learning in Healthcare, etc.",
            help="Enter keywords or phrases describing your research topic"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_button = st.button("🚀 Start Search", type="primary", width="stretch")
        
        with col2:
            if st.button("🔄 Clear Cache", width="stretch"):
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
                        if key != 'email':
                            os.environ[f"{key.upper()}_API_KEY"] = value
                        else:
                            os.environ['USER_EMAIL'] = value
                    
                    # Initialize orchestrator
                    orchestrator = ResearchOrchestrator(config=config)
                    
                    status_text.text("🔍 Searching across multiple databases...")
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
                st.image(chart_path, width="stretch")
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
        - **Semantic Scholar** - AI-powered academic search
        - **arXiv** - Preprint repository for STEM fields
        - **PubMed** - Biomedical literature database
        - **Google Scholar** - Broad academic search (via SERP API)
        - **Crossref/DOI** - Digital Object Identifier resolution
        - **OpenAlex** - Open catalog of scholarly papers
        - **CORE** - Open access research aggregator
        - **SCOPUS** - Open access research aggregator

        #### ✨ Key Features
        - **Multi-source consensus detection** - Identifies papers found across multiple databases
        - **Intelligent relevance scoring** - Combines citations, source count, and recency
        - **Deep abstract fetching** - Retrieves full abstracts for top papers
        - **Publication analytics** - Generates trend visualizations and statistics
        - **Multiple export formats** - CSV, JSON, and BibTeX support
        - **Recency boosting** - Optional preference for recent publications
        - **High-consensus alerts** - Automatic notifications for widely-indexed papers
        
        #### 🎯 Use Cases
        - Literature reviews and systematic reviews
        - Research gap analysis
        - Citation mapping
        - Trend identification in research fields
        - Multi-database validation
        
        #### 🔧 Configuration
        Configure your search using the sidebar settings:
        - Adjust scoring weights for citations and sources
        - Enable/disable recency boosting
        - Set consensus thresholds
        - Choose export formats
        - Control visualization generation
        
        #### 📖 How to Use
        1. Configure your API keys in Streamlit secrets
        2. Enter your research query in the Search tab
        3. Adjust settings in the sidebar as needed
        4. Click "Start Search" and wait for results
        5. View analytics and download reports in the Results tab
        
        #### 🔑 API Keys Required
        - **S2_API_KEY** - Semantic Scholar API key (recommended)
        - **SERP_API_KEY** - SerpAPI key for Google Scholar (optional)
        - **CORE_API_KEY** - CORE API key (optional)
        - **SCOPUS_API_KEY** - SCOPUS API key (optional)
        - **USER_EMAIL** - Your email for API requests (required for some services)
        
        #### 📝 Output Files
        - `MASTER_REPORT_FINAL.csv` - Complete results table
        - `EXECUTIVE_SUMMARY.txt` - Top papers with abstracts
        - `research_data.json` - Structured JSON export
        - `references.bib` - BibTeX citations
        - `research_analytics.png` - Visualization dashboard
        - `SESSION_REPORT.txt` - Search session metadata
        
        ---
        
        **Version:** Enhanced v2.0  
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
