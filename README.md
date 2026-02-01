# SROrch - Scholarly Research Orchestrator

**Multi-Engine Academic Literature Search & Analysis Platform**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Security](https://img.shields.io/badge/Security-Session--Only-brightgreen.svg)](#security)

---

## 🎯 Overview

SROrch orchestrates comprehensive academic literature searches across **18 scholarly databases** simultaneously, providing:

- **Multi-source consensus detection** - Find papers validated across multiple databases
- **Intelligent relevance scoring** - Combine citations, source count, and recency
- **Research gap analysis** - Identify underexplored areas with domain-specific patterns
- **Deep abstract fetching** - Retrieve detailed metadata for top papers
- **Publication analytics** - Generate trend visualizations and statistics
- **Flexible API key management** - Dev mode for testing, production mode for deployment

---

## ✨ Key Features

### 🔍 Search Capabilities
- **18 Scholarly Databases**: 5 premium + 13 free engines
- **Parallel Processing**: Search all engines simultaneously
- **Smart Deduplication**: Identifies same papers across different sources
- **Relevance Scoring**: Weighted combination of citations, sources, and recency
- **Deep Look Mode**: Fetches full abstracts for top-ranked papers

### 📊 Analytics & Insights
- **Research Gap Detection**: Domain-specific pattern recognition
- **Citation Analysis**: Track impact and influence
- **Publication Trends**: Visualize temporal patterns
- **Cross-Database Coverage**: Understand source consensus
- **Quality Metrics**: Multi-dimensional paper assessment

### 🔐 Security & Flexibility
- **Dual-Mode Operation**: Development (auto-loaded keys) + Production (user keys)
- **Session-Only Storage**: Keys never persisted to disk (production)
- **Smart Detection**: Automatically chooses best available key source
- **Zero-Trust Design**: Users bring their own API keys
- **Easy Deployment**: Delete secrets.toml to switch modes

### 📦 Export Options
- **CSV Reports**: Structured data for analysis
- **JSON Data**: Machine-readable format
- **BibTeX**: Direct citation import
- **Research Summaries**: Executive overviews
- **Gap Analysis**: Detailed opportunity reports

---

## 🚀 Quick Start

### Option 1: Try Immediately (Free, No Setup)

```bash
streamlit run streamlit_app.py
```

✅ Works instantly with 13 free engines  
✅ No API keys required  
✅ No registration needed  

### Option 2: Enhanced Coverage (5 minutes)

```bash
# 1. Get free Semantic Scholar key (highly recommended)
# Visit: https://www.semanticscholar.org/product/api

# 2. Create secrets file
cp secrets.toml.template .streamlit/secrets.toml

# 3. Add your key
echo 'S2_API_KEY = "your-key-here"' >> .streamlit/secrets.toml

# 4. Run
streamlit run streamlit_app.py
```

✅ 14 engines active  
✅ Excellent metadata quality  
✅ Keys auto-loaded on startup  

---

## 📚 Supported Databases

### Premium Engines (Optional - Require API Keys)

| Engine | Cost | Coverage | Quality | Get Key |
|--------|------|----------|---------|---------|
| **Semantic Scholar** | FREE! | 200M+ papers | ⭐⭐⭐⭐⭐ | [semanticscholar.org](https://www.semanticscholar.org/product/api) |
| **Google Scholar** | $0.002/search | Largest | ⭐⭐⭐⭐⭐ | [serpapi.com](https://serpapi.com/) |
| **CORE** | FREE (academic) | 200M+ | ⭐⭐⭐⭐ | [core.ac.uk](https://core.ac.uk/services/api) |
| **SCOPUS** | Institutional | 80M+ | ⭐⭐⭐⭐⭐ | [dev.elsevier.com](https://dev.elsevier.com/) |
| **Springer Nature** | Varies | 10M+ | ⭐⭐⭐⭐ | [dev.springernature.com](https://dev.springernature.com/) |

### Free Engines (Always Available - No Keys Needed)

| Engine | Specialty | Coverage |
|--------|-----------|----------|
| **arXiv** | STEM preprints | 2M+ |
| **PubMed** | Biomedical | 35M+ |
| **Crossref/DOI** | DOI resolution | 140M+ |
| **OpenAlex** | Open access | 250M+ |
| **Europe PMC** | Life sciences | 42M+ |
| **PLOS** | Open access | 300K+ |
| **SSRN** | Social sciences | 1M+ |
| **DeepDyve** | Rentals | 15M+ |
| **Wiley** | Scientific | 5M+ |
| **Taylor & Francis** | Broad | 2.5M+ |
| **ACM** | Computer science | 600K+ |
| **DBLP** | CS bibliography | 6M+ |
| **SAGE** | Social sciences | 1K+ journals |

**Total: 18 engines (5 premium + 13 free)**

---

## 🎛️ Dual-Mode Architecture

### Development Mode 🔧

**For:** Local development, testing, rapid iteration

**Setup:**
```bash
cp secrets.toml.template .streamlit/secrets.toml
nano .streamlit/secrets.toml  # Add your keys
```

**Benefits:**
- ✅ Keys auto-loaded on startup
- ✅ No manual entry each session
- ✅ Fast development workflow
- ✅ Easy key management
- ✅ Refresh-persistent keys

**UI Indicator:**
```
🔧 DEV MODE ACTIVE
Pre-configured keys detected: 3
```

### Production Mode 👥

**For:** Public deployment, end users, maximum security

**Setup:**
```bash
# Just don't create secrets.toml
# OR delete it before deployment
rm .streamlit/secrets.toml
```

**Benefits:**
- ✅ Users provide own keys
- ✅ Zero developer liability
- ✅ Session-only storage
- ✅ Maximum security
- ✅ No quota sharing

**UI Indicator:**
```
🔒 Keys are temporary - Lost when you refresh
```

### Smart Detection Logic

```python
Priority: User Input → Secrets File → Empty String

1. User enters key in UI → USE IT (highest priority)
2. No user input, secrets exist → USE SECRETS
3. Neither available → USE FREE ENGINES ONLY
```

---

## 🔐 Security

### Development Mode Security

**Protected by:**
- Local-only file storage
- Each developer uses own keys
- Clear documentation warnings

**Best Practices:**
```bash
# 1. Always verify .gitignore
cat .gitignore | grep secrets.toml

# 2. Never share secrets.toml
# 3. Use development/test API keys only
# 4. Set up billing alerts
# 5. Rotate keys regularly
```

### Production Mode Security

**Protected by:**
- Session-only key storage (browser memory)
- Auto-clear on refresh/close
- No disk persistence
- No server-side storage
- Each user manages own quotas

**Guarantee:**
```
❌ Keys never saved to disk
❌ Keys never sent to server
❌ Keys never shared between users
✅ Keys cleared on page refresh
✅ Complete user control
```

---

## 🛠️ Installation

### Prerequisites
```bash
Python 3.8+
pip or conda
```

### Core Dependencies
```bash
pip install streamlit pandas requests matplotlib python-dotenv
```

### Optional (for enhanced features)
```bash
pip install arxiv biopython crossref-commons scholarly
```

### Complete Setup
```bash
# 1. Clone repository
git clone <your-repo-url>
cd srorch

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Configure dev mode
cp secrets.toml.template .streamlit/secrets.toml
nano .streamlit/secrets.toml

# 4. Run
streamlit run streamlit_app.py
```

---

## 📊 Usage Examples

### Example 1: Basic Search
```python
from master_orchestrator import ResearchOrchestrator

orchestrator = ResearchOrchestrator()
results = orchestrator.run_search("machine learning healthcare", limit_per_engine=25)
orchestrator.save_master_csv(results, "machine learning healthcare")
```

### Example 2: Custom Configuration
```python
config = {
    'abstract_limit': 15,
    'citation_weight': 2.0,
    'recency_boost': True,
    'recency_years': 3,
    'high_consensus_threshold': 5
}

orchestrator = ResearchOrchestrator(config=config)
results = orchestrator.run_search("quantum computing", limit_per_engine=30)
```

### Example 3: Streamlit Interface
```bash
streamlit run streamlit_app.py
```
1. Enter search query
2. Adjust parameters (optional)
3. Click "Start Search"
4. View results and analytics
5. Download reports

---

## 🎯 Use Cases

### Academic Research
- ✅ Literature reviews
- ✅ Systematic reviews
- ✅ Meta-analyses
- ✅ Research gap identification
- ✅ Citation tracking

### Industry Applications
- ✅ Competitive intelligence
- ✅ Technology scouting
- ✅ Patent landscaping
- ✅ Trend analysis
- ✅ Expert identification

### Educational
- ✅ Teaching research methods
- ✅ Student projects
- ✅ Library services
- ✅ Information literacy

---

## 🔄 Workflow

```
┌─────────────────┐
│   Enter Query   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Parallel Search │──▶ 18 Databases Searched Simultaneously
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Deduplicate    │──▶ Merge Results, Identify Duplicates
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Relevance Score │──▶ Weight: Citations + Sources + Recency
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Deep Look     │──▶ Fetch Full Abstracts for Top Papers
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Gap Analysis   │──▶ Identify Research Opportunities
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Analytics     │──▶ Generate Charts & Statistics
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Export      │──▶ CSV, JSON, BibTeX, Reports
└─────────────────┘
```

---

## 📈 Performance

### Typical Search Times

| Configuration | Time | Papers | Engines |
|--------------|------|--------|---------|
| Free only | 15-30s | 200-300 | 13 |
| + Premium (cached) | 20-40s | 300-400 | 16-18 |
| + Premium (uncached) | 30-60s | 400-500 | 16-18 |

**Note:** Times vary based on query complexity and API response times

### Optimization Tips
1. Use specific queries (better relevance)
2. Enable caching for repeated searches
3. Adjust `limit_per_engine` based on needs
4. Consider rate limits for premium APIs

---

## 🤝 Contributing

### Adding New Engines

1. **Create utils file**: `new_engine_utils.py`
2. **Implement fetch function**: `fetch_and_process_newengine()`
3. **Add to orchestrator**: Import and call in `run_search()`
4. **Update UI**: Add to engine list in `streamlit_app.py`
5. **Document**: Update README and guides

### Improving Gap Analysis

1. **Add patterns**: Edit `gap_utils.py`
2. **Test patterns**: Run on diverse queries
3. **Document patterns**: Add to pattern documentation
4. **Submit PR**: Include test results

### Bug Reports

Please include:
- Operating system
- Python version
- Error message/traceback
- Steps to reproduce
- Which mode (dev/production)
- Which engines enabled

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

### Data Sources
- Semantic Scholar
- Google Scholar
- arXiv
- PubMed/NCBI
- Crossref
- OpenAlex
- CORE
- SCOPUS/Elsevier
- Springer Nature
- And 9 more scholarly databases

### Technologies
- Streamlit (UI framework)
- Python (core language)
- Matplotlib (visualizations)
- Concurrent.futures (parallel processing)

---

## 🗺️ Roadmap

### Planned Features

- [ ] Additional scholarly databases (IEEE, ERIC, …)
- [ ] Machine learning-based relevance scoring
- [ ] Collaborative filtering recommendations
- [ ] Export to Zotero/Mendeley
- [ ] API for programmatic access
- [ ] Advanced visualization dashboards
- [ ] Citation network analysis
- [ ] Author disambiguation
- [ ] Institution tracking

---

## 📊 Statistics

- **18** Scholarly Databases
- **500M+** Total Papers Available
- **13** Free Engines (No Keys Required)
- **5** Premium Engines (Optional)
- **2** Operating Modes (Dev + Production)
- **Multiple** Export Formats (CSV, JSON, BibTeX)

---

**Built with ❤️ for the academic research community**

**Version:** 2.0 (Smart Key Detection)  
**Last Updated:** January 2026  
**Status:** Production Ready
