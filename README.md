# SROrch Streamlit Interface

A web-based interface for the Scholarly Research Orchestrator (SROrch) - a powerful multi-engine academic literature search and analysis tool.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project files
cd srorch-streamlit

# Install dependencies
pip install -r requirements_streamlit.txt
```

### 2. Configure API Keys

Create a `.streamlit/secrets.toml` file in your project directory:

```bash
mkdir -p .streamlit
cp .streamlit/secrets.toml.template .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` with your actual API keys:

```toml
S2_API_KEY = "your_semantic_scholar_api_key"
SERP_API_KEY = "your_serpapi_key"
CORE_API_KEY = "your_core_api_key"
USER_EMAIL = "your.email@example.com"
```

### 3. Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 🔑 API Keys Setup

### Required API Keys

1. **Semantic Scholar API Key** (Highly Recommended)
   - Visit: https://www.semanticscholar.org/product/api
   - Sign up for a free account
   - Generate your API key
   - Provides: Abstract fetching, enhanced metadata

2. **User Email** (Required)
   - Used for PubMed API requests (NCBI requirement)
   - No signup needed, just provide a valid email

### Optional API Keys (For Enhanced Coverage)

3. **SERP API Key** (Optional but Recommended)
   - Visit: https://serpapi.com/
   - Free tier: 100 searches/month
   - Paid plans available for higher usage
   - Provides: Google Scholar access

4. **CORE API Key** (Optional)
   - Visit: https://core.ac.uk/services/api
   - Free tier available
   - Provides: Open access research papers

### Free Engines (No API Key Required)

The following engines work without API keys:
- arXiv
- PubMed (requires email only)
- Crossref/DOI
- OpenAlex

## 📁 Project Structure

```
srorch-streamlit/
├── app.py                          # Streamlit web interface
├── master_orchestrator.py          # Core orchestrator logic
├── requirements_streamlit.txt      # Python dependencies
├── .streamlit/
│   ├── secrets.toml               # API keys (create this!)
│   └── secrets.toml.template      # Template for secrets
├── s2_utils.py                    # Semantic Scholar integration
├── arxiv_utils.py                 # arXiv integration
├── pubmed_utils.py                # PubMed integration
├── scholar_utils.py               # Google Scholar integration
├── doi_utils.py                   # DOI/Crossref integration
├── openalex_utils.py              # OpenAlex integration
└── core_utils.py                  # CORE integration
```

## 🎯 Features

### Multi-Engine Search
- **7 Academic Databases** integrated
- Parallel search execution
- Automatic result aggregation
- Deduplication and consensus detection

### Intelligent Ranking
- **Relevance Scoring** based on:
  - Citation counts
  - Multi-source consensus
  - Publication recency (optional boost)
- Configurable scoring weights

### Deep Analysis
- Abstract fetching for top papers
- AI-generated TL;DR summaries (when available)
- Keyword extraction
- Publication trend visualization

### Export Options
- **CSV** - Spreadsheet format for data analysis
- **JSON** - Structured data for integration
- **BibTeX** - Reference manager compatible
- **ZIP** - Complete archive of all outputs

### Analytics Dashboard
- Publication timeline chart
- Source consensus distribution
- Citation statistics
- Research metrics summary

## ⚙️ Configuration Options

### Search Parameters
- **Results per engine** (5-50): Number of papers from each database
- **Deep Look Limit** (1-20): Papers to fetch full abstracts for

### Scoring Settings
- **Citation Weight** (0.1-5.0): Importance of citation counts
- **Source Weight** (10-500): Value of multi-database presence
- **Consensus Threshold** (2-7): Sources needed for alert

### Recency Boost
- **Enable/Disable**: Prefer recent publications
- **Time Window** (1-10 years): Define "recent"
- **Multiplier** (1.0-2.0): Boost strength for recent papers

### Output Options
- **Consensus Alerts**: Real-time high-impact paper notifications
- **Visualizations**: Generate charts and graphs
- **Export Formats**: Choose CSV, JSON, BibTeX (or all)

## 📊 Using the Interface

### Search Tab
1. Enter your research query
2. Adjust configuration in sidebar (optional)
3. Click "Start Search"
4. Monitor progress bar
5. Wait for completion

### Results Tab
1. View summary metrics
2. Browse top papers with expandable details
3. Review analytics visualizations
4. Download reports in various formats
5. Check session details

### About Tab
- Feature documentation
- Supported databases
- Use case examples
- System information

## 📝 Output Files Explained

### MASTER_REPORT_FINAL.csv
Complete results table with columns:
- Relevance score
- Source count
- Authors (IEEE format)
- Title, venue, year
- Citations, DOI, URL
- Abstract, keywords, TL;DR
- Recency boost flag

### EXECUTIVE_SUMMARY.txt
Curated summary featuring:
- Top N papers (configurable)
- Full abstracts
- Complete metadata
- Direct links

### research_data.json
Structured JSON with:
- Search metadata
- All paper details
- Session information
- Execution statistics

### references.bib
BibTeX format citations:
- Auto-generated citation keys
- Complete bibliographic data
- Compatible with Zotero, Mendeley, etc.

### research_analytics.png
Visual dashboard showing:
- Publication timeline
- Source consensus distribution

### SESSION_REPORT.txt
Search session details:
- Query and timestamps
- Execution duration
- Engine success/failure status
- Configuration used

## 🔧 Troubleshooting

### "Error loading secrets"
- Ensure `.streamlit/secrets.toml` exists
- Check file permissions
- Verify TOML syntax

### "Search engine failed"
- Check API key validity
- Verify internet connection
- Review rate limits (especially for SERP API)
- Some engines may be temporarily unavailable

### "No abstracts found"
- Semantic Scholar API key may be missing
- Papers may not have abstracts in database
- Rate limiting may be active

### Import errors
- Run: `pip install -r requirements_streamlit.txt`
- Ensure all utility files (s2_utils.py, etc.) are present
- Check Python version (3.8+ recommended)

## 🌐 Deployment Options

### Local Deployment
```bash
streamlit run app.py
```

### Streamlit Cloud
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Add secrets via Cloud dashboard
4. Deploy!

### Docker Deployment
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements_streamlit.txt .
RUN pip install -r requirements_streamlit.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 📚 Research Use Cases

### Literature Review
- Comprehensive coverage across databases
- Identify seminal papers via consensus
- Track publication trends

### Gap Analysis
- Find under-researched areas
- Compare coverage across venues
- Identify emerging topics

### Citation Analysis
- Find highly-cited papers
- Track citation patterns
- Discover influential authors

### Multi-Database Validation
- Verify paper presence across sources
- Cross-check metadata
- Assess research impact

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional search engine integrations
- Enhanced visualization options
- Advanced filtering capabilities
- Export format extensions

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

This tool integrates with:
- Semantic Scholar
- arXiv
- PubMed/NCBI
- Google Scholar (via SerpAPI)
- Crossref
- OpenAlex
- CORE

## 📞 Support

For issues or questions:
1. Check this README
2. Review error messages in Streamlit interface
3. Verify API key configuration
4. Check individual engine status

---

**Happy Researching! 🔬📚**
