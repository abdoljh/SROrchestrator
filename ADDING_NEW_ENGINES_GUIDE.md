# Guide: Adding New Search Engines to SROrch

This guide explains how to add new search engines to SROrch. All placeholders are clearly marked with `📌 PLACEHOLDER` comments in the code.

---

## 🎯 Quick Overview

To add a new search engine, you need to:
1. Create a utility module (e.g., `new_engine_utils.py`)
2. Update `master_orchestrator.py` (3 locations)
3. Update `streamlit_app.py` (5 locations)

Total time: ~30 minutes for a simple engine

---

## 📁 Step 1: Create the Utility Module

Create a new file: `new_engine_utils.py`

### Template Structure:

```python
"""
Utility module for [Engine Name] API integration
API Documentation: [URL to API docs]
"""

import requests
import csv
from datetime import datetime

def format_authors(author_data):
    """
    Convert author data to IEEE format: 'I. Surname'
    
    Args:
        author_data: Author data from API (format varies by engine)
    
    Returns:
        str: Formatted author string (e.g., "J. Smith et al.")
    """
    # Implement author formatting specific to this engine
    pass

def fetch_and_process_new_engine(api_key, query, max_limit=25):
    """
    Fetch and process papers from [Engine Name]
    
    Args:
        api_key: API key for authentication (None if free engine)
        query: Search query string
        max_limit: Maximum number of results to return
    
    Returns:
        list: List of paper dictionaries with standardized fields
    """
    
    # 1. Validate inputs
    if not query:
        print("[NewEngine] Error: Empty query")
        return []
    
    # 2. Set up API request
    base_url = "https://api.newengine.com/search"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    params = {
        "query": query,
        "limit": max_limit,
        # Add other parameters as needed
    }
    
    # 3. Make API request
    try:
        response = requests.get(base_url, headers=headers, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"[NewEngine] API request failed: {e}")
        return []
    
    # 4. Process results
    processed = []
    results = data.get('results', [])  # Adjust based on API response structure
    
    for item in results:
        # Extract and standardize fields
        paper = {
            'ieee_authors': format_authors(item.get('authors')),
            'title': item.get('title', 'No title'),
            'venue': item.get('journal') or item.get('conference', 'Unknown'),
            'year': item.get('year', 'n.d.'),
            'citations': int(item.get('citation_count', 0)),
            'doi': item.get('doi', 'N/A'),
            'url': item.get('url', 'N/A'),
            'abstract': item.get('abstract', ''),  # Optional: may be fetched later
        }
        processed.append(paper)
    
    print(f"[NewEngine] Retrieved {len(processed)} papers")
    return processed

# Optional: Add helper functions as needed
```

### Key Requirements for the Utility Module:

**Required Fields in Output:**
- `ieee_authors` - String in format "I. Surname et al."
- `title` - Paper title
- `venue` - Journal/conference name
- `year` - Publication year (string or int)
- `citations` - Citation count (int)
- `doi` - DOI identifier or 'N/A'
- `url` - Link to paper or 'N/A'

**Optional Fields:**
- `abstract` - Paper abstract (can be empty)
- `keywords` - Comma-separated keywords
- `tldr` - AI-generated summary

---

## 🔧 Step 2: Update master_orchestrator.py

### Location 1: Import Section (Top of File)

```python
# Current imports
import s2_utils
import arxiv_utils
# ... existing imports ...

# 📌 ADD NEW IMPORT HERE
import new_engine_utils  # Your new module
```

### Location 2: __init__ Method - API Keys

Find the `📌 PLACEHOLDER: Add new API keys here` comment:

```python
self.api_keys = {
    's2': os.getenv('S2_API_KEY'),
    # ... existing keys ...
    
    # 📌 ADD NEW KEY HERE
    'new_engine': os.getenv('NEW_ENGINE_API_KEY'),  # For premium engines
}
```

**Note:** Skip this for free engines (no API key needed)

### Location 3: run_search Method - Task Submission

Find the `📌 PLACEHOLDER: Add new premium engines here` or `📌 PLACEHOLDER: Add new free engines here` section:

**For Premium Engines (require API key):**

```python
# In the PREMIUM ENGINES section
if is_valid_key(self.api_keys.get('new_engine')):
    tasks[executor.submit(
        new_engine_utils.fetch_and_process_new_engine,
        self.api_keys['new_engine'],
        query,
        max_limit=limit_per_engine
    )] = "New Engine Name"
    print(f"  ✓ New Engine Name enabled (API key provided)")
else:
    print(f"  ✗ New Engine Name skipped (no valid API key)")
    self.session_metadata['failed_engines'].append("New Engine Name (no API key)")
```

**For Free Engines (no API key):**

```python
# In the FREE ENGINES section
tasks[executor.submit(
    new_engine_utils.fetch_and_process_new_engine,
    query,
    max_limit=limit_per_engine
)] = "New Engine Name"
print(f"  ✓ New Engine Name enabled (free)")
```

---

## 💻 Step 3: Update streamlit_app.py

### Location 1: initialize_session_state Function

Find `📌 PLACEHOLDER: Add new engine keys here`:

```python
# Only for premium engines
if 'user_new_engine_key' not in st.session_state:
    st.session_state['user_new_engine_key'] = ''
```

### Location 2: load_api_keys Function

Find `📌 PLACEHOLDER: Add new engine keys here`:

```python
return {
    # ... existing keys ...
    'new_engine': st.session_state.get('user_new_engine_key', '').strip(),
}
```

### Location 3: render_api_key_input_section Function

Find `📌 PLACEHOLDER: Add new premium engine inputs here`:

```python
new_engine_key = st.text_input(
    "New Engine API Key",
    value="",
    type="password",
    help="Get key at: https://newengine.com/api",
    key="new_engine_input_widget",
    placeholder="Enter your New Engine API key"
)
```

Then in the "Apply Keys" button section:

```python
if st.button("✅ Apply Keys..."):
    # ... existing updates ...
    st.session_state['user_new_engine_key'] = new_engine_key.strip()
```

And in the "Show active keys" section:

```python
if st.session_state.get('user_new_engine_key'):
    active_keys.append("New Engine Name")
```

### Location 4: check_api_keys Function

Find `📌 PLACEHOLDER: Add validation for new engines`:

```python
status['new_engine'] = "✅" if api_keys.get('new_engine') and len(api_keys.get('new_engine', '')) > 5 else "❌"
```

### Location 5: get_available_engines Function

Find `📌 PLACEHOLDER: Add checks for new premium engines`:

```python
if key_status.get('new_engine') == "✅":
    available.append("New Engine Name")
```

Or for free engines, find `📌 PLACEHOLDER: Add new free engines here`:

```python
available.extend(["New Engine Name"])  # Add to free engines list
```

### Location 6: Engine Status Display

Find `📌 PLACEHOLDER: Add new premium engines here`:

```python
engine_display = {
    # ... existing engines ...
    "New Engine Name": key_status.get('new_engine', '❌'),
}
```

**Important:** Update the total count in the info message:

```python
st.info(f"**Active Engines:** {len(available_engines)}/9")  # Update from /8 to /9
```

---

## ✅ Testing Your New Engine

### 1. Test the Utility Module Independently

```python
# test_new_engine.py
from new_engine_utils import fetch_and_process_new_engine

# Test with a simple query
results = fetch_and_process_new_engine(
    api_key="your_test_key",
    query="machine learning",
    max_limit=5
)

print(f"Retrieved {len(results)} papers")
for paper in results[:2]:
    print(f"Title: {paper['title']}")
    print(f"Authors: {paper['ieee_authors']}")
    print(f"Year: {paper['year']}")
    print("---")
```

### 2. Test Integration

1. Run the Streamlit app
2. Enter your API key (if premium engine)
3. Run a test search
4. Check:
   - Engine appears in "Available Engines" list
   - Engine shows in session report
   - Papers from this engine appear in results
   - Deduplication works correctly

### 3. Verify Session Report

```json
{
  "successful_engines": [
    "New Engine Name",  // ✅ Should appear here
    // ... other engines
  ],
  "failed_engines": []  // Should be empty if key is valid
}
```

---

## 🎨 Example: Adding IEEE Xplore

Here's a complete example of adding IEEE Xplore:

### 1. Create `ieee_utils.py`:

```python
import requests

def format_ieee_authors(authors):
    """Format IEEE API authors to IEEE citation format"""
    if not authors:
        return "Unknown Author"
    
    author_list = [a.get('full_name', '') for a in authors if a.get('full_name')]
    
    if len(author_list) >= 3:
        return f"{author_list[0]} et al."
    elif len(author_list) == 2:
        return f"{author_list[0]} and {author_list[1]}"
    return author_list[0] if author_list else "Unknown Author"

def fetch_and_process_ieee(api_key, query, max_limit=25):
    """Fetch papers from IEEE Xplore"""
    
    if not api_key or len(api_key) < 10:
        print("[IEEE] Invalid API key")
        return []
    
    base_url = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
    
    params = {
        'apikey': api_key,
        'querytext': query,
        'max_records': max_limit,
        'start_record': 1
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"[IEEE] Error: {e}")
        return []
    
    processed = []
    articles = data.get('articles', [])
    
    for article in articles:
        paper = {
            'ieee_authors': format_ieee_authors(article.get('authors', {}).get('authors', [])),
            'title': article.get('title', 'No title'),
            'venue': article.get('publication_title', 'IEEE'),
            'year': str(article.get('publication_year', 'n.d.')),
            'citations': int(article.get('citing_paper_count', 0)),
            'doi': article.get('doi', 'N/A'),
            'url': article.get('pdf_url') or article.get('html_url', 'N/A'),
            'abstract': article.get('abstract', '')
        }
        processed.append(paper)
    
    print(f"[IEEE] Retrieved {len(processed)} papers")
    return processed
```

### 2. Update master_orchestrator.py:

```python
# At top
import ieee_utils

# In __init__
self.api_keys = {
    # ... existing ...
    'ieee': os.getenv('IEEE_API_KEY'),
}

# In run_search
if is_valid_key(self.api_keys.get('ieee')):
    tasks[executor.submit(ieee_utils.fetch_and_process_ieee, self.api_keys['ieee'], query, max_limit=limit_per_engine)] = "IEEE Xplore"
    print(f"  ✓ IEEE Xplore enabled (API key provided)")
else:
    print(f"  ✗ IEEE Xplore skipped (no valid API key)")
    self.session_metadata['failed_engines'].append("IEEE Xplore (no API key)")
```

### 3. Update streamlit_app.py:

```python
# In initialize_session_state
if 'user_ieee_key' not in st.session_state:
    st.session_state['user_ieee_key'] = ''

# In load_api_keys
return {
    # ... existing ...
    'ieee': st.session_state.get('user_ieee_key', '').strip(),
}

# In render_api_key_input_section
ieee_key = st.text_input(
    "IEEE Xplore API Key",
    value="",
    type="password",
    help="Get key at: https://developer.ieee.org/",
    key="ieee_input_widget",
    placeholder="Enter your IEEE API key"
)

# In Apply Keys button
st.session_state['user_ieee_key'] = ieee_key.strip()

# In active keys display
if st.session_state.get('user_ieee_key'):
    active_keys.append("IEEE Xplore")

# In check_api_keys
status['ieee'] = "✅" if api_keys.get('ieee') and len(api_keys.get('ieee', '')) > 5 else "❌"

# In get_available_engines
if key_status.get('ieee') == "✅":
    available.append("IEEE Xplore")

# In engine_display
engine_display = {
    # ... existing ...
    "IEEE Xplore": key_status.get('ieee', '❌'),
}

# Update count
st.info(f"**Active Engines:** {len(available_engines)}/9")
```

Done! IEEE Xplore is now integrated.

---

## 📋 Checklist for Adding New Engine

- [ ] Created utility module (`new_engine_utils.py`)
- [ ] Tested utility module independently
- [ ] Updated `master_orchestrator.py`:
  - [ ] Added import
  - [ ] Added API key to `__init__` (if premium)
  - [ ] Added task submission in `run_search`
- [ ] Updated `streamlit_app.py`:
  - [ ] Added session state initialization
  - [ ] Added key to `load_api_keys`
  - [ ] Added input field in `render_api_key_input_section`
  - [ ] Updated Apply Keys button handler
  - [ ] Updated active keys display
  - [ ] Added validation in `check_api_keys`
  - [ ] Added to `get_available_engines`
  - [ ] Added to engine status display
  - [ ] Updated total engine count
- [ ] Tested integration:
  - [ ] Engine appears in UI
  - [ ] API key validation works
  - [ ] Search returns results
  - [ ] Session report is accurate
  - [ ] Deduplication works

---

## 🐛 Common Issues

**Issue:** Engine always shows as "failed (no API key)"
- **Fix:** Check that API key is added to all required locations in streamlit_app.py

**Issue:** Papers not appearing in results
- **Fix:** Verify utility module returns correct field names (especially `ieee_authors`, `title`, `year`)

**Issue:** Duplicate papers not being merged
- **Fix:** Ensure DOI is correctly extracted and formatted

**Issue:** Engine count not updating
- **Fix:** Update the total count in `st.info(f"**Active Engines:** {len(available_engines)}/X")`

---

## 💡 Tips

1. **Start Simple:** Begin with a basic implementation, then add features
2. **Follow Existing Patterns:** Look at `arxiv_utils.py` or `s2_utils.py` for examples
3. **Test Incrementally:** Test each step before moving to the next
4. **Handle Errors Gracefully:** Use try-except blocks and return empty lists on error
5. **Document API Limits:** Note rate limits in comments
6. **Standardize Output:** Ensure all engines return the same field structure

---

## 📚 Resources

- **Existing Utility Modules:** Check `s2_utils.py`, `arxiv_utils.py`, etc. for reference
- **API Documentation:** Always refer to the official API docs for each engine
- **Testing:** Use small queries during development to avoid hitting rate limits

---

Ready to add a new engine? Follow this guide step-by-step and you'll be done in 30 minutes! 🚀
