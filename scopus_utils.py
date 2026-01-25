# scopus_utils.py
import requests
import csv
import re
import os
from datetime import datetime

def format_scopus_authors(author_str):
    """Converts Scopus author string 'Surname, I.' into IEEE 'I. Surname'."""
    if not author_str:
        return "Unknown Author"
    
    # Scopus usually returns authors separated by ';' or ','
    authors = [a.strip() for a in re.split(r'[;,]', author_str) if a.strip()]
    formatted = []
    
    for auth in authors:
        parts = auth.split(',')
        if len(parts) > 1:
            surname = parts[0].strip()
            initial = parts[1].strip()[0]
            formatted.append(f"{initial}. {surname}")
        else:
            formatted.append(auth)
            
    if len(formatted) >= 3:
        return f"{formatted[0]} et al."
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    return formatted[0] if formatted else "Unknown Author"

def fetch_and_process_scopus(api_key, query, max_limit=20, save_csv=True):
#def fetch_and_process_scopus0(query, max_limit=25):
    """Searches Scopus via Elsevier's API."""
    api_key = os.getenv('SCOPUS_API_KEY')
    inst_token = os.getenv('SCOPUS_INST_TOKEN') # Optional for some institutions
    
    if not api_key:
        print("[Error] SCOPUS_API_KEY not found in environment.")
        return []

    base_url = "https://api.elsevier.com/content/search/scopus"
    headers = {
        "X-ELS-APIKey": api_key,
        "Accept": "application/json"
    }
    if inst_token:
        headers["X-ELS-Insttoken"] = inst_token

    params = {
        "query": f"TITLE-ABS-KEY({query})",
        "count": max_limit,
        "view": "STANDARD"
    }

    try:
        response = requests.get(base_url, headers=headers, params=params, timeout=20)
        if response.status_code != 200:
            print(f"[Error] Scopus API returned status: {response.status_code}")
            return []
            
        data = response.json()
        entries = data.get('search-results', {}).get('entry', [])
        processed = []

        for entry in entries:
            # Handle potential error entries
            if 'error' in entry: continue

            # Extract Year
            cover_date = entry.get('prism:coverDate', '')
            year = cover_date.split('-')[0] if cover_date else 'n.d.'

            processed.append({
                'ieee_authors': format_scopus_authors(entry.get('dc:creator')),
                'title': entry.get('dc:title'),
                'venue': entry.get('prism:publicationName', 'Scopus Indexed Journal'),
                'year': year,
                'citations': int(entry.get('citedby-count', 0)),
                'doi': entry.get('prism:doi', 'N/A'),
                'url': entry.get('link', [{}])[2].get('@href', '') # Usually the scopus link
            })
            
        return processed
    except Exception as e:
        print(f"[Error] Scopus integration failure: {e}")
        return []
