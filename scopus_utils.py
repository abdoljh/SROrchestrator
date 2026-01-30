# scopus_utils.py
import requests
import csv
import re
import os
import time
from datetime import datetime

def format_scopus_authors(author_str):
    """Converts Scopus author string 'Surname, I.' into IEEE 'I. Surname'."""
    if not author_str:
        return "Unknown Author", "Unknown"
    
    # Scopus usually returns authors separated by ';' or ','
    authors = [a.strip() for a in re.split(r'[;,]', author_str) if a.strip()]
    formatted = []
    
    for auth in authors:
        parts = auth.split(',')
        if len(parts) > 1:
            surname = parts[0].strip()
            initial = parts[1].strip()[0] if parts[1].strip() else ""
            formatted.append(f"{initial}. {surname}")
        else:
            formatted.append(auth)
            
    # Get sort key (first author's full name for sorting)
    sort_key = authors[0] if authors else "Unknown"
    
    if len(formatted) >= 3:
        return f"{formatted[0]} et al.", sort_key
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}", sort_key
    return formatted[0] if formatted else "Unknown Author", sort_key

def fetch_and_process_scopus(api_key, query, max_limit=20, save_csv=True):
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
        processed_data = []
        seen_dois = set()

        for entry in entries:
            # Handle potential error entries
            if 'error' in entry: continue

            # Extract Year
            cover_date = entry.get('prism:coverDate', '')
            year = cover_date.split('-')[0] if cover_date else 'n.d.'
            
            # Get DOI for deduplication
            doi = entry.get('prism:doi', '')
            if doi and doi in seen_dois:
                continue
            if doi:
                seen_dois.add(doi)

            # Format authors and get sort key
            ieee_authors, sort_key = format_scopus_authors(entry.get('dc:creator'))

            processed_data.append({
                'sort_name': sort_key,
                'ieee_authors': ieee_authors,
                'title': entry.get('dc:title'),
                'venue': entry.get('prism:publicationName', 'Scopus Indexed Journal'),
                'year': year,
                'citations': int(entry.get('citedby-count', 0)),
                'doi': doi or 'N/A',
                'url': entry.get('link', [{}])[2].get('@href', '') # Usually the scopus link
            })
            
        # Sort by Author Name
        processed_data.sort(key=lambda x: x['sort_name'].lower())

        # Save to Unique CSV
        if save_csv and processed_data:
            clean_q = re.sub(r"[^\w\s-]", "", query).strip().replace(" ", "_")
            filename = f"scopus_{clean_q}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['ieee_authors', 'title', 'venue', 'year', 'citations', 'doi', 'url'])
                writer.writeheader()
                for row in processed_data:
                    # Filter out the helper sort_name key
                    writer.writerow({k: v for k, v in row.items() if k != 'sort_name'})
            print(f"[System] Scopus results ({len(processed_data)} papers) saved to {filename}")

        # Strategic delay for API respect
        time.sleep(1)

        return processed_data
    except Exception as e:
        print(f"[Error] Scopus integration failure: {e}")
        return []