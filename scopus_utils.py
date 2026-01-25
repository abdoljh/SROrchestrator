# scopus_utils.py
import requests
import csv
import re
import os
from datetime import datetime

def format_scopus_authors(author_str):
    if not author_str:
        return "Unknown Author", "Unknown"
    
    authors = [a.strip() for a in re.split(r'[;,]', author_str) if a.strip()]
    formatted = []
    sort_key = "Unknown"
    
    for i, auth in enumerate(authors):
        parts = auth.split(',')
        if len(parts) > 1:
            surname = parts[0].strip()
            initial = parts[1].strip()[0]
            if i == 0: sort_key = surname
            formatted.append(f"{initial}. {surname}")
        else:
            if i == 0: sort_key = auth
            formatted.append(auth)
            
    if len(formatted) >= 3:
        display = f"{formatted[0]} et al."
    elif len(formatted) == 2:
        display = f"{formatted[0]} and {formatted[1]}"
    else:
        display = formatted[0] if formatted else "Unknown Author"
        
    return display, sort_key

def fetch_and_process_scopus(api_key, query, max_limit=25, save_csv=True):
    # CRITICAL: Use passed key first (Streamlit compatibility)
    scopus_key = api_key or os.getenv('SCOPUS_API_KEY')
    inst_token = os.getenv('SCOPUS_INST_TOKEN')
    
    if not scopus_key:
        print("[Error] Scopus API Key missing.")
        return []

    base_url = "https://api.elsevier.com/content/search/scopus"
    headers = {"X-ELS-APIKey": scopus_key, "Accept": "application/json"}
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
            return []
            
        data = response.json()
        entries = data.get('search-results', {}).get('entry', [])
        processed_data = []

        for entry in entries:
            if 'error' in entry: continue
            cover_date = entry.get('prism:coverDate', '')
            year = cover_date.split('-')[0] if cover_date else 'n.d.'
            display_authors, sort_key = format_scopus_authors(entry.get('dc:creator'))

            processed_data.append({
                'sort_name': sort_key,
                'ieee_authors': display_authors,
                'title': entry.get('dc:title', 'Untitled Document'),
                'venue': entry.get('prism:publicationName', 'Scopus'),
                'year': year,
                'citations': int(entry.get('citedby-count', 0)),
                'doi': entry.get('prism:doi', 'N/A'),
                'url': entry.get('link', [{}, {}, {}])[2].get('@href', '')
            })
            
        processed_data.sort(key=lambda x: x['sort_name'].lower())

        if save_csv and processed_data:
            clean_q = re.sub(r'[^\w\s-]', '', query).strip().replace(' ', '_')
            filename = f"scopus_{clean_q}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['ieee_authors', 'title', 'venue', 'year', 'citations', 'doi', 'url'])
                writer.writeheader()
                for row in processed_data:
                    writer.writerow({k: v for k, v in row.items() if k != 'sort_name'})
            print(f"[System] Scopus results saved to {filename}")

        return processed_data
    except Exception as e:
        print(f"[Error] Scopus failure: {e}")
        return []
