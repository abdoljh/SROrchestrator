# plos_utils.py
import requests
import re
import os

def format_plos_authors(author_list):
    """
    Converts PLOS author list (usually a list of strings) into IEEE 'I. Surname'.
    PLOS typically returns authors as ['Surname, Firstname', ...].
    """
    if not author_list or not isinstance(author_list, list):
        return "Unknown Author"
    
    formatted = []
    for auth in author_list:
        # PLOS format is usually "Surname, Firstname"
        parts = auth.split(',')
        if len(parts) > 1:
            surname = parts[0].strip()
            # Get the first letter of the first name
            first_name = parts[1].strip()
            initial = first_name[0] if first_name else ""
            formatted.append(f"{initial}. {surname}")
        else:
            formatted.append(auth)
            
    if len(formatted) >= 3:
        return f"{formatted[0]} et al."
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    return formatted[0] if formatted else "Unknown Author"

def fetch_and_process_plos(query, max_limit=20):
    """
    Searches PLOS via their Solr-based Search API.
    API Documentation: http://api.plos.org/
    """
    # PLOS API base URL
    base_url = "http://api.plos.org/search"
    
    # Parameters for the search
    # 'fl' defines the field list we want returned
    params = {
        "q": f"title:\"{query}\" OR abstract:\"{query}\"",
        "fl": "author_display,title,journal,publication_date,id,counter_total_all",
        "wt": "json",       # Response format
        "rows": max_limit   # Number of results
    }

    try:
        response = requests.get(base_url, params=params, timeout=20)
        if response.status_code != 200:
            print(f"[Error] PLOS API returned status: {response.status_code}")
            return []
            
        data = response.json()
        # PLOS results are inside response -> docs
        entries = data.get('response', {}).get('docs', [])
        processed = []

        for entry in entries:
            # Extract Year from publication_date (format: 2023-10-24T00:00:00Z)
            pub_date = entry.get('publication_date', '')
            year = pub_date.split('-')[0] if pub_date else 'n.d.'

            processed.append({
                'ieee_authors': format_plos_authors(entry.get('author_display')),
                'title': entry.get('title'),
                'venue': entry.get('journal', 'PLOS Indexed Journal'),
                'year': year,
                # PLOS uses 'counter_total_all' for total views/usage as a metric
                'citations': int(entry.get('counter_total_all', 0)),
                'doi': entry.get('id', 'N/A'), # 'id' in PLOS is the DOI
                'url': f"https://journals.plos.org/plosone/article?id={entry.get('id')}"
            })
            
        return processed
    except Exception as e:
        print(f"[Error] PLOS integration failure: {e}")
        return []

# Example Usage:
# results = fetch_and_process_plos("machine learning in healthcare", max_limit=5)
# for res in results:
#     print(f"{res['year']} - {res['title']}")
