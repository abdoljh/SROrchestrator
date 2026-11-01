import time
import csv
import re
import requests
from datetime import datetime
from crossref_common import CROSSREF_SELECT, parse_crossref_item
from query_utils import safe_query_slug, clean_abstract, to_plain_keywords

def format_crossref_authors(authors_list):
    """Formats Crossref author list into IEEE 'I. Surname'."""
    if not authors_list:
        return "Unknown Author"
    
    formatted = []
    for auth in authors_list:
        family = auth.get('family', '')
        given = auth.get('given', '')
        if family and given:
            # IEEE Style: Initials. Surname
            formatted.append(f"{given[0]}. {family}")
        else:
            formatted.append(family if family else "Unknown")
            
    if len(formatted) >= 3:
        return f"{formatted[0]} et al."
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    return formatted[0]

def fetch_and_process_doi(query, max_limit=10, save_csv=True, email="your@email.com"):
    """
    Searches Crossref (the DOI registry) for papers matching the query.
    Returns a list of dictionaries in the unified IEEE schema.
    """
    base_url = "https://api.crossref.org/works"
    # Using 'Polite' API etiquette by including an email in params
    params = {
        "query": to_plain_keywords(query),
        "rows": min(max_limit * 2, 100),  # oversample: non-papers are dropped
        "mailto": email,
        "select": CROSSREF_SELECT
    }

    try:
        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        items = data.get("message", {}).get("items", [])
    except Exception as e:
        print(f"[Error] Crossref API failure: {e}")
        return []

    processed_data = []
    
    _seen = set()
    for item in items:
        paper = parse_crossref_item(item, 'Unknown Venue')
        if paper is None:  # peer reviews, supplementary files, ...
            continue
        key = paper['doi'].lower()
        if key in _seen:
            continue
        _seen.add(key)
        processed_data.append(paper)
        if len(processed_data) >= max_limit:
            break

    # Sort alphabetical by author
    processed_data.sort(key=lambda x: x['sort_name'].lower())

    if save_csv and processed_data:
        clean_q = safe_query_slug(query)
        filename = f"doi_bulk_{clean_q}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['ieee_authors', 'title', 'venue', 'year', 'citations', 'doi', 'url', 'abstract'], extrasaction='ignore')
            writer.writeheader()
            for row in processed_data:
                writer.writerow({k: v for k, v in row.items() if k != 'sort_name'})
        print(f"[System] Success: {len(processed_data)} DOI records saved to {filename}")

    time.sleep(1) # Strategic Delay
    return processed_data
