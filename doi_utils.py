import time
import csv
import re
import requests
from datetime import datetime

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
        "query": query,
        "rows": max_limit,
        "mailto": email,
        "select": "DOI,title,author,container-title,published-print,published-online,URL,is-referenced-by-count"
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
    
    for item in items:
        # 1. Author Logic & Sort Key
        authors_data = item.get("author", [])
        display_authors = format_crossref_authors(authors_data)
        sort_key = authors_data[0].get('family', 'Unknown') if authors_data else "Unknown"

        # 2. Date Extraction (Crossref has complex nested dates)
        year = "n.d."
        pub_date = item.get("published-print") or item.get("published-online")
        if pub_date and "date-parts" in pub_date:
            year = pub_date["date-parts"][0][0]

        # 3. Venue (Journal/Conference name)
        venue_list = item.get("container-title", [])
        venue = venue_list[0] if venue_list else "Unknown Venue"

        processed_data.append({
            'sort_name': sort_key,
            'ieee_authors': display_authors,
            'title': item.get('title', ['Untitled'])[0],
            'venue': venue,
            'year': year,
            'citations': item.get('is-referenced-by-count', 0),
            'doi': item.get('DOI', 'N/A'),
            'url': item.get('URL', f"https://doi.org/{item.get('DOI')}")
        })

    # Sort alphabetical by author
    processed_data.sort(key=lambda x: x['sort_name'].lower())

    if save_csv and processed_data:
        clean_q = re.sub(r'[^\w\s-]', '', query).strip().replace(' ', '_')
        filename = f"doi_bulk_{clean_q}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['ieee_authors', 'title', 'venue', 'year', 'citations', 'doi', 'url'])
            writer.writeheader()
            for row in processed_data:
                writer.writerow({k: v for k, v in row.items() if k != 'sort_name'})
        print(f"[System] Success: {len(processed_data)} DOI records saved to {filename}")

    time.sleep(1) # Strategic Delay
    return processed_data
