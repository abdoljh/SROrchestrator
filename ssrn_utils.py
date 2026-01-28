# ssrn_utils.py
import requests
import os

def format_ssrn_authors(author_list):
    """
    Converts CrossRef/SSRN author list into IEEE 'I. Surname'.
    CrossRef returns authors as a list of dicts: [{'given': 'John', 'family': 'Smith'}, ...]
    """
    if not author_list:
        return "Unknown Author"
    
    formatted = []
    for auth in author_list:
        family = auth.get('family', '')
        given = auth.get('given', '')
        
        if family and given:
            initial = given[0]
            formatted.append(f"{initial}. {family}")
        elif family:
            formatted.append(family)
            
    if len(formatted) >= 3:
        return f"{formatted[0]} et al."
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    return formatted[0] if formatted else "Unknown Author"

def fetch_and_process_ssrn(query, max_limit=20):
    """
    Searches SSRN papers via the CrossRef API.
    Filters by the SSRN DOI prefix (10.2139).
    """
    # CrossRef API endpoint
    base_url = "https://api.crossref.org/works"
    
    # Parameters for the search
    params = {
        "query": query,
        # Filter for SSRN prefix (10.2139) to ensure results are from SSRN
        "filter": "prefix:10.2139",
        "rows": max_limit,
        "select": "DOI,title,author,container-title,published-print,published-online,URL"
    }

    # CrossRef recommends including an email for their "Polite" pool
    headers = {
        "User-Agent": "ResearchScript/1.0 (mailto:your-email@example.com)"
    }

    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=20)
        if response.status_code != 200:
            print(f"[Error] CrossRef API returned status: {response.status_code}")
            return []
            
        data = response.json()
        entries = data.get('message', {}).get('items', [])
        processed = []

        for entry in entries:
            # Extract Year from published-print or published-online
            pub_parts = entry.get('published-print', entry.get('published-online', {}))
            year_list = pub_parts.get('date-parts', [[None]])[0]
            year = str(year_list[0]) if year_list[0] else 'n.d.'

            # Title is returned as a list
            titles = entry.get('title', ['No Title'])
            title = titles[0] if titles else 'No Title'

            processed.append({
                'ieee_authors': format_ssrn_authors(entry.get('author')),
                'title': title,
                'venue': 'SSRN Electronic Journal',
                'year': year,
                'citations': 0, # CrossRef does not provide citation counts in this endpoint
                'doi': entry.get('DOI', 'N/A'),
                'url': entry.get('URL', '')
            })
            
        return processed
    except Exception as e:
        print(f"[Error] SSRN (CrossRef) integration failure: {e}")
        return []
