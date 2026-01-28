import requests
import os

def format_acm_authors(author_list):
    """
    Converts ACM/CrossRef author list into IEEE 'I. Surname'.
    Input: [{'given': 'John', 'family': 'Smith'}, ...]
    """
    if not author_list:
        return "Unknown Author"
    
    formatted = []
    for auth in author_list:
        family = auth.get('family', '')
        given = auth.get('given', '')
        
        if family and given:
            # Standard IEEE: Initial of first name + Surname
            initial = given[0]
            formatted.append(f"{initial}. {family}")
        elif family:
            formatted.append(family)
            
    if len(formatted) >= 3:
        return f"{formatted[0]} et al."
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    return formatted[0] if formatted else "Unknown Author"

def fetch_and_process_acm(query, max_limit=10):
    """
    Searches ACM Digital Library using CrossRef's prefix filter (10.1145).
    This targets the Association for Computing Machinery specifically.
    """
    base_url = "https://api.crossref.org/works"
    
    params = {
        "query": query,
        # 10.1145 is the unique DOI owner prefix for ACM
        "filter": "prefix:10.1145",
        "rows": max_limit,
        "select": "DOI,title,author,container-title,published-print,URL"
    }

    # CrossRef 'Polite' User-Agent is recommended
    headers = {
        "User-Agent": "ResearchScript/1.0 (mailto:your-email@example.com)"
    }

    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=20)
        if response.status_code != 200:
            print(f"[Error] ACM (CrossRef) API returned status: {response.status_code}")
            return []
            
        data = response.json()
        entries = data.get('message', {}).get('items', [])
        processed = []

        for entry in entries:
            # Extract Year from publication date parts
            pub_parts = entry.get('published-print', {}).get('date-parts', [[None]])[0]
            year = str(pub_parts[0]) if pub_parts[0] else 'n.d.'

            # Title handling (standardize list to string)
            titles = entry.get('title', ['No Title'])
            title = titles[0] if titles else 'No Title'
            
            # Venue (Journal or Conference proceedings name)
            venues = entry.get('container-title', ['ACM Digital Library'])
            venue = venues[0] if venues else 'ACM Digital Library'

            processed.append({
                'ieee_authors': format_acm_authors(entry.get('author')),
                'title': title,
                'venue': venue,
                'year': year,
                'citations': 0, # Live citation counts require CrossRef CitedBy participation
                'doi': entry.get('DOI', 'N/A'),
                'url': entry.get('URL', '')
            })
            
        return processed
    except Exception as e:
        print(f"[Error] ACM integration failure: {e}")
        return []
