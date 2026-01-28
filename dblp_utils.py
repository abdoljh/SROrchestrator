import requests
import time

def format_dblp_authors(author_data):
    """Converts DBLP author list into IEEE 'I. Surname'."""
    if not author_data:
        return "Unknown Author"
    
    # DBLP can return a single dict or a list of dicts
    authors_list = author_data if isinstance(author_data, list) else [author_data]
    
    formatted = []
    for a in authors_list:
        name = a.get('text', '') if isinstance(a, dict) else str(a)
        if not name: continue
        
        # DBLP names are typically "First Last"
        parts = name.split(' ')
        if len(parts) > 1:
            surname = parts[-1]
            # Handle possible middle names by taking the first character of the first part
            initial = parts[0][0]
            formatted.append(f"{initial}. {surname}")
        else:
            formatted.append(name)
            
    if len(formatted) >= 3:
        return f"{formatted[0]} et al."
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    return formatted[0] if formatted else "Unknown Author"

def fetch_and_process_dblp(query, max_limit=10):
    """
    Searches DBLP with strict JSON endpoint and bot-prevention headers.
    The correct API endpoint is /search/publ/api
    """
    # Fix: Ensure 'publ' (publications) is used in the URL
    base_url = "https://dblp.org/search/publ/api"
    
    # Fix: Use a browser-like User-Agent and explicitly request JSON
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json"
    }
    
    params = {
        "q": query,
        "format": "json",
        "h": max_limit
    }

    try:
        # DBLP requires a gap between requests to avoid 429 errors
        time.sleep(1.0)
        
        response = requests.get(base_url, params=params, headers=headers, timeout=20)
        
        # Verify if the response is actually JSON before parsing
        content_type = response.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            print(f"[Error] DBLP failed to return JSON. Received: {content_type}")
            return []

        data = response.json()
        hits = data.get('result', {}).get('hits', {}).get('hit', [])
        
        processed = []
        for hit in hits:
            info = hit.get('info', {})
            author_data = info.get('authors', {}).get('author', [])
            
            processed.append({
                'ieee_authors': format_dblp_authors(author_data),
                'title': info.get('title', 'No Title'),
                'venue': info.get('venue', 'DBLP Indexed'),
                'year': info.get('year', 'n.d.'),
                'citations': 0, 
                'doi': info.get('doi', 'N/A'),
                'url': info.get('ee', '')
            })
        return processed

    except Exception as e:
        print(f"[Error] DBLP integration failure: {e}")
        return []
