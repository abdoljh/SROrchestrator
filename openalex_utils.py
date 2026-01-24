import time
import csv
import re
import requests
from datetime import datetime

def format_openalex_authors(authorships):
    """Formats OpenAlex authorship objects into IEEE 'I. Surname'."""
    if not authorships:
        return "Unknown Author"
    
    formatted = []
    for auth in authorships:
        display_name = auth.get('author', {}).get('display_name', '')
        parts = display_name.split()
        if len(parts) > 1:
            formatted.append(f"{parts[0][0]}. {' '.join(parts[1:])}")
        else:
            formatted.append(display_name if display_name else "Unknown")
            
    if len(formatted) >= 3:
        return f"{formatted[0]} et al."
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    return formatted[0]

def fetch_and_process_openalex(query, max_limit=20, save_csv=True, email="your@email.com"):
    """
    Searches OpenAlex for papers. No API key required, 
    but an email is recommended for the 'polite pool'.
    """
    # OpenAlex API endpoint
    base_url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per_page": max_limit,
        "mailto": email,
        # We select specific fields to keep the response fast
        "select": "id,title,publication_year,authorships,primary_location,cited_by_count,doi"
    }

    try:
        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
    except Exception as e:
        print(f"[Error] OpenAlex API failure: {e}")
        return []

    processed_data = []
    
    for work in results:
        # 1. Author Logic
        authorships = work.get("authorships", [])
        display_authors = format_openalex_authors(authorships)
        
        # Sort Key (Surname of first author)
        first_author_full = authorships[0].get('author', {}).get('display_name', 'Unknown') if authorships else "Unknown"
        sort_key = first_author_full.split()[-1] if ' ' in first_author_full else first_author_full

        # 2. Venue (Source)
        source = work.get("primary_location", {}).get("source", {})
        venue = source.get("display_name", "Unknown Venue") if source else "Unknown Venue"

        # 3. DOI & URL
        doi = work.get("doi", "N/A")
        # OpenAlex DOIs are full URLs, we clean them for the DOI column
        clean_doi = doi.replace("https://doi.org/", "") if doi else "N/A"
        
        # Use OpenAlex web UI link or the DOI link
        url = work.get("id", "") 

        processed_data.append({
            'sort_name': sort_key,
            'ieee_authors': display_authors,
            'title': work.get('title', 'Untitled Document'),
            'venue': venue,
            'year': work.get('publication_year', 'n.d.'),
            'citations': work.get('cited_by_count', 0),
            'doi': clean_doi,
            'url': url
        })

    # Sort alphabetically by surname
    processed_data.sort(key=lambda x: x['sort_name'].lower())

    if save_csv and processed_data:
        clean_q = re.sub(r'[^\w\s-]', '', query).strip().replace(' ', '_')
        filename = f"openalex_{clean_q}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['ieee_authors', 'title', 'venue', 'year', 'citations', 'doi', 'url'])
            writer.writeheader()
            for row in processed_data:
                writer.writerow({k: v for k, v in row.items() if k != 'sort_name'})
        print(f"[System] OpenAlex results ({len(processed_data)} papers) saved to {filename}")

    time.sleep(1)
    return processed_data
