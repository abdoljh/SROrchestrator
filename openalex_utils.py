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
        # Includes summary_stats for impact factor proxy and author h-index
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

        # 2. Venue (Source) and Impact Factor proxy
        primary_loc = work.get("primary_location", {}) or {}
        source = primary_loc.get("source", {}) or {}
        venue = source.get("display_name", "Unknown Venue") if source else "Unknown Venue"

        # Impact Factor proxy: OpenAlex 2-year mean citedness ≈ Journal Impact Factor
        impact_factor = None
        source_stats = source.get("summary_stats", {}) if source else {}
        if source_stats:
            raw_if = source_stats.get("2yr_mean_citedness")
            if raw_if is not None:
                try:
                    impact_factor = round(float(raw_if), 2)
                except (ValueError, TypeError):
                    pass

        # 3. First author h-index (from OpenAlex author summary stats)
        h_index = None
        if authorships:
            first_author_obj = authorships[0].get('author', {}) or {}
            author_stats = first_author_obj.get('summary_stats', {}) or {}
            if author_stats:
                raw_h = author_stats.get('h_index')
                if raw_h is not None:
                    try:
                        h_index = int(raw_h)
                    except (ValueError, TypeError):
                        pass

        # 4. DOI & URL
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
            'url': url,
            'impact_factor': impact_factor,
            'h_index': h_index,
        })

    # Sort alphabetically by surname
    processed_data.sort(key=lambda x: x['sort_name'].lower())

    if save_csv and processed_data:
        clean_q = re.sub(r'[^\w\s-]', '', query).strip().replace(' ', '_')
        filename = f"openalex_{clean_q}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['ieee_authors', 'title', 'venue', 'year', 'citations', 'doi', 'url', 'impact_factor', 'h_index'])
            writer.writeheader()
            for row in processed_data:
                writer.writerow({k: v for k, v in row.items() if k != 'sort_name'})
        print(f"[System] OpenAlex results ({len(processed_data)} papers) saved to {filename}")

    time.sleep(1)
    return processed_data
