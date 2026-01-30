import requests
import time
import csv
import re
from datetime import datetime

def format_dblp_authors(author_data):
    """
    Converts DBLP author list into IEEE 'I. Surname'.
    Returns: (formatted_string, sort_key)
    """
    if not author_data:
        return "Unknown Author", "Unknown"

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

    # Get sort key (first author's full name for sorting)
    first_author = authors_list[0] if authors_list else {}
    sort_key = first_author.get('text', '') if isinstance(first_author, dict) else str(first_author)
    if not sort_key:
        sort_key = "Unknown"

    if len(formatted) >= 3:
        return f"{formatted[0]} et al.", sort_key
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}", sort_key
    return formatted[0] if formatted else "Unknown Author", sort_key

def fetch_and_process_dblp(query, max_limit=10, save_csv=True):
    """
    Searches DBLP with strict JSON endpoint and bot-prevention headers.
    The correct API endpoint is /search/publ/api
    Processes papers into IEEE format, sorts by author,
    and saves a unique CSV file.
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

        if not hits:
            return []

        processed_data = []
        seen_ids = set()

        for hit in hits:
            info = hit.get('info', {})
            entry_id = info.get('doi', '') or info.get('ee', '') or info.get('title', '')

            # Deduplication
            if entry_id in seen_ids:
                continue
            seen_ids.add(entry_id)

            author_data = info.get('authors', {}).get('author', [])

            # Format authors and get sort key
            ieee_authors, sort_key = format_dblp_authors(author_data)

            processed_data.append({
                'sort_name': sort_key,
                'ieee_authors': ieee_authors,
                'title': info.get('title', 'No Title'),
                'venue': info.get('venue', 'DBLP Indexed'),
                'year': info.get('year', 'n.d.'),
                'citations': 0, 
                'doi': info.get('doi', 'N/A'),
                'url': info.get('ee', '')
            })

        # Sort by Author Name
        processed_data.sort(key=lambda x: x['sort_name'].lower())

        # Save to Unique CSV
        if save_csv and processed_data:
            clean_q = re.sub(r"[^\w\s-]", "", query).strip().replace(" ", "_")
            filename = f"dblp_{clean_q}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['ieee_authors', 'title', 'venue', 'year', 'citations', 'doi', 'url'])
                writer.writeheader()
                for row in processed_data:
                    # Filter out the helper sort_name key
                    writer.writerow({k: v for k, v in row.items() if k != 'sort_name'})
            print(f"[System] DBLP bulk results ({len(processed_data)} papers) saved to {filename}")

        # Strategic delay for API respect (if called in loops)
        time.sleep(1)

        return processed_data

    except Exception as e:
        print(f"[Error] DBLP integration failure: {e}")
        return []
