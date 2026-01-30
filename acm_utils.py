import requests
import os
import csv
import time
import re
from datetime import datetime

def format_acm_authors(author_list):
    """
    Converts ACM/CrossRef author list into IEEE 'I. Surname'.
    Input: [{'given': 'John', 'family': 'Smith'}, ...]
    Returns: (formatted_string, sort_key)
    """
    if not author_list:
        return "Unknown Author", "Unknown"

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

    # Get sort key (first author's full name for sorting)
    first_author = author_list[0] if author_list else {}
    sort_key = f"{first_author.get('given', '')} {first_author.get('family', '')}".strip()
    if not sort_key:
        sort_key = "Unknown"

    if len(formatted) >= 3:
        return f"{formatted[0]} et al.", sort_key
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}", sort_key
    return formatted[0] if formatted else "Unknown Author", sort_key

def fetch_and_process_acm(query, max_limit=10, save_csv=True):
    """
    Searches ACM Digital Library using CrossRef's prefix filter (10.1145).
    This targets the Association for Computing Machinery specifically.
    Processes papers into IEEE format, sorts by author,
    and saves a unique CSV file.
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

        if not entries:
            return []

        processed_data = []
        seen_ids = set()

        for entry in entries:
            # Deduplication
            entry_id = entry.get('DOI', '')
            if entry_id in seen_ids:
                continue
            seen_ids.add(entry_id)

            # Extract Year from publication date parts
            pub_parts = entry.get('published-print', {}).get('date-parts', [[None]])[0]
            year = str(pub_parts[0]) if pub_parts[0] else 'n.d.'

            # Title handling (standardize list to string)
            titles = entry.get('title', ['No Title'])
            title = titles[0] if titles else 'No Title'

            # Venue (Journal or Conference proceedings name)
            venues = entry.get('container-title', ['ACM Digital Library'])
            venue = venues[0] if venues else 'ACM Digital Library'

            # Format authors and get sort key
            ieee_authors, sort_key = format_acm_authors(entry.get('author'))

            processed_data.append({
                'sort_name': sort_key,
                'ieee_authors': ieee_authors,
                'title': title,
                'venue': venue,
                'year': year,
                'citations': 0, # Live citation counts require CrossRef CitedBy participation
                'doi': entry_id,
                'url': entry.get('URL', '')
            })

        # Sort by Author Name
        processed_data.sort(key=lambda x: x['sort_name'].lower())

        # Save to Unique CSV
        if save_csv and processed_data:
            clean_q = re.sub(r"[^\w\s-]", "", query).strip().replace(" ", "_")
            filename = f"acm_{clean_q}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['ieee_authors', 'title', 'venue', 'year', 'citations', 'doi', 'url'])
                writer.writeheader()
                for row in processed_data:
                    # Filter out the helper sort_name key
                    writer.writerow({k: v for k, v in row.items() if k != 'sort_name'})
            print(f"[System] ACM bulk results ({len(processed_data)} papers) saved to {filename}")

        # Strategic delay for API respect (if called in loops)
        time.sleep(1)

        return processed_data
    except Exception as e:
        print(f"[Error] ACM integration failure: {e}")
        return []
