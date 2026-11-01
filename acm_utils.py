import requests
import os
import csv
import time
import re
from datetime import datetime
from crossref_common import CROSSREF_SELECT, parse_crossref_item
from query_utils import safe_query_slug, clean_abstract, to_plain_keywords

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
        "query": to_plain_keywords(query),
        # 10.1145 is the unique DOI owner prefix for ACM
        "filter": "prefix:10.1145",
        "rows": min(max_limit * 2, 100),  # oversample: non-papers are dropped
        "select": CROSSREF_SELECT
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

        _seen = set()
        for entry in entries:
            paper = parse_crossref_item(entry, 'ACM Digital Library')
            if paper is None:  # peer reviews, supplementary files, ...
                continue
            key = paper['doi'].lower()
            if key in _seen:
                continue
            _seen.add(key)
            processed_data.append(paper)
            if len(processed_data) >= max_limit:
                break

        # Sort by Author Name
        processed_data.sort(key=lambda x: x['sort_name'].lower())

        # Save to Unique CSV
        if save_csv and processed_data:
            clean_q = safe_query_slug(query)
            filename = f"acm_{clean_q}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['ieee_authors', 'title', 'venue', 'year', 'citations', 'doi', 'url', 'abstract'], extrasaction='ignore')
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
