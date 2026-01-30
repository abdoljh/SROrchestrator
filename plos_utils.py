# plos_utils.py
import requests
import re
import os
import csv
import time
from datetime import datetime

def format_plos_authors(author_list):
    """
    Converts PLOS author list (usually a list of strings) into IEEE 'I. Surname'.
    PLOS typically returns authors as ['Surname, Firstname', ...].
    """
    if not author_list or not isinstance(author_list, list):
        return "Unknown Author", "Unknown"

    formatted = []
    for auth in author_list:
        # PLOS format is usually "Surname, Firstname"
        parts = auth.split(',')
        if len(parts) > 1:
            surname = parts[0].strip()
            # Get the first letter of the first name
            first_name = parts[1].strip()
            initial = first_name[0] if first_name else ""
            formatted.append(f"{initial}. {surname}")
        else:
            formatted.append(auth)

    # Get sort key (first author's full name for sorting)
    sort_key = author_list[0] if author_list else "Unknown"

    if len(formatted) >= 3:
        return f"{formatted[0]} et al.", sort_key
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}", sort_key
    return formatted[0] if formatted else "Unknown Author", sort_key

def fetch_and_process_plos(query, max_limit=20, save_csv=True):
    """
    Searches PLOS via their Solr-based Search API.
    API Documentation: http://api.plos.org/
    Processes papers into IEEE format, sorts by author,
    and saves a unique CSV file.
    """
    # PLOS API base URL
    base_url = "http://api.plos.org/search"

    # Parameters for the search
    # 'fl' defines the field list we want returned
    params = {
        "q": 'title:"' + query + '" OR abstract:"' + query + '"',
        "fl": "author_display,title,journal,publication_date,id,counter_total_all",
        "wt": "json",       # Response format
        "rows": max_limit   # Number of results
    }

    try:
        response = requests.get(base_url, params=params, timeout=20)
        if response.status_code != 200:
            print(f"[Error] PLOS API returned status: {response.status_code}")
            return []

        data = response.json()
        # PLOS results are inside response -> docs
        entries = data.get('response', {}).get('docs', [])

        if not entries:
            return []

        processed_data = []
        seen_ids = set()

        for entry in entries:
            # Deduplication
            entry_id = entry.get('id', '')
            if entry_id in seen_ids:
                continue
            seen_ids.add(entry_id)

            # Extract Year from publication_date (format: 2023-10-24T00:00:00Z)
            pub_date = entry.get('publication_date', '')
            year = pub_date.split('-')[0] if pub_date else 'n.d.'

            # Format authors and get sort key
            ieee_authors, sort_key = format_plos_authors(entry.get('author_display'))

            processed_data.append({
                'sort_name': sort_key,
                'ieee_authors': ieee_authors,
                'title': entry.get('title'),
                'venue': entry.get('journal', 'PLOS Indexed Journal'),
                'year': year,
                # PLOS uses 'counter_total_all' for total views/usage as a metric
                'citations': int(entry.get('counter_total_all', 0)),
                'doi': entry_id, # 'id' in PLOS is the DOI
                'url': f"https://journals.plos.org/plosone/article?id={entry_id}"
            })

        # Sort by Author Name
        processed_data.sort(key=lambda x: x['sort_name'].lower())

        # Save to Unique CSV
        if save_csv and processed_data:
            clean_q = re.sub(r"[^\w\s-]", "", query).strip().replace(" ", "_")
            filename = f"plos_{clean_q}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['ieee_authors', 'title', 'venue', 'year', 'citations', 'doi', 'url'])
                writer.writeheader()
                for row in processed_data:
                    # Filter out the helper sort_name key
                    writer.writerow({k: v for k, v in row.items() if k != 'sort_name'})
            print(f"[System] PLOS bulk results ({len(processed_data)} papers) saved to {filename}")

        # Strategic delay for API respect (if called in loops)
        time.sleep(1)

        return processed_data
    except Exception as e:
        print(f"[Error] PLOS integration failure: {e}")
        return []

# Example Usage:
# results = fetch_and_process_plos("machine learning in healthcare", max_limit=5)
# for res in results:
#     print(f"{res['year']} - {res['title']}")
