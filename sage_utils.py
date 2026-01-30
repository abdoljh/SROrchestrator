# sage_utils.py
import requests
import csv
import re
import time
import os
from datetime import datetime

def format_sage_authors(author_list):
    """
    Converts SAGE/CrossRef author list into IEEE 'I. Surname'.
    Input: [{'given': 'John', 'family': 'Smith'}, ...]
    """
    if not author_list:
        return "Unknown Author", "Unknown"
    
    formatted = []
    full_names = []
    for auth in author_list:
        family = auth.get('family', '')
        given = auth.get('given', '')
        
        if family and given:
            # Standard IEEE: Initial of first name + Surname
            initial = given[0]
            formatted.append(f"{initial}. {family}")
            full_names.append(f"{family}, {given}")
        elif family:
            formatted.append(family)
            full_names.append(family)
            
    # Get sort key (first author's full name for sorting)
    sort_key = full_names[0] if full_names else "Unknown"
    
    if len(formatted) >= 3:
        return f"{formatted[0]} et al.", sort_key
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}", sort_key
    return formatted[0] if formatted else "Unknown Author", sort_key

def fetch_and_process_sage(query, max_limit=10, save_csv=True):
    """
    Searches SAGE Journals using CrossRef's prefix filter (10.1177).
    """
    base_url = "https://api.crossref.org/works"
    
    params = {
        "query": query,
        # 10.1177 is the unique DOI owner prefix for SAGE Publications
        "filter": "prefix:10.1177",
        "rows": max_limit,
        "select": "DOI,title,author,container-title,published-print,URL"
    }

    # CrossRef 'Polite' User-Agent
    headers = {
        "User-Agent": "ResearchScript/1.0 (mailto:your-email@example.com)"
    }

    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=20)
        if response.status_code != 200:
            print(f"[Error] SAGE (CrossRef) API returned status: {response.status_code}")
            return []
            
        data = response.json()
        entries = data.get('message', {}).get('items', [])
        processed_data = []
        seen_dois = set()

        for entry in entries:
            # Deduplication by DOI
            doi = entry.get('DOI', '')
            if doi and doi in seen_dois:
                continue
            if doi:
                seen_dois.add(doi)

            # Extract Year from publication date parts
            pub_parts = entry.get('published-print', {}).get('date-parts', [[None]])[0]
            year = str(pub_parts[0]) if pub_parts[0] else 'n.d.'

            # Title handling (list to string)
            titles = entry.get('title', ['No Title'])
            title = titles[0] if titles else 'No Title'
            
            # Venue (Journal name)
            venues = entry.get('container-title', ['SAGE Journals'])
            venue = venues[0] if venues else 'SAGE Journals'

            # Format authors and get sort key
            ieee_authors, sort_key = format_sage_authors(entry.get('author'))

            processed_data.append({
                'sort_name': sort_key,
                'ieee_authors': ieee_authors,
                'title': title,
                'venue': venue,
                'year': year,
                'citations': 0, 
                'doi': doi or 'N/A',
                'url': entry.get('URL', '')
            })
            
        # Sort by Author Name
        processed_data.sort(key=lambda x: x['sort_name'].lower())

        # Save to Unique CSV
        if save_csv and processed_data:
            clean_q = re.sub(r"[^\w\s-]", "", query).strip().replace(" ", "_")
            filename = f"sage_{clean_q}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['ieee_authors', 'title', 'venue', 'year', 'citations', 'doi', 'url'])
                writer.writeheader()
                for row in processed_data:
                    # Filter out the helper sort_name key
                    writer.writerow({k: v for k, v in row.items() if k != 'sort_name'})
            print(f"[System] SAGE results ({len(processed_data)} papers) saved to {filename}")

        # Strategic delay for API respect
        time.sleep(1)

        return processed_data
    except Exception as e:
        print(f"[Error] SAGE integration failure: {e}")
        return []