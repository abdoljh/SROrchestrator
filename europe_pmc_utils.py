import requests
import re
import csv
import time  # Added for API respect
from datetime import datetime

def format_epmc_authors(author_str):
    if not author_str:
        return "Unknown Author", "Unknown"
    
    authors = [a.strip() for a in author_str.split(',') if a.strip()]
    formatted = []
    
    for auth in authors:
        parts = auth.rsplit(' ', 1)
        if len(parts) > 1:
            surname, initial = parts[0], parts[1][0]
            formatted.append(f"{initial}. {surname}")
        else:
            formatted.append(auth)
            
    sort_key = authors[0] if authors else "Unknown"
    
    if len(formatted) >= 3:
        return f"{formatted[0]} et al.", sort_key
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}", sort_key
    return formatted[0] if formatted else "Unknown Author", sort_key

def fetch_and_process_europe_pmc(query, max_limit=20, save_csv=True):
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {"query": query, "format": "json", "pageSize": max_limit, "resultType": "core"}

    try:
        response = requests.get(base_url, params=params, timeout=20)
        if response.status_code != 200:
            return []
            
        entries = response.json().get('resultList', {}).get('result', [])
        processed = []
        seen_ids = set()

        for entry in entries:
            entry_id = entry.get('doi') or entry.get('id')
            if entry_id in seen_ids: continue
            seen_ids.add(entry_id)

            ieee_authors, sort_key = format_epmc_authors(entry.get('authorString'))

            processed.append({
                'sort_name': sort_key,
                'ieee_authors': ieee_authors,
                'title': entry.get('title'),
                'venue': entry.get('journalTitle', 'Europe PMC Indexed Journal'),
                'year': entry.get('pubYear', 'n.d.'),
                'citations': int(entry.get('citedByCount', 0)),
                'doi': entry.get('doi', 'N/A'),
                'url': f"https://europepmc.org/article/MED/{entry.get('id')}" if 'id' in entry else ""
            })
            
        processed.sort(key=lambda x: x['sort_name'].lower())

        if save_csv and processed:
            clean_q = re.sub(r"[^\w\s-]", "", query).strip().replace(" ", "_")
            filename = f"epmc_{clean_q}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['ieee_authors', 'title', 'venue', 'year', 'citations', 'doi', 'url'])
                writer.writeheader()
                for row in processed:
                    writer.writerow({k: v for k, v in row.items() if k != 'sort_name'})
            print(f"[System] Europe PMC results ({len(processed)} papers) saved to {filename}")

        # Strategic delay for API respect (if called in loops)
        time.sleep(1) # Added to match acm_utils behavior

        return processed
    except Exception as e:
        print(f"[Error] Europe PMC integration failure: {e}")
        return []
