import time
import csv
import re
import requests
from datetime import datetime

def format_core_authors(authors_list):
    """Standardized IEEE author formatting: I. Surname."""
    if not authors_list:
        return "Unknown Author"
    
    formatted = []
    for author in authors_list:
        name = author.get('name', '')
        parts = name.strip().split()
        if len(parts) > 1:
            formatted.append(f"{parts[0][0]}. {' '.join(parts[1:])}")
        else:
            formatted.append(name if name else "Unknown")
            
    if len(formatted) >= 3:
        return f"{formatted[0]} et al."
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    return formatted[0] if formatted else "Unknown Author"

def fetch_and_process_core(api_key, query, max_limit=20, save_csv=True):
    """
    Highly stable CORE v3 retrieval using POST, Pagination, and Retry Logic.
    """
    base_url = "https://api.core.ac.uk/v3/search/works"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    all_results = []
    limit_per_page = 100 
    offset = 0
    max_retries = 3
    
    print(f"[System] CORE: Fetching up to {max_limit} results...")

    while offset < max_limit:
        payload = {
            "q": query,
            "limit": int(min(limit_per_page, max_limit - offset)),
            "offset": offset
        }
        
        batch_success = False
        for attempt in range(max_retries):
            try:
                # Increased timeout to 90s for large CORE queries
                response = requests.post(base_url, json=payload, headers=headers, timeout=100)
                
                if response.status_code == 200:
                    data = response.json()
                    batch = data.get('results', [])
                    all_results.extend(batch)
                    batch_success = True
                    
                    # If we got fewer results than requested, we've reached the end
                    if len(batch) < payload["limit"]:
                        offset = max_limit # Break outer while
                    break 
                else:
                    print(f"  ! CORE Attempt {attempt+1} failed (Status {response.status_code})")
                    time.sleep(2 ** attempt) # Exponential backoff
            except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
                print(f"  ! CORE Attempt {attempt+1} error: {e}")
                time.sleep(2 ** attempt)

        if not batch_success:
            print("[Critical] CORE: Max retries reached for this batch. Skipping remainder.")
            break
            
        offset += limit_per_page
        time.sleep(0.5)

    processed_data = []
    for item in all_results:
        authors_raw = item.get('authors', [])
        display_authors = format_core_authors(authors_raw)
        
        sort_key = "Unknown"
        if authors_raw:
            last_name_parts = authors_raw[0].get('name', '').split()
            sort_key = last_name_parts[-1] if last_name_parts else "Unknown"

        processed_data.append({
            'sort_name': sort_key,
            'ieee_authors': display_authors,
            'title': item.get('title', 'Untitled Document'),
            'venue': item.get('publisher') or "Open Access Repository",
            'year': item.get('yearPublished', 'n.d.'),
            'citations': "N/A",
            'doi': item.get('doi', 'N/A'),
            'url': item.get('downloadUrl') or f"https://core.ac.uk/works/{item.get('id')}"
        })

    processed_data.sort(key=lambda x: x['sort_name'].lower())

    if save_csv and processed_data:
        clean_q = re.sub(r'[^\w\s-]', '', query).strip().replace(' ', '_')
        filename = f"core_bulk_{clean_q}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['ieee_authors', 'title', 'venue', 'year', 'citations', 'doi', 'url'])
            writer.writeheader()
            for row in processed_data:
                writer.writerow({k: v for k, v in row.items() if k != 'sort_name'})
        print(f"[System] CORE Success: {len(processed_data)} papers processed.")

    return processed_data
