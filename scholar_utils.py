import time
import csv
import re
from datetime import datetime
#from serpapi import GoogleSearch
from serpapi.google_search import GoogleSearch

def format_scholar_authors(authors_list):
    if not authors_list:
        return "Unknown Author"
    
    formatted = []
    for auth in authors_list:
        full_name = auth.get('name', '')
        parts = full_name.split()
        if len(parts) > 1:
            formatted.append(f"{parts[0][0]}. {' '.join(parts[1:])}")
        else:
            formatted.append(full_name)
            
    if len(formatted) >= 3:
        return f"{formatted[0]} et al."
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    return formatted[0]

def fetch_and_process_scholar(api_key, query, max_limit=10, save_csv=True):
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": api_key,
        "num": max_limit
    }

    try:
        search = GoogleSearch(params)
        results_dict = search.get_dict()
        
        # Check for API errors in the response
        if "error" in results_dict:
            print(f"[API Error] {results_dict['error']}")
            return []

        organic_results = results_dict.get("organic_results", [])
        if not organic_results:
            print(f"[System] No results found for query: {query}")
            return []

    except Exception as e:
        print(f"[Critical Error] failure: {e}")
        return []

    processed_data = []
    
    for item in organic_results:
        publication_info = item.get("publication_info", {})
        authors_data = publication_info.get("authors", [])
        
        display_authors = format_scholar_authors(authors_data)
        sort_key = authors_data[0].get('name', 'Unknown') if authors_data else "Unknown"

        summary = publication_info.get("summary", "")
        year_match = re.search(r'\b(19|20)\d{2}\b', summary)
        year = year_match.group(0) if year_match else "n.d."

        processed_data.append({
            'sort_name': sort_key,
            'ieee_authors': display_authors,
            'title': item.get('title', 'Untitled Document'),
            'venue': "Google Scholar",
            'year': year,
            'citations': item.get('inline_links', {}).get('cited_by', {}).get('total', 0),
            'url': item.get('link', '')
        })

    processed_data.sort(key=lambda x: x['sort_name'].lower())

    if save_csv and processed_data:
        clean_q = re.sub(r'[^\w\s-]', '', query).strip().replace(' ', '_')
        filename = f"scholar_{clean_q}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['ieee_authors', 'title', 'venue', 'year', 'citations', 'url'])
            writer.writeheader()
            for row in processed_data:
                writer.writerow({k: v for k, v in row.items() if k != 'sort_name'})
        print(f"[System] Success! {len(processed_data)} papers saved to {filename}")

    time.sleep(1)
    return processed_data
