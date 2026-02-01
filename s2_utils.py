import time
import csv
import re
from datetime import datetime
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

def abbreviate_venue(venue_name):
    if not venue_name: return "Unknown Venue"
    abbreviations = {
        "Journal": "J.", "Proceedings": "Proc.", "Conference": "Conf.",
        "International": "Int.", "Transactions": "Trans.", "Society": "Soc.",
        "Research": "Res.", "Engineering": "Eng.", "Computer": "Comput.",
        "Science": "Sci.", "Technology": "Technol.", "Intelligence": "Intell.",
        "Communications": "Commun."
    }
    words = venue_name.split()
    return ' '.join([abbreviations.get(word.strip(','), word) for word in words])

def format_author_name(auth):
    name = auth.get('name', 'Unknown')
    parts = name.split()
    return f"{parts[0][0]}. {' '.join(parts[1:])}" if len(parts) > 1 else name

def fetch_and_process_papers(api_key, query, filters=None, save_csv=True, csv_limit=1000):
    http = Session()
    http.mount('https://', HTTPAdapter(max_retries=Retry(total=5, backoff_factor=1)))

    # Added 'citationCount' to the fields list
    params = {
        'query': query, 
        'fields': "paperId,title,year,authors,venue,url,citationCount", 
        'limit': csv_limit 
    }
    if filters:
        params.update(filters)

    response = http.get("https://api.semanticscholar.org/graph/v1/paper/search/bulk",
                        headers={'x-api-key': api_key}, params=params)
    response.raise_for_status()
    
    raw_papers = response.json().get('data', [])

    processed_data = []
    for paper in raw_papers:
        authors = paper.get('authors', [])
        if not authors:
            display_authors, sort_key = "Unknown Author", "Unknown"
        else:
            first_auth = format_author_name(authors[0])
            sort_key = authors[0].get('name', 'Unknown') 
            if len(authors) >= 3:
                display_authors = f"{first_auth} et al."
            elif len(authors) == 2:
                display_authors = f"{first_auth} and {format_author_name(authors[1])}"
            else:
                display_authors = first_auth

        processed_data.append({
            'sort_name': sort_key,
            'ieee_authors': display_authors,
            'title': paper.get('title', 'Untitled Document'),
            'venue': abbreviate_venue(paper.get('venue')),
            'year': paper.get('year', 'n.d.'),
            'citations': paper.get('citationCount', 0), # Fallback to 0 if missing
            'url': paper.get('url', '') 
        })

    # Sorting alphabetically by author
    processed_data.sort(key=lambda x: x['sort_name'].lower())

    if save_csv and processed_data:
        clean_q = re.sub(r'[^\w\s-]', '', query).strip().replace(' ', '_')
        filename = f"s2_{clean_q}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            # Added 'citations' to CSV columns
            writer = csv.DictWriter(f, fieldnames=['ieee_authors', 'title', 'venue', 'year', 'citations', 'url'])
            writer.writeheader()
            for row in processed_data:
                writer.writerow({k: v for k, v in row.items() if k != 'sort_name'})
        print(f"[System] Full bulk results ({len(processed_data)} papers) saved to {filename}")

    time.sleep(1)
    return processed_data
    
