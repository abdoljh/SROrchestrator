import arxiv
import re
import csv
import time
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

def format_author_name(author_obj):
    """Converts 'Full Name' to 'F. Surname' to match IEEE style."""
    name_parts = author_obj.name.split()
    if len(name_parts) > 1:
        return f"{name_parts[0][0]}. {' '.join(name_parts[1:])}"
    return author_obj.name

def fetch_and_process_arxiv(query, max_limit=10, save_csv=True):
    """
    Searches arXiv, processes papers into IEEE format, sorts by author,
    and saves a unique CSV file.
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_limit,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = list(client.results(search))
    except Exception as e:
        print(f"Error accessing arXiv API: {e}")
        return []

    if not results:
        return []

    processed_data = []
    seen_ids = set()

    for result in results:
        # Deduplication
        if result.entry_id in seen_ids:
            continue
        seen_ids.add(result.entry_id)

        # 1. Author Logic (IEEE Initials + et al.)
        authors = result.authors
        if not authors:
            display_authors, sort_key = "Unknown Author", "Unknown"
        else:
            first_auth_formatted = format_author_name(authors[0])
            sort_key = authors[0].name # For alphabetical sorting
            if len(authors) >= 3:
                display_authors = f"{first_auth_formatted} et al."
            elif len(authors) == 2:
                display_authors = f"{first_auth_formatted} and {format_author_name(authors[1])}"
            else:
                display_authors = first_auth_formatted

        # 2. Extract arXiv ID
        arxiv_id_match = re.search(r'([0-9]{4}\.[0-9]{5}(v[0-9]+)?)', result.entry_id)
        arxiv_id = arxiv_id_match.group(0) if arxiv_id_match else "N/A"

        processed_data.append({
            'sort_name': sort_key,
            'ieee_authors': display_authors,
            'title': result.title,
            'venue': f"arXiv preprint arXiv:{arxiv_id}",
            'year': result.published.year,
            'citations': "N/A", # arXiv API doesn't provide citation counts natively
            'url': result.entry_id
        })

    # 3. Sort by Author Name
    processed_data.sort(key=lambda x: x['sort_name'].lower())

    # 4. Save to Unique CSV
    if save_csv and processed_data:
        clean_q = re.sub(r'[^\w\s-]', '', query).strip().replace(' ', '_')
        filename = f"arxiv_{clean_q}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['ieee_authors', 'title', 'venue', 'year', 'citations', 'url'])
            writer.writeheader()
            for row in processed_data:
                # Filter out the helper sort_name key
                writer.writerow({k: v for k, v in row.items() if k != 'sort_name'})
        print(f"[System] arXiv bulk results ({len(processed_data)} papers) saved to {filename}")

    # Strategic delay for API respect (if called in loops)
    time.sleep(1)
    return processed_data
