import time
import csv
import re
from datetime import datetime
from serpapi import GoogleSearch

def format_scholar_authors(authors_list):
    if not authors_list: return "Unknown Author"
    formatted = [f"{a.get('name', '').split()[0][0]}. {' '.join(a.get('name', '').split()[1:])}" 
                 if len(a.get('name', '').split()) > 1 else a.get('name', '') for a in authors_list]
    
    if len(formatted) >= 3: return f"{formatted[0]} et al."
    if len(formatted) == 2: return f"{formatted[0]} and {formatted[1]}"
    return formatted[0]

def fetch_and_process_scholar(api_key, query, max_results=50, save_csv=True):
    """
    Fetches Google Scholar results with pagination support.
    max_results: The total number of papers you want (e.g., 100).
    """
    processed_data = []
    results_per_page = 20  # SerpApi default for Google Scholar
    pages_to_fetch = (max_results + results_per_page - 1) // results_per_page

    print(f"[System] Starting retrieval for {max_results} results across {pages_to_fetch} pages...")

    for page in range(pages_to_fetch):
        start_offset = page * results_per_page
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": api_key,
            "num": results_per_page,
            "start": start_offset
        }

        try:
            search = GoogleSearch(params)
            results_dict = search.get_dict()
            
            if "error" in results_dict:
                print(f"[API Error] Page {page+1}: {results_dict['error']}")
                break

            organic_results = results_dict.get("organic_results", [])
            if not organic_results:
                break # Stop if no more results are found

            for item in organic_results:
                if len(processed_data) >= max_results:
                    break
                
                publication_info = item.get("publication_info", {})
                authors_data = publication_info.get("authors", [])
                
                summary = publication_info.get("summary", "")
                year_match = re.search(r'\b(19|20)\d{2}\b', summary)
                year = year_match.group(0) if year_match else "n.d."

                processed_data.append({
                    'sort_name': authors_data[0].get('name', 'Unknown') if authors_data else "Unknown",
                    'ieee_authors': format_scholar_authors(authors_data),
                    'title': item.get('title', 'Untitled Document'),
                    'venue': "Google Scholar",
                    'year': year,
                    'citations': item.get('inline_links', {}).get('cited_by', {}).get('total', 0),
                    'url': item.get('link', '')
                })

            print(f"[Page {page+1}] Retrieved {len(organic_results)} results...")
            
            # Brief delay between page requests to be safe
            time.sleep(0.5)

        except Exception as e:
            print(f"[Critical Error] Page {page+1} failed: {e}")
            break

    # Sort the entire cumulative list by author
    processed_data.sort(key=lambda x: x['sort_name'].lower())

    # Save Cumulative Results to CSV
    if save_csv and processed_data:
        clean_q = re.sub(r'[^\w\s-]', '', query).strip().replace(' ', '_')
        filename = f"scholar_bulk_{clean_q}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['ieee_authors', 'title', 'venue', 'year', 'citations', 'url'])
            writer.writeheader()
            for row in processed_data:
                writer.writerow({k: v for k, v in row.items() if k != 'sort_name'})
        print(f"[System] Success! {len(processed_data)} total papers saved to {filename}")

    time.sleep(1) # Strategic delay after function ends
    return processed_data
