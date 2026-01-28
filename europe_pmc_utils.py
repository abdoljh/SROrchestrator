# europe_pmc_utils.py
import requests
import re
import os

def format_epmc_authors(author_str):
    """
    Converts Europe PMC author string 'Surname I, Surname I' into IEEE 'I. Surname'.
    Europe PMC typically returns a single comma-separated string of authors.
    """
    if not author_str:
        return "Unknown Author"
    
    # Split authors by comma
    authors = [a.strip() for a in author_str.split(',') if a.strip()]
    formatted = []
    
    for auth in authors:
        # Europe PMC format is usually "Surname Initial" (e.g., "Smith J")
        parts = auth.rsplit(' ', 1)
        if len(parts) > 1:
            surname = parts[0]
            initial = parts[1][0]
            formatted.append(f"{initial}. {surname}")
        else:
            formatted.append(auth)
            
    if len(formatted) >= 3:
        return f"{formatted[0]} et al."
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    return formatted[0] if formatted else "Unknown Author"

def fetch_and_process_epmc(query, max_limit=20):
    """
    Searches Europe PMC via their RESTful API.
    API Documentation: https://europepmc.org/RestfulWebService
    """
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    
    # Parameters for the search
    params = {
        "query": query,         # Supports advanced syntax like TITLE:"query"
        "format": "json",       # Ensure JSON response
        "pageSize": max_limit,  # Number of results per page
        "resultType": "core"    # 'core' provides full metadata including citations
    }

    try:
        response = requests.get(base_url, params=params, timeout=20)
        if response.status_code != 200:
            print(f"[Error] Europe PMC API returned status: {response.status_code}")
            return []
            
        data = response.json()
        # Europe PMC nests results under resultList -> result
        entries = data.get('resultList', {}).get('result', [])
        processed = []

        for entry in entries:
            # Extract Year
            pub_year = entry.get('pubYear', 'n.d.')

            processed.append({
                'ieee_authors': format_epmc_authors(entry.get('authorString')),
                'title': entry.get('title'),
                'venue': entry.get('journalTitle', 'Europe PMC Indexed Journal'),
                'year': pub_year,
                'citations': int(entry.get('citedByCount', 0)),
                'doi': entry.get('doi', 'N/A'),
                # Europe PMC provides multiple links; usually, the first one is the landing page
                'url': f"https://europepmc.org/article/MED/{entry.get('id')}" if 'id' in entry else ""
            })
            
        return processed
    except Exception as e:
        print(f"[Error] Europe PMC integration failure: {e}")
        return []

        
        
        
        
