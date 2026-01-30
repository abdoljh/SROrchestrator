import requests
import os
import xml.etree.ElementTree as ET
import csv
import re
import time
from datetime import datetime

def format_springer_pam_authors(author_elements):
    if not author_elements: return "Unknown Author", "Unknown"
    formatted = []
    for creator in author_elements:
        name = creator.text or ""
        if ',' in name:
            surname, first = name.split(',', 1)
            formatted.append(f"{first.strip()[0]}. {surname.strip()}")
        else: formatted.append(name)
            
    sort_key = author_elements[0].text if author_elements else "Unknown"
    if len(formatted) >= 3: return f"{formatted[0]} et al.", sort_key
    elif len(formatted) == 2: return f"{formatted[0]} and {formatted[1]}", sort_key
    return formatted[0], sort_key

def fetch_and_process_springer(query, max_limit=5, save_csv=True):
    api_key = os.environ.get('META_SPRINGER_API_KEY')
    base_url = 'https://api.springernature.com/meta/v2/pam'
    params = {'api_key': api_key, 'p': max_limit, 'q': f'(keyword:"{query}")'}
    ns = {'dc': 'http://purl.org/dc/elements/1.1/', 'prism': 'http://prismstandard.org/namespaces/basic/2.2/'}

    try:
        response = requests.get(base_url, params=params, timeout=20)
        root = ET.fromstring(response.content)
        processed, seen_ids = [], set()

        for record in root.findall(".//record"):
            head = record.find(".//{http://www.w3.org/1999/xhtml}head")
            if head is None: continue
            doi = head.findtext(f"{{{ns['prism']}}}doi")
            if doi in seen_ids: continue
            seen_ids.add(doi)

            ieee_authors, sort_key = format_springer_pam_authors(head.findall(f"{{{ns['dc']}}}creator"))
            processed.append({
                'sort_name': sort_key, 'ieee_authors': ieee_authors, 'title': head.findtext(f"{{{ns['dc']}}}title"),
                'venue': head.findtext(f"{{{ns['prism']}}}publicationName") or 'Springer Nature',
                'year': (head.findtext(f"{{{ns['prism']}}}publicationDate") or "n.d.").split('-')[0],
                'citations': 0, 'doi': doi or 'N/A', 'url': head.find(".//{*}url").text if head.find(".//{*}url") is not None else ""
            })
            
        processed.sort(key=lambda x: x['sort_name'].lower())
        if save_csv and processed:
            clean_q = re.sub(r"[^\w\s-]", "", query).strip().replace(" ", "_")
            filename = f"springer_{clean_q}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['ieee_authors', 'title', 'venue', 'year', 'citations', 'doi', 'url'])
                writer.writeheader()
                for row in processed: writer.writerow({k: v for k, v in row.items() if k != 'sort_name'})
            print(f"[System] Springer Nature results ({len(processed)} papers) saved to {filename}")

        # Strategic delay for API respect (if called in loops)
        time.sleep(1)
        return processed
    except Exception as e:
        print(f"[Error] Springer failure: {e}")
        return []
        
    
