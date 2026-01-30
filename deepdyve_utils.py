import requests
import xml.etree.ElementTree as ET
import re
import csv
import time # Added for API respect
from datetime import datetime

def format_pubmed_authors(author_list):
    if not author_list:
        return "Unknown Author", "Unknown"
    
    formatted = []
    raw_names = []
    for auth in author_list:
        surname = auth.findtext('LastName', '')
        initials = auth.findtext('Initials', '')
        if surname:
            raw_names.append(f"{surname} {initials}".strip())
            formatted.append(f"{initials[0]}. {surname}" if initials else surname)
            
    sort_key = raw_names[0] if raw_names else "Unknown"

    if len(formatted) >= 3:
        return f"{formatted[0]} et al.", sort_key
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}", sort_key
    return formatted[0] if formatted else "Unknown Author", sort_key

def fetch_and_process_deepdyve(query, max_limit=10, save_csv=True):
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    try:
        search_res = requests.get(search_url, params={"db": "pubmed", "term": query, "retmax": max_limit}, timeout=20)
        search_tree = ET.fromstring(search_res.content)
        ids = [id_el.text for id_el in search_tree.findall(".//IdList/Id")]
        if not ids: return []

        summary_res = requests.get(summary_url, params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}, timeout=20)
        summary_tree = ET.fromstring(summary_res.content)
        
        processed = []
        seen_ids = set()

        for article in summary_tree.findall(".//PubmedArticle"):
            medline = article.find("MedlineCitation")
            pmid = medline.findtext("PMID")
            if pmid in seen_ids: continue
            seen_ids.add(pmid)

            info = medline.find("Article")
            doi = next((id_el.text for id_el in article.findall(".//ArticleIdList/ArticleId") if id_el.get("IdType") == "doi"), "")
            dd_url = f"https://www.deepdyve.com/lp/doi/{doi}" if doi else f"https://www.deepdyve.com/pubmed/{pmid}"

            ieee_authors, sort_key = format_pubmed_authors(info.findall(".//Author"))

            processed.append({
                'sort_name': sort_key,
                'ieee_authors': ieee_authors,
                'title': info.findtext("ArticleTitle"),
                'venue': info.find(".//Journal/Title").text if info.find(".//Journal/Title") is not None else "Unknown Journal",
                'year': info.findtext(".//Journal/JournalIssue/PubDate/Year", "n.d."),
                'citations': 0,
                'doi': doi or f"PMID:{pmid}",
                'url': dd_url
            })
            
        processed.sort(key=lambda x: x['sort_name'].lower())

        if save_csv and processed:
            clean_q = re.sub(r"[^\w\s-]", "", query).strip().replace(" ", "_")
            filename = f"deepdyve_{clean_q}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['ieee_authors', 'title', 'venue', 'year', 'citations', 'doi', 'url'])
                writer.writeheader()
                for row in processed:
                    writer.writerow({k: v for k, v in row.items() if k != 'sort_name'})
            print(f"[System] DeepDyve results ({len(processed)} papers) saved to {filename}")

        # Strategic delay for API respect (if called in loops)
        time.sleep(1) # Added to match acm_utils behavior

        return processed
    except Exception as e:
        print(f"[Error] DeepDyve/PubMed integration failure: {e}")
        return []
