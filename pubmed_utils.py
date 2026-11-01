import time
import csv
import re
from datetime import datetime
from query_utils import safe_query_slug, clean_abstract

def abbreviate_venue(venue_name):
    if not venue_name: return "Unknown Journal"
    abbreviations = {
        "Journal": "J.", "Proceedings": "Proc.", "Conference": "Conf.",
        "International": "Int.", "Transactions": "Trans.", "Society": "Soc.",
        "Research": "Res.", "Engineering": "Eng.", "Computer": "Comput.",
        "Science": "Sci.", "Technology": "Technol.", "Intelligence": "Intell."
    }
    words = venue_name.split()
    return ' '.join([abbreviations.get(word.strip(','), word) for word in words])

def format_pubmed_author(author):
    """Formats PubMed author objects to 'I. Surname'."""
    last_name = author.get('LastName', '')
    initials = author.get('Initials', '')
    if last_name and initials:
        # PubMed initials are usually 'JD', we add a period to the first
        return f"{initials[0]}. {last_name}"
    return author.get('CollectiveName', 'Unknown')

def parse_pubmed_xml(content: bytes):
    """Parse an efetch PubmedArticleSet (plain ElementTree; no DTD download needed)."""
    import xml.etree.ElementTree as ET
    out = []
    for art in ET.fromstring(content).findall(".//PubmedArticle"):
        med = art.find("MedlineCitation")
        info = med.find("Article") if med is not None else None
        if info is None:
            continue
        pmid = med.findtext("PMID", "")

        authors = []
        for au in info.findall(".//AuthorList/Author"):
            last, ini = au.findtext("LastName", ""), au.findtext("Initials", "")
            if last:
                authors.append((f"{ini[0]}. {last}" if ini else last, last))
            elif au.findtext("CollectiveName"):
                authors.append((au.findtext("CollectiveName"), au.findtext("CollectiveName")))
        if not authors:
            display, sort_key = "Unknown Author", "Unknown"
        else:
            sort_key = authors[0][1]
            display = (f"{authors[0][0]} et al." if len(authors) >= 3 else
                       f"{authors[0][0]} and {authors[1][0]}" if len(authors) == 2 else authors[0][0])

        parts = []
        for el in info.findall(".//Abstract/AbstractText"):
            text = "".join(el.itertext()).strip()
            if text:
                parts.append(f"{el.get('Label')}: {text}" if el.get("Label") else text)

        year = (info.findtext(".//Journal/JournalIssue/PubDate/Year")
                or (info.findtext(".//Journal/JournalIssue/PubDate/MedlineDate") or "")[:4] or "n.d.")
        doi = next((e.text for e in art.findall(".//PubmedData/ArticleIdList/ArticleId")
                    if e.get("IdType") == "doi" and e.text), "")
        out.append({
            'sort_name': sort_key,
            'ieee_authors': display,
            'title': "".join(info.find("ArticleTitle").itertext()).strip() if info.find("ArticleTitle") is not None else 'Untitled Document',
            'venue': abbreviate_venue(info.findtext(".//Journal/Title", "Unknown Journal")),
            'year': year,
            'citations': "N/A",
            'doi': doi or 'N/A',
            'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            'abstract': clean_abstract(' '.join(parts)),
        })
    return out


def fetch_and_process_pubmed(query, max_limit=10, save_csv=True):
    """
    PubMed via NCBI E-utilities (esearch + efetch), parsed with ElementTree.
    (Bio.Entrez.read needs to fetch DTDs, which failed silently on hosted deployments.)
    """
    import os
    import requests
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    common = {"tool": "SROrch"}
    if os.getenv("USER_EMAIL"):
        common["email"] = os.getenv("USER_EMAIL")

    r = requests.get(f"{base}/esearch.fcgi", timeout=20,
                     params={**common, "db": "pubmed", "term": query, "retmax": max_limit, "retmode": "json"})
    if r.status_code != 200:
        raise RuntimeError(f"PubMed esearch HTTP {r.status_code}: {r.text[:120]}")
    id_list = r.json().get("esearchresult", {}).get("idlist", [])
    if not id_list:
        return []
    time.sleep(0.4)  # NCBI: max 3 requests/second without an API key
    r = requests.get(f"{base}/efetch.fcgi", timeout=30,
                     params={**common, "db": "pubmed", "id": ",".join(id_list), "retmode": "xml"})
    if r.status_code != 200:
        raise RuntimeError(f"PubMed efetch HTTP {r.status_code}: {r.text[:120]}")
    processed_data = parse_pubmed_xml(r.content)

    # 3. Sort by Author
    processed_data.sort(key=lambda x: x['sort_name'].lower())

    # 4. Save to Unique CSV
    if save_csv and processed_data:
        clean_q = safe_query_slug(query)
        filename = f"pubmed_{clean_q}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['ieee_authors', 'title', 'venue', 'year', 'citations', 'url', 'abstract'], extrasaction='ignore')
            writer.writeheader()
            for row in processed_data:
                writer.writerow({k: v for k, v in row.items() if k != 'sort_name'})
        print(f"[System] PubMed bulk results ({len(processed_data)} papers) saved to {filename}")

    time.sleep(1) # Strategic Delay
    return processed_data
