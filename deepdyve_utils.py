import requests
import xml.etree.ElementTree as ET
import re

def format_pubmed_authors(author_list):
    """Converts PubMed XML author list into IEEE 'I. Surname'."""
    if not author_list:
        return "Unknown Author"
    
    formatted = []
    for auth in author_list:
        surname = auth.findtext('LastName', '')
        initials = auth.findtext('Initials', '')
        if surname:
            formatted.append(f"{initials[0]}. {surname}" if initials else surname)
            
    if len(formatted) >= 3:
        return f"{formatted[0]} et al."
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    return formatted[0] if formatted else "Unknown Author"

def fetch_and_process_deepdyve(query, max_limit=10):
    """
    Searches via PubMed and generates DeepDyve-specific access URLs.
    PubMed API (E-Utilities) Documentation: https://www.ncbi.nlm.nih.gov/books/NBK25500/
    """
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    # Step 1: Search for IDs
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_limit,
        "usehistory": "y"
    }
    
    try:
        search_res = requests.get(search_url, params=search_params, timeout=20)
        search_tree = ET.fromstring(search_res.content)
        ids = [id_el.text for id_el in search_tree.findall(".//IdList/Id")]
        
        if not ids:
            return []

        # Step 2: Fetch Metadata for these IDs
        summary_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml"
        }
        
        summary_res = requests.get(summary_url, params=summary_params, timeout=20)
        summary_tree = ET.fromstring(summary_res.content)
        
        processed = []
        for article in summary_tree.findall(".//PubmedArticle"):
            medline = article.find("MedlineCitation")
            info = medline.find("Article")
            
            # Extract DOI and PMID
            pmid = medline.findtext("PMID")
            doi = ""
            for id_el in article.findall(".//ArticleIdList/ArticleId"):
                if id_el.get("IdType") == "doi":
                    doi = id_el.text

            # DeepDyve URL Construction
            # DeepDyve typically uses DOI or PMID to generate its landing pages
            dd_url = f"https://www.deepdyve.com/lp/doi/{doi}" if doi else f"https://www.deepdyve.com/pubmed/{pmid}"

            processed.append({
                'ieee_authors': format_pubmed_authors(info.findall(".//Author")),
                'title': info.findtext("ArticleTitle"),
                'venue': info.find(".//Journal/Title").text if info.find(".//Journal/Title") is not None else "Unknown Journal",
                'year': info.findtext(".//Journal/JournalIssue/PubDate/Year", "n.d."),
                'citations': 0, # PubMed EFETCH doesn't include live citation counts
                'doi': doi or f"PMID:{pmid}",
                'url': dd_url
            })
            
        return processed
    except Exception as e:
        print(f"[Error] DeepDyve/PubMed integration failure: {e}")
        return []
