import time
import csv
import re
from datetime import datetime
from Bio import Entrez

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

#def fetch_and_process_pubmed(email, query, max_limit=10, save_csv=True):
def fetch_and_process_pubmed(query, max_limit=10, save_csv=True):
    import os
    Entrez.email = os.getenv("USER_EMAIL")
    #Entrez.email = email
    processed_data = []

    try:
        # 1. Search for IDs
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_limit, retmode="xml")
        record = Entrez.read(handle)
        handle.close()
        id_list = record.get("IdList", [])

        if not id_list:
            return []

        # 2. Fetch Details
        fetch_handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="xml")
        papers = Entrez.read(fetch_handle)
        fetch_handle.close()

        for article_data in papers['PubmedArticle']:
            citation = article_data['MedlineCitation']
            article = citation['Article']
            pmid = str(citation['PMID'])
            
            # Author Logic
            auth_list = article.get('AuthorList', [])
            if not auth_list:
                display_authors, sort_key = "Unknown Author", "Unknown"
            else:
                first_auth = format_pubmed_author(auth_list[0])
                sort_key = auth_list[0].get('LastName', 'Unknown')
                if len(auth_list) >= 3:
                    display_authors = f"{first_auth} et al."
                elif len(auth_list) == 2:
                    display_authors = f"{first_auth} and {format_pubmed_author(auth_list[1])}"
                else:
                    display_authors = first_auth

            # Venue and Date
            raw_venue = article.get('Journal', {}).get('Title', 'Unknown Journal')
            year = 'n.d.'
            pub_date = article.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
            if 'Year' in pub_date:
                year = pub_date['Year']
            
            processed_data.append({
                'sort_name': sort_key,
                'ieee_authors': display_authors,
                'title': article.get('ArticleTitle', 'Untitled Document'),
                'venue': abbreviate_venue(raw_venue),
                'year': year,
                'citations': "N/A", # PubMed API requires a separate link-out for citations
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            })

    except Exception as e:
        print(f"[Error] PubMed API failure: {e}")
        return []

    # 3. Sort by Author
    processed_data.sort(key=lambda x: x['sort_name'].lower())

    # 4. Save to Unique CSV
    if save_csv and processed_data:
        clean_q = re.sub(r'[^\w\s-]', '', query).strip().replace(' ', '_')
        filename = f"pubmed_{clean_q}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['ieee_authors', 'title', 'venue', 'year', 'citations', 'url'])
            writer.writeheader()
            for row in processed_data:
                writer.writerow({k: v for k, v in row.items() if k != 'sort_name'})
        print(f"[System] PubMed bulk results ({len(processed_data)} papers) saved to {filename}")

    time.sleep(1) # Strategic Delay
    return processed_data
