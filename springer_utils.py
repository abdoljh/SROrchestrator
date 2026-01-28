import requests
import os
import xml.etree.ElementTree as ET

def format_springer_pam_authors(author_elements):
    """
    Converts XML <dc:creator> elements into IEEE 'I. Surname'.
    Springer PAM returns: <dc:creator>Surname, Firstname</dc:creator>
    """
    if not author_elements:
        return "Unknown Author"
    
    formatted = []
    for creator in author_elements:
        name = creator.text if creator.text else ""
        if ',' in name:
            surname, first_part = name.split(',', 1)
            initial = first_part.strip()[0] if first_part.strip() else ""
            formatted.append(f"{initial}. {surname.strip()}")
        else:
            formatted.append(name)
            
    if len(formatted) >= 3:
        return f"{formatted[0]} et al."
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    return formatted[0] if formatted else "Unknown Author"

def fetch_and_process_springer(query, max_limit=5):
    """
    Searches Springer Nature using the Meta v2 PAM (XML) endpoint.
    """
    api_key = os.environ.get('META_SPRINGER_API_KEY')
    base_url = 'https://api.springernature.com/meta/v2/pam'
    
    params = {
        'api_key': api_key,
        's': 1,
        'p': max_limit,
        'q': f'(keyword:"{query}")'
    }

    # Define XML namespaces used in Springer's PAM response
    ns = {
        'dc': 'http://purl.org/dc/elements/1.1/',
        'prism': 'http://prismstandard.org/namespaces/basic/2.2/',
        'pam': 'http://prismstandard.org/namespaces/pam/2.2/'
    }

    try:
        response = requests.get(base_url, params=params, timeout=20)
        if response.status_code != 200:
            print(f"[Error] Springer API Status {response.status_code}")
            return []

        # Parse the XML response
        root = ET.fromstring(response.content)
        records = root.findall(".//record")
        processed = []

        for record in records:
            # Metadata is nested within pam:article/xhtml:head (or similar)
            # We search for the specific tags using namespaces
            head = record.find(".//{http://www.w3.org/1999/xhtml}head")
            if head is None: continue

            title = head.findtext(f"{{{ns['dc']}}}title")
            doi = head.findtext(f"{{{ns['prism']}}}doi")
            venue = head.findtext(f"{{{ns['prism']}}}publicationName")
            pub_date = head.findtext(f"{{{ns['prism']}}}publicationDate")
            year = pub_date.split('-')[0] if pub_date else "n.d."
            
            # Extract authors (dc:creator)
            creators = head.findall(f"{{{ns['dc']}}}creator")
            
            # Find the best URL (preferring web link)
            url = ""
            urls = head.findall(f"{{{ns['prism']}}}url")
            for u in urls:
                if u.get("{http://prismstandard.org/namespaces/basic/2.2/}platform") == "web":
                    url = u.text
                    break
            if not url and urls: url = urls[0].text

            processed.append({
                'ieee_authors': format_springer_pam_authors(creators),
                'title': title,
                'venue': venue or 'Springer Nature',
                'year': year,
                'citations': 0, # XML Metadata doesn't include counts
                'doi': doi or 'N/A',
                'url': url
            })
            
        return processed
    except Exception as e:
        print(f"[Error] Springer XML integration failure: {e}")
        return []
