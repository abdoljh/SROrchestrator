# ==============================================
# latex_export.py
# SROrch LaTeX Export Module
# ==============================================

"""
Generates publication-ready LaTeX (.tex) and BibTeX (.bib) files
from SROrch verified report data. Supports IEEE, ACM, Springer,
Elsevier, and Plain article templates.
"""

import re
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional


# ==============================================
# LATEX CHARACTER ESCAPING
# ==============================================

def escape_latex(text: str) -> str:
    """Escape special LaTeX characters while preserving intentional commands."""
    if not text or not isinstance(text, str):
        return ''

    # Preserve existing LaTeX math mode and commands
    preserved = []
    counter = [0]

    def preserve(match):
        preserved.append(match.group(0))
        placeholder = f"__LATEX_PRESERVE_{counter[0]}__"
        counter[0] += 1
        return placeholder

    # Preserve $...$ math mode and \command sequences
    result = re.sub(r'\$[^$]+\$', preserve, text)
    result = re.sub(r'\\[a-zA-Z]+(?:\{[^}]*\})?', preserve, result)

    # Escape special characters (order matters)
    char_map = [
        ('\\', r'\textbackslash{}'),
        ('&', r'\&'),
        ('%', r'\%'),
        ('$', r'\$'),
        ('#', r'\#'),
        ('_', r'\_'),
        ('{', r'\{'),
        ('}', r'\}'),
        ('~', r'\textasciitilde{}'),
        ('^', r'\textasciicircum{}'),
    ]
    for char, replacement in char_map:
        result = result.replace(char, replacement)

    # Restore preserved sequences
    for i, original in enumerate(preserved):
        result = result.replace(f"__LATEX_PRESERVE_{i}__", original)

    return result


# ==============================================
# HTML TO LATEX CONVERSION
# ==============================================

def html_to_latex(text: str) -> str:
    """Convert basic HTML formatting to LaTeX equivalents."""
    if not text or not isinstance(text, str):
        return ''

    result = text

    # Convert HTML tags to LaTeX
    result = re.sub(r'</?p>', '\n\n', result)
    result = re.sub(r'<br\s*/?>', r'\\\\', result)
    result = re.sub(r'<i>(.*?)</i>', r'\\textit{\1}', result, flags=re.DOTALL)
    result = re.sub(r'<em>(.*?)</em>', r'\\textit{\1}', result, flags=re.DOTALL)
    result = re.sub(r'<b>(.*?)</b>', r'\\textbf{\1}', result, flags=re.DOTALL)
    result = re.sub(r'<strong>(.*?)</strong>', r'\\textbf{\1}', result, flags=re.DOTALL)
    result = re.sub(r'<a\s+href="([^"]*)"[^>]*>(.*?)</a>', r'\\href{\1}{\2}', result, flags=re.DOTALL)

    # Remove any remaining HTML tags
    result = re.sub(r'<[^>]+>', '', result)

    # Clean up multiple blank lines
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result.strip()


def format_section_content(text: str) -> str:
    """Convert section content from HTML to clean LaTeX paragraphs."""
    if not text:
        return ''

    # First convert HTML, then escape special characters in the plain text parts
    content = html_to_latex(text)

    # Escape LaTeX special chars but preserve already-converted commands
    # Split by LaTeX commands to avoid double-escaping
    parts = re.split(r'(\\[a-zA-Z]+\{[^}]*\}|\\[a-zA-Z]+)', content)
    escaped_parts = []
    for i, part in enumerate(parts):
        if i % 2 == 0:  # Plain text
            escaped_parts.append(escape_latex(part))
        else:  # LaTeX command - keep as is
            escaped_parts.append(part)

    return ''.join(escaped_parts)


# ==============================================
# CITATION CONVERSION
# ==============================================

def generate_cite_key(source: Dict, index: int) -> str:
    """Generate a BibTeX citation key from source metadata."""
    meta = source.get('metadata', {})
    authors = meta.get('authors', 'Unknown')

    # Extract first author surname
    if isinstance(authors, str):
        first_author = authors.split(',')[0].split(' and ')[0].strip()
        surname = first_author.split()[-1] if ' ' in first_author else first_author
    else:
        surname = 'Unknown'

    # Clean surname for use as key
    surname = re.sub(r'[^a-zA-Z]', '', surname)
    if not surname:
        surname = 'Unknown'

    year = str(meta.get('year', 'nd')).strip()
    if not year or year == 'N/A':
        year = 'nd'

    return f"{surname}{year}_{index}"


def convert_citations_to_latex(text: str, cite_key_map: Dict[int, str]) -> str:
    """Convert [N] citation markers to \\cite{key} commands."""
    if not text:
        return ''

    def replace_citation(match):
        num = int(match.group(1))
        key = cite_key_map.get(num)
        if key:
            return f'\\cite{{{key}}}'
        return match.group(0)

    # Handle grouped citations like [1, 2, 3] or [1][2][3]
    # First handle comma-separated: [1, 2, 3]
    def replace_grouped(match):
        nums_str = match.group(1)
        nums = [int(n.strip()) for n in nums_str.split(',') if n.strip().isdigit()]
        keys = [cite_key_map[n] for n in nums if n in cite_key_map]
        if keys:
            return f'\\cite{{{",".join(keys)}}}'
        return match.group(0)

    result = re.sub(r'\[([\d,\s]+)\]', replace_grouped, text)
    # Handle remaining single citations
    result = re.sub(r'\[(\d+)\]', replace_citation, result)

    return result


# ==============================================
# BIBTEX GENERATION
# ==============================================

def escape_bibtex(text: str) -> str:
    """Escape text for BibTeX fields."""
    if not text or not isinstance(text, str):
        return ''
    return text.replace('{', '\\{').replace('}', '\\}')


def generate_bibtex(
    sources: List[Dict],
    cited_refs: set,
    old_to_new: Dict[int, int]
) -> Tuple[str, Dict[int, str]]:
    """
    Generate BibTeX entries for cited sources.
    Returns: (bibtex_string, cite_key_map: {new_ref_num -> cite_key})
    """
    entries = []
    cite_key_map = {}

    cited_sorted = sorted(cited_refs)

    for old_ref_num in cited_sorted:
        new_ref_num = old_to_new.get(old_ref_num, old_ref_num)
        if old_ref_num > len(sources):
            continue

        source = sources[old_ref_num - 1]
        meta = source.get('metadata', {})
        orch = source.get('_orchestrator_data', {})

        cite_key = generate_cite_key(source, new_ref_num)
        cite_key_map[new_ref_num] = cite_key

        # Determine entry type
        venue = str(meta.get('venue', '')).lower()
        url = source.get('url', '').lower()

        if 'conference' in venue or 'proceedings' in venue or 'proc.' in venue:
            entry_type = 'inproceedings'
        elif 'arxiv' in venue or 'arxiv' in url:
            entry_type = 'misc'
        else:
            entry_type = 'article'

        # Build entry
        lines = [f"@{entry_type}{{{cite_key},"]
        lines.append(f"  title = {{{escape_bibtex(meta.get('title', 'Untitled'))}}},")
        lines.append(f"  author = {{{escape_bibtex(meta.get('authors', 'Unknown'))}}},")
        lines.append(f"  year = {{{meta.get('year', 'n.d.')}}},")

        venue_str = meta.get('venue', '')
        if venue_str and venue_str != 'N/A':
            journal_key = 'booktitle' if entry_type == 'inproceedings' else 'journal'
            lines.append(f"  {journal_key} = {{{escape_bibtex(venue_str)}}},")

        doi = meta.get('doi', '')
        if doi and doi not in ('N/A', 'NA', 'Unknown', ''):
            clean_doi = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '').strip()
            if clean_doi:
                lines.append(f"  doi = {{{clean_doi}}},")

        paper_url = source.get('url', '')
        if paper_url:
            lines.append(f"  url = {{{paper_url}}},")

        abstract = orch.get('abstract', '') or meta.get('abstract', '')
        if abstract and abstract != 'Abstract not available.':
            lines.append(f"  abstract = {{{escape_bibtex(abstract[:500])}}},")

        # Remove trailing comma from last field
        if lines[-1].endswith(','):
            lines[-1] = lines[-1][:-1]

        lines.append("}")
        entries.append('\n'.join(lines))

    return '\n\n'.join(entries), cite_key_map


# ==============================================
# TEMPLATE PREAMBLES
# ==============================================

def get_latex_preamble(template: str, form_data: Dict) -> str:
    """Generate LaTeX preamble for the selected template."""
    topic = escape_latex(form_data.get('topic', 'Research Report'))
    researcher = escape_latex(form_data.get('researcher', 'Author'))
    institution = escape_latex(form_data.get('institution', ''))

    try:
        report_date = datetime.strptime(
            form_data.get('date', ''), '%Y-%m-%d'
        ).strftime('%B %d, %Y')
    except (ValueError, TypeError):
        report_date = datetime.now().strftime('%B %d, %Y')

    common_packages = r"""
\usepackage[utf8]{inputenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{graphicx}
"""

    if template == 'ieee':
        return rf"""\documentclass[conference]{{IEEEtran}}
{common_packages}
\usepackage{{cite}}
\title{{{topic}}}
\author{{\IEEEauthorblockN{{{researcher}}}
\IEEEauthorblockA{{{institution}\\
{report_date}}}}}
"""

    elif template == 'acm':
        return rf"""\documentclass[sigconf]{{acmart}}
{common_packages}
\title{{{topic}}}
\author{{{researcher}}}
\affiliation{{%
  \institution{{{institution}}}
}}
\date{{{report_date}}}
"""

    elif template == 'springer':
        return rf"""\documentclass{{llncs}}
{common_packages}
\usepackage{{cite}}
\title{{{topic}}}
\author{{{researcher}}}
\institute{{{institution}}}
"""

    elif template == 'elsevier':
        return rf"""\documentclass[preprint,12pt]{{elsarticle}}
{common_packages}
\journal{{}}
\begin{{frontmatter}}
\title{{{topic}}}
\author{{{researcher}}}
\address{{{institution}}}
\date{{{report_date}}}
"""

    else:  # plain
        return rf"""\documentclass[12pt]{{article}}
{common_packages}
\usepackage{{cite}}
\usepackage[margin=1in]{{geometry}}
\title{{{topic}}}
\author{{{researcher} \\ {institution}}}
\date{{{report_date}}}
"""


def get_bib_style(template: str) -> str:
    """Return the appropriate bibliography style for the template."""
    styles = {
        'ieee': 'IEEEtran',
        'acm': 'ACM-Reference-Format',
        'springer': 'splncs04',
        'elsevier': 'elsarticle-num',
        'plain': 'plain',
    }
    return styles.get(template, 'plain')


# ==============================================
# MAIN LATEX GENERATION
# ==============================================

def generate_latex_report(
    refined_draft: Dict,
    form_data: Dict,
    sources: List[Dict],
    verification_report: Dict,
    template: str = 'ieee',
    max_sources: int = 25
) -> Tuple[str, str]:
    """
    Generate a complete LaTeX report with BibTeX.

    Args:
        refined_draft: The refined draft dict with sections.
        form_data: Form metadata (topic, researcher, etc.).
        sources: List of source dicts.
        verification_report: Verification results.
        template: One of 'ieee', 'acm', 'springer', 'elsevier', 'plain'.
        max_sources: Maximum sources to include in references.

    Returns:
        Tuple of (latex_string, bibtex_string).
    """
    # Import here to avoid circular dependency
    from streamlit_app import extract_cited_references_enhanced, renumber_citations_in_draft

    # Extract cited references and create renumbering map
    cited, contexts = extract_cited_references_enhanced(refined_draft)
    cited_refs_sorted = sorted(cited)

    old_to_new = {}
    for new_num, old_num in enumerate(cited_refs_sorted, 1):
        old_to_new[old_num] = new_num

    renumbered_draft = renumber_citations_in_draft(refined_draft, old_to_new)

    # Generate BibTeX and get cite key mapping
    bibtex_str, cite_key_map = generate_bibtex(sources, cited, old_to_new)

    # Build the LaTeX document
    preamble = get_latex_preamble(template, form_data)
    bib_style = get_bib_style(template)

    # Determine bib filename (without extension)
    topic_clean = re.sub(r'[^a-zA-Z0-9]', '_', form_data.get('topic', 'report')).strip('_')
    bib_filename = f"{topic_clean}_refs"

    # Start document
    parts = [preamble]

    # Handle Elsevier frontmatter closing
    if template == 'elsevier':
        # Abstract inside frontmatter for Elsevier
        abstract_text = format_section_content(renumbered_draft.get('abstract', ''))
        abstract_text = convert_citations_to_latex(abstract_text, cite_key_map)
        parts.append(rf"""
\begin{{abstract}}
{abstract_text}
\end{{abstract}}
\end{{frontmatter}}
""")
    else:
        parts.append(r"\begin{document}")
        parts.append(r"\maketitle")
        # Abstract
        abstract_text = format_section_content(renumbered_draft.get('abstract', ''))
        abstract_text = convert_citations_to_latex(abstract_text, cite_key_map)
        parts.append(rf"""
\begin{{abstract}}
{abstract_text}
\end{{abstract}}
""")

    # Executive Summary
    exec_summary = format_section_content(renumbered_draft.get('executiveSummary', ''))
    exec_summary = convert_citations_to_latex(exec_summary, cite_key_map)
    if exec_summary:
        parts.append(rf"""
\section{{Executive Summary}}
{exec_summary}
""")

    # Introduction
    intro = format_section_content(renumbered_draft.get('introduction', ''))
    intro = convert_citations_to_latex(intro, cite_key_map)
    if intro:
        parts.append(rf"""
\section{{Introduction}}
{intro}
""")

    # Literature Review
    lit_review = format_section_content(renumbered_draft.get('literatureReview', ''))
    lit_review = convert_citations_to_latex(lit_review, cite_key_map)
    if lit_review:
        parts.append(rf"""
\section{{Literature Review}}
{lit_review}
""")

    # Main Sections (as subsections)
    for section in renumbered_draft.get('mainSections', []):
        title = escape_latex(section.get('title', 'Section'))
        content = format_section_content(section.get('content', ''))
        content = convert_citations_to_latex(content, cite_key_map)
        if content:
            parts.append(rf"""
\subsection{{{title}}}
{content}
""")

    # Data & Analysis
    data_analysis = format_section_content(renumbered_draft.get('dataAnalysis', ''))
    data_analysis = convert_citations_to_latex(data_analysis, cite_key_map)
    if data_analysis:
        parts.append(rf"""
\section{{Data and Analysis}}
{data_analysis}
""")

    # Challenges
    challenges = format_section_content(renumbered_draft.get('challenges', ''))
    challenges = convert_citations_to_latex(challenges, cite_key_map)
    if challenges:
        parts.append(rf"""
\section{{Challenges}}
{challenges}
""")

    # Future Outlook
    future = format_section_content(renumbered_draft.get('futureOutlook', ''))
    future = convert_citations_to_latex(future, cite_key_map)
    if future:
        parts.append(rf"""
\section{{Future Outlook}}
{future}
""")

    # Conclusion
    conclusion = format_section_content(renumbered_draft.get('conclusion', ''))
    conclusion = convert_citations_to_latex(conclusion, cite_key_map)
    if conclusion:
        parts.append(rf"""
\section{{Conclusion}}
{conclusion}
""")

    # Bibliography
    parts.append(rf"""
\bibliographystyle{{{bib_style}}}
\bibliography{{{bib_filename}}}

\end{{document}}
""")

    latex_str = '\n'.join(parts)

    return latex_str, bibtex_str
