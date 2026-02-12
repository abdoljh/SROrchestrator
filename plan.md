# SROrch v2.2 Implementation Plan
## Feature 1: LaTeX Export for Academic Submissions
## Feature 2: Advanced Filtering (Impact Factor, H-index)

---

## Feature 1: LaTeX Export for Academic Submissions

### Overview
Add a LaTeX/BibTeX export pipeline that converts the existing verified HTML report into a publication-ready `.tex` file with a companion `.bib` file. Users will be able to select from common academic templates (IEEE, ACM, Springer, Elsevier, Plain) directly from the Reviewer tab.

### New File: `latex_export.py`

A self-contained module (~350 lines) with the following functions:

1. **`generate_latex_report(refined_draft, form_data, sources, verification_report, template, max_sources)`**
   - Main entry point. Accepts the same data as `generate_html_report_strict()`.
   - Returns a tuple: `(latex_string, bibtex_string)`.
   - Calls the template-specific preamble generator, then assembles sections.

2. **`get_latex_preamble(template, form_data)`**
   - Returns the `\documentclass`, `\usepackage`, `\title`, `\author`, `\affiliation`, `\date` block for the chosen template.
   - Templates: `ieee` (IEEEtran), `acm` (acmart), `springer` (llncs/svjour3), `elsevier` (elsarticle), `plain` (article class).

3. **`escape_latex(text)`**
   - Escapes special LaTeX characters: `& % $ # _ { } ~ ^ \`.
   - Preserves intentional math-mode `$...$` if present.

4. **`html_citations_to_latex(text)`**
   - Converts `[1]`, `[2]` citation markers to `\cite{key1}`, `\cite{key2}`.
   - Uses cite keys generated from author+year+index.

5. **`generate_bibtex(sources, cited_refs, old_to_new)`**
   - Reuses logic from `master_orchestrator.export_bibtex()` but returns a string.
   - Generates `@article`, `@inproceedings`, `@misc` entries with proper escaping.
   - Returns BibTeX string and a mapping of ref_num -> cite_key.

6. **`format_section_content(text)`**
   - Converts basic HTML tags (`<i>`, `<b>`, `<p>`) to LaTeX equivalents.
   - Handles paragraphs, emphasis, bold, and lists.

### Changes to `streamlit_app.py`

1. **Import** (near line 38):
   ```python
   from latex_export import generate_latex_report
   ```

2. **Reviewer Tab - Input form** (near line 3070, after citation style selector):
   - Add a new selectbox: "Export Format" with options `["HTML", "LaTeX (IEEE)", "LaTeX (ACM)", "LaTeX (Springer)", "LaTeX (Elsevier)", "LaTeX (Plain)"]`.
   - Store in `form_data['export_format']`.

3. **Pipeline execution** (`execute_report_pipeline`, near line 2384):
   - After HTML generation, if `export_format` starts with "LaTeX":
     - Call `generate_latex_report()` with the appropriate template.
     - Store result in `st.session_state.report_latex` and `st.session_state.report_bibtex`.

4. **Download buttons** (Tab 3 complete step, near line 3246):
   - If LaTeX was generated, add two download buttons:
     - "Download LaTeX (.tex)"
     - "Download BibTeX (.bib)"
   - Add a "Download LaTeX Bundle (.zip)" button containing both files.
   - Keep existing HTML download button available regardless.

### Template Details

Each template produces a complete, compilable `.tex` file:

- **IEEE**: `\documentclass[conference]{IEEEtran}` with `\IEEEauthorblockN`, proper column format.
- **ACM**: `\documentclass[sigconf]{acmart}` with ACM copyright/DOI blocks.
- **Springer**: `\documentclass{llncs}` with LNCS formatting.
- **Elsevier**: `\documentclass[preprint,12pt]{elsarticle}` with journal metadata.
- **Plain**: `\documentclass[12pt]{article}` with standard geometry package.

All templates include: `\usepackage[utf8]{inputenc}`, `\usepackage{hyperref}`, `\usepackage{cite}`, `\bibliographystyle{...}`.

---

## Feature 2: Advanced Filtering (Impact Factor, H-index)

### Overview
Enrich the paper data model with journal-level impact factor and author-level h-index metrics, then expose them as filter controls in the Interactive Data Explorer and as ranking factors in the scoring algorithm.

### Phase 2a: Data Enrichment (Engine Parsers)

**Changes to `openalex_utils.py`:**
- OpenAlex provides `cited_by_count` at the **source (journal) level** and author `h_index` via the `/authors` endpoint.
- In `fetch_and_process_openalex()`:
  - Add `"primary_location.source.summary_stats"` to the `select` fields parameter.
  - Extract `source.summary_stats.2yr_mean_citedness` as a proxy for impact factor (OpenAlex's "2-year mean citedness" ≈ JIF).
  - Extract first author's `h_index` from `authorships[0].author.summary_stats.h_index` (available when `select` includes `authorships.author`).
  - Add `impact_factor` and `h_index` fields to each paper dict.

**Changes to `s2_utils.py`:**
- Semantic Scholar provides `citationCount` per paper and author `hIndex` via author details.
- In `fetch_and_process_papers()`:
  - Add `"externalIds,influentialCitationCount"` to the fields parameter.
  - For each paper, store `influential_citations` count.
  - S2 doesn't provide journal IF directly, so set `impact_factor` = None (will be filled by OpenAlex or Scopus if available).

**Changes to `scopus_utils.py`:**
- Scopus provides `SJR` (Scimago Journal Rank) which correlates with impact factor.
- In `fetch_and_process_scopus()`:
  - Extract `prism:sjr` or `source-id` from the response if available in STANDARD view.
  - Add `sjr` field to each paper dict.

**Changes to `doi_utils.py`:**
- Crossref provides `is-referenced-by-count` and journal ISSN.
- Enhance to extract `references-count` and store as additional metadata.

### Phase 2b: Data Model Update

**Changes to `master_orchestrator.py`:**

1. **`deduplicate_and_score()`** (line 97):
   - When merging duplicates, take the best (non-null) `impact_factor` and `h_index` from any source.
   - Add these fields to the scoring formula:
     ```python
     # New scoring components
     if_boost = min(paper.get('impact_factor', 0) * 2, 50)  # Cap at 50
     h_boost = min(paper.get('h_index', 0) * 0.5, 25)       # Cap at 25
     base_score += if_boost + h_boost
     ```

2. **`save_master_csv()`** (line 536):
   - Add `impact_factor` and `h_index` to the CSV `keys` list (line 596).

### Phase 2c: UI Filter Controls

**Changes to `interactive_data_viewer.py`:**

1. **Quick stats row** (line 32):
   - Add a 5th metric: "Avg Impact Factor" showing the mean IF across papers with known values.

2. **Filter controls** (inside `with st.expander("Filter Data")`, line 46):
   - In `filter_col1`: Add a slider "Minimum Impact Factor" (range 0.0 to max IF, step 0.5).
   - In `filter_col2`: Add a slider "Minimum H-index" (range 0 to max h-index, step 1).

3. **Apply filters** (line 126):
   - Add filter application for `impact_factor >= min_if` and `h_index >= min_h`.

4. **Column config** (line 248):
   - Add column config entries:
     ```python
     "impact_factor": st.column_config.NumberColumn("IF", format="%.1f"),
     "h_index": st.column_config.NumberColumn("H-index", format="%d"),
     ```

5. **Default columns** (line 170):
   - Add `impact_factor` and `h_index` to `default_columns`.

6. **Quick filters** (line 303):
   - Add "High Impact (IF > 5)" button.
   - Add "Prolific Authors (h > 20)" button.

7. **Detailed paper view** (line 336):
   - Display IF and h-index in the metrics column.

### Phase 2d: Streamlit App Integration

**Changes to `streamlit_app.py`:**

1. **Results Tab (Tab 2)** - Analytics section:
   - Add IF/h-index distribution charts (optional, in analytics expander).

2. **`convert_orchestrator_to_source_format()`** (line 1197):
   - Preserve `impact_factor` and `h_index` in the `_orchestrator_data` dict.

3. **Report generation** (in `generate_draft_strict`):
   - Include IF and h-index in source boundary prompt metadata when available.

### Phase 2e: Dependencies

No new dependencies required. All data comes from existing API responses - we're just extracting fields that are already returned but not currently captured.

---

## Implementation Order

| Step | Feature | File(s) | Description |
|------|---------|---------|-------------|
| 1 | LaTeX | `latex_export.py` (new) | Create LaTeX export module |
| 2 | LaTeX | `streamlit_app.py` | Add import, UI controls, download buttons |
| 3 | Filtering | `openalex_utils.py` | Extract IF and h-index from OpenAlex |
| 4 | Filtering | `s2_utils.py` | Extract influential citations |
| 5 | Filtering | `scopus_utils.py` | Extract SJR |
| 6 | Filtering | `master_orchestrator.py` | Update scoring and CSV export |
| 7 | Filtering | `interactive_data_viewer.py` | Add filter UI controls |
| 8 | Filtering | `streamlit_app.py` | Integrate IF/h-index into report pipeline |
| 9 | Both | `requirements.txt` | No new deps needed |
| 10 | Both | `README.md` | Update roadmap checkboxes |

---

## Testing Strategy

- LaTeX output verified by checking that generated `.tex` compiles with `pdflatex` (manual spot-check).
- BibTeX output verified by checking valid entry format.
- IF/h-index filters verified by running a search query and confirming non-null values appear in the data viewer.
- Scoring changes verified by comparing relevance scores before/after with known high-IF papers.
