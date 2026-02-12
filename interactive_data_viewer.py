"""
Interactive Data Viewer Component for SROrch
Provides rich manipulation capabilities for MASTER_REPORT_FINAL.csv
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime

def render_interactive_data_viewer(output_dir):
    """
    Render an interactive data viewer with filtering, sorting, and export capabilities

    Args:
        output_dir: Directory containing MASTER_REPORT_FINAL.csv
    """
    csv_path = os.path.join(output_dir, "MASTER_REPORT_FINAL.csv")

    if not os.path.exists(csv_path):
        st.warning("📊 CSV data not available yet. Complete a search first!")
        return

    # Load the data
    try:
        df = pd.read_csv(csv_path)

        # Data overview header
        st.subheader("📊 Interactive Data Explorer")

        # Quick stats
        has_if = 'impact_factor' in df.columns and df['impact_factor'].notna().any()
        has_h = 'h_index' in df.columns and df['h_index'].notna().any()

        stat_cols = st.columns(5 if (has_if or has_h) else 4)
        with stat_cols[0]:
            st.metric("Total Papers", len(df))
        with stat_cols[1]:
            st.metric("Unique Authors", df['ieee_authors'].nunique() if 'ieee_authors' in df.columns else "N/A")
        with stat_cols[2]:
            st.metric("Total Citations", int(df['citations'].sum()) if 'citations' in df.columns else "N/A")
        with stat_cols[3]:
            avg_year = df['year'].astype(str).str.extract(r'(\d{4})')[0].astype(float).mean()
            st.metric("Avg Year", f"{avg_year:.0f}" if not pd.isna(avg_year) else "N/A")
        if has_if or has_h:
            with stat_cols[4]:
                if has_if:
                    avg_if = df['impact_factor'].dropna().mean()
                    st.metric("Avg Impact Factor", f"{avg_if:.1f}" if not pd.isna(avg_if) else "N/A")
                elif has_h:
                    avg_h = df['h_index'].dropna().mean()
                    st.metric("Avg H-index", f"{avg_h:.0f}" if not pd.isna(avg_h) else "N/A")

        st.divider()

        # === FILTERING CONTROLS ===
        with st.expander("🔍 Filter Data", expanded=True):
            filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

            with filter_col1:
                # Year filter
                if 'year' in df.columns:
                    years = df['year'].astype(str).str.extract(r'(\d{4})')[0].dropna().astype(int).unique()
                    years = sorted([y for y in years if 1900 <= y <= 2100])

                    if len(years) > 0:
                        year_range = st.slider(
                            "Publication Year Range",
                            min_value=int(min(years)),
                            max_value=int(max(years)),
                            value=(int(min(years)), int(max(years))),
                            key="year_filter"
                        )

                # Citation filter
                if 'citations' in df.columns:
                    max_cites = int(df['citations'].max())
                    if max_cites > 0:
                        min_citations = st.number_input(
                            "Minimum Citations",
                            min_value=0,
                            max_value=max_cites,
                            value=0,
                            key="citation_filter"
                        )

            with filter_col2:
                # Source count filter
                if 'source_count' in df.columns:
                    max_sources = int(df['source_count'].max())
                    min_sources = st.slider(
                        "Minimum Source Count",
                        min_value=1,
                        max_value=max_sources,
                        value=1,
                        key="source_filter",
                        help="Papers found in N or more databases"
                    )

                # Relevance score filter
                if 'relevance_score' in df.columns:
                    max_score = int(df['relevance_score'].max())
                    min_score = st.number_input(
                        "Minimum Relevance Score",
                        min_value=0,
                        max_value=max_score,
                        value=0,
                        key="score_filter"
                    )

            with filter_col3:
                # Impact Factor filter
                if has_if:
                    max_if_val = float(df['impact_factor'].dropna().max())
                    min_if = st.slider(
                        "Minimum Impact Factor",
                        min_value=0.0,
                        max_value=max_if_val,
                        value=0.0,
                        step=0.5,
                        key="if_filter",
                        help="Journal Impact Factor (2-year mean citedness)"
                    )

                # H-index filter
                if has_h:
                    max_h_val = int(df['h_index'].dropna().max())
                    min_h = st.slider(
                        "Minimum H-index",
                        min_value=0,
                        max_value=max_h_val,
                        value=0,
                        step=1,
                        key="h_filter",
                        help="First author's h-index"
                    )

            with filter_col4:
                # Text search
                search_term = st.text_input(
                    "Search in Title/Authors",
                    placeholder="Enter keywords...",
                    key="text_search"
                )

                # Venue filter
                if 'venue' in df.columns:
                    venues = df['venue'].dropna().unique().tolist()
                    if len(venues) > 0:
                        selected_venues = st.multiselect(
                            "Filter by Venue",
                            options=sorted(venues)[:50],  # Limit to top 50
                            key="venue_filter"
                        )

                # Recency boost filter
                if 'recency_boosted' in df.columns:
                    show_boosted_only = st.checkbox(
                        "Show Recent Papers Only",
                        value=False,
                        key="boosted_filter"
                    )

        # === APPLY FILTERS ===
        filtered_df = df.copy()

        # Year filter
        if 'year' in df.columns and 'year_range' in locals():
            year_mask = filtered_df['year'].astype(str).str.extract(r'(\d{4})')[0].astype(float).between(year_range[0], year_range[1])
            filtered_df = filtered_df[year_mask]

        # Citation filter
        if 'citations' in df.columns and 'min_citations' in locals():
            filtered_df = filtered_df[filtered_df['citations'] >= min_citations]

        # Source count filter
        if 'source_count' in df.columns and 'min_sources' in locals():
            filtered_df = filtered_df[filtered_df['source_count'] >= min_sources]

        # Relevance score filter
        if 'relevance_score' in df.columns and 'min_score' in locals():
            filtered_df = filtered_df[filtered_df['relevance_score'] >= min_score]

        # Impact Factor filter
        if has_if and 'min_if' in locals() and min_if > 0:
            filtered_df = filtered_df[filtered_df['impact_factor'].fillna(0) >= min_if]

        # H-index filter
        if has_h and 'min_h' in locals() and min_h > 0:
            filtered_df = filtered_df[filtered_df['h_index'].fillna(0) >= min_h]

        # Text search filter
        if search_term:
            mask = (
                filtered_df['title'].str.contains(search_term, case=False, na=False) |
                filtered_df['ieee_authors'].str.contains(search_term, case=False, na=False)
            )
            filtered_df = filtered_df[mask]

        # Venue filter
        if 'selected_venues' in locals() and len(selected_venues) > 0:
            filtered_df = filtered_df[filtered_df['venue'].isin(selected_venues)]

        # Recency boost filter
        if 'show_boosted_only' in locals() and show_boosted_only:
            filtered_df = filtered_df[filtered_df['recency_boosted'] == True]

        # === DISPLAY CONTROLS ===
        st.markdown(f"**Showing {len(filtered_df)} of {len(df)} papers**")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Column selector
            all_columns = df.columns.tolist()
            default_columns = ['relevance_score', 'source_count', 'ieee_authors', 'title', 'venue', 'year', 'citations']
            # Add IF and h-index to defaults if available
            if has_if:
                default_columns.append('impact_factor')
            if has_h:
                default_columns.append('h_index')
            default_columns = [col for col in default_columns if col in all_columns]

            selected_columns = st.multiselect(
                "Select Columns to Display",
                options=all_columns,
                default=default_columns,
                key="column_selector"
            )

        with col2:
            # Sort options
            if selected_columns:
                sort_column = st.selectbox(
                    "Sort by",
                    options=selected_columns,
                    index=0 if 'relevance_score' in selected_columns else 0,
                    key="sort_column"
                )

                sort_order = st.radio(
                    "Order",
                    options=['Descending', 'Ascending'],
                    horizontal=True,
                    key="sort_order"
                )

        with col3:
            # Display options
            show_abstract = st.checkbox("Show Abstracts", value=False, key="show_abstract")
            show_tldr = st.checkbox("Show TLDRs", value=False, key="show_tldr")

            rows_per_page = st.selectbox(
                "Rows per page",
                options=[10, 25, 50, 100, 'All'],
                index=1,
                key="rows_per_page"
            )

        # === SORT DATA ===
        if selected_columns and 'sort_column' in locals():
            filtered_df = filtered_df.sort_values(
                by=sort_column,
                ascending=(sort_order == 'Ascending')
            )

        # === PREPARE DISPLAY DATAFRAME ===
        display_df = filtered_df[selected_columns].copy() if selected_columns else filtered_df.copy()

        # Add abstract/TLDR if requested
        if show_abstract and 'abstract' in df.columns and 'abstract' not in selected_columns:
            display_df['abstract'] = filtered_df['abstract']

        if show_tldr and 'tldr' in df.columns and 'tldr' not in selected_columns:
            display_df['tldr'] = filtered_df['tldr']

        # === PAGINATION ===
        if rows_per_page != 'All':
            total_pages = (len(display_df) - 1) // rows_per_page + 1
            page = st.number_input(
                f"Page (1-{total_pages})",
                min_value=1,
                max_value=max(1, total_pages),
                value=1,
                key="page_number"
            )
            start_idx = (page - 1) * rows_per_page
            end_idx = start_idx + rows_per_page
            display_df = display_df.iloc[start_idx:end_idx]

        st.divider()

        # === DISPLAY THE DATA ===
        column_config = {
            "url": st.column_config.LinkColumn("URL"),
            "doi": st.column_config.TextColumn("DOI", width="medium"),
            "relevance_score": st.column_config.NumberColumn("Score", format="%d"),
            "citations": st.column_config.NumberColumn("Citations", format="%d"),
            "source_count": st.column_config.NumberColumn("Sources", format="%d"),
            "year": st.column_config.TextColumn("Year", width="small"),
            "abstract": st.column_config.TextColumn("Abstract", width="large"),
            "tldr": st.column_config.TextColumn("TLDR", width="large"),
        }
        if has_if:
            column_config["impact_factor"] = st.column_config.NumberColumn("IF", format="%.1f")
        if has_h:
            column_config["h_index"] = st.column_config.NumberColumn("H-index", format="%d")
        if 'influential_citations' in df.columns:
            column_config["influential_citations"] = st.column_config.NumberColumn("Influential", format="%d")

        st.dataframe(
            display_df,
            use_container_width=True,
            height=500,
            hide_index=False,
            column_config=column_config
        )

        st.divider()

        # === BULK OPERATIONS ===
        st.subheader("🔧 Bulk Operations")

        op_col1, op_col2, op_col3 = st.columns(3)

        with op_col1:
            # Export filtered data
            if st.button("📥 Export Filtered Data (CSV)", use_container_width=True):
                csv_data = filtered_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Filtered CSV",
                    data=csv_data,
                    file_name=f"filtered_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with op_col2:
            # Export as JSON
            if st.button("📥 Export as JSON", use_container_width=True):
                json_data = filtered_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="⬇️ Download JSON",
                    data=json_data,
                    file_name=f"filtered_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )

        with op_col3:
            # Copy to clipboard (display selection)
            if st.button("📋 Show Selection Stats", use_container_width=True):
                stats_text = f"""
                **Selection Statistics:**
                - Papers: {len(filtered_df)}
                - Total Citations: {int(filtered_df['citations'].sum() if 'citations' in filtered_df.columns else 0)}
                - Avg Citations: {filtered_df['citations'].mean():.1f if 'citations' in filtered_df.columns else 'N/A'}
                - High Consensus (≥4 sources): {len(filtered_df[filtered_df['source_count'] >= 4]) if 'source_count' in filtered_df.columns else 'N/A'}
                """
                if has_if:
                    avg_if_sel = filtered_df['impact_factor'].dropna().mean()
                    stats_text += f"\n                - Avg Impact Factor: {avg_if_sel:.1f}" if not pd.isna(avg_if_sel) else ""
                if has_h:
                    avg_h_sel = filtered_df['h_index'].dropna().mean()
                    stats_text += f"\n                - Avg H-index: {avg_h_sel:.0f}" if not pd.isna(avg_h_sel) else ""
                st.info(stats_text)

        # === QUICK ACTIONS ===
        with st.expander("⚡ Quick Filters (One-Click)"):
            quick_col1, quick_col2, quick_col3 = st.columns(3)

            with quick_col1:
                if st.button("🔥 Highly Cited (>100)", use_container_width=True):
                    st.session_state['citation_filter'] = 100
                    st.rerun()

                if st.button("🌟 High Consensus (≥4)", use_container_width=True):
                    st.session_state['source_filter'] = 4
                    st.rerun()

            with quick_col2:
                if st.button("📅 Last 3 Years", use_container_width=True):
                    current_year = datetime.now().year
                    st.session_state['year_filter'] = (current_year - 3, current_year)
                    st.rerun()

                if st.button("🔝 Top 10 by Score", use_container_width=True):
                    st.session_state['rows_per_page'] = 10
                    st.session_state['sort_column'] = 'relevance_score'
                    st.session_state['sort_order'] = 'Descending'
                    st.rerun()

            with quick_col3:
                if has_if:
                    if st.button("📈 High Impact (IF > 5)", use_container_width=True):
                        st.session_state['if_filter'] = 5.0
                        st.rerun()

                if has_h:
                    if st.button("🎓 Prolific Authors (h > 20)", use_container_width=True):
                        st.session_state['h_filter'] = 20
                        st.rerun()

                if st.button("🔄 Reset All Filters", use_container_width=True):
                    # Clear all filter-related session state
                    for key in list(st.session_state.keys()):
                        if 'filter' in key or key in ['sort_column', 'sort_order', 'page_number', 'text_search']:
                            del st.session_state[key]
                    st.rerun()

        # === DETAILED VIEW ===
        with st.expander("🔍 Detailed Paper View"):
            if len(filtered_df) > 0:
                paper_index = st.selectbox(
                    "Select Paper to View",
                    options=range(len(filtered_df)),
                    format_func=lambda x: f"#{x+1}: {filtered_df.iloc[x]['title'][:60]}..."
                )

                paper = filtered_df.iloc[paper_index]

                # Display full paper details
                st.markdown(f"### {paper.get('title', 'N/A')}")

                detail_col1, detail_col2 = st.columns([2, 1])

                with detail_col1:
                    st.markdown(f"**Authors:** {paper.get('ieee_authors', 'N/A')}")
                    st.markdown(f"**Venue:** {paper.get('venue', 'N/A')}")
                    st.markdown(f"**Year:** {paper.get('year', 'N/A')}")

                    if paper.get('doi') and paper['doi'] != 'N/A':
                        st.markdown(f"**DOI:** {paper['doi']}")

                    if paper.get('url'):
                        st.markdown(f"**URL:** [{paper['url']}]({paper['url']})")

                    if paper.get('keywords'):
                        st.markdown(f"**Keywords:** {paper['keywords']}")

                    if paper.get('tldr'):
                        st.info(f"**TLDR:** {paper['tldr']}")

                    if paper.get('abstract'):
                        with st.expander("📄 Full Abstract"):
                            st.write(paper['abstract'])

                with detail_col2:
                    st.metric("Relevance Score", paper.get('relevance_score', 0))
                    st.metric("Citations", paper.get('citations', 0))
                    st.metric("Source Count", paper.get('source_count', 1))

                    # Show Impact Factor and H-index if available
                    if has_if and pd.notna(paper.get('impact_factor')):
                        st.metric("Impact Factor", f"{paper['impact_factor']:.1f}")
                    if has_h and pd.notna(paper.get('h_index')):
                        st.metric("H-index", int(paper['h_index']))
                    if 'influential_citations' in df.columns and pd.notna(paper.get('influential_citations')):
                        ic_val = paper['influential_citations']
                        if ic_val and int(ic_val) > 0:
                            st.metric("Influential Citations", int(ic_val))

                    if paper.get('recency_boosted'):
                        st.success("🔥 Recent Paper")

    except Exception as e:
        st.error(f"Error loading CSV data: {e}")
        with st.expander("🔍 Error Details"):
            import traceback
            st.code(traceback.format_exc())
