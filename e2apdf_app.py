"""
E2A PDF Translator — Streamlit App
English to Arabic PDF translation with full RTL support.
"""

import io
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Make sure the e2apdf package is importable
sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="E2A PDF Translator",
    page_icon="📄",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Title & description
# ---------------------------------------------------------------------------
st.title("📄 E2A PDF Translator")
st.markdown(
    "**Translate English PDF files to Arabic with proper RTL layout, "
    "letter shaping, and BiDi reordering.**"
)
st.divider()

# ---------------------------------------------------------------------------
# Helper: lazy import with a friendly error message
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _check_deps() -> tuple[bool, str]:
    """Return (ok, error_message) for required dependencies."""
    missing = []
    for pkg, import_name in [
        ("pypdfium2", "pypdfium2"),
        ("pypdf", "pypdf"),
        ("reportlab", "reportlab"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    if missing:
        return False, f"Missing required packages: {', '.join(missing)}. Run: pip install {' '.join(missing)}"
    return True, ""


deps_ok, deps_err = _check_deps()
if not deps_ok:
    st.error(deps_err)
    st.stop()

from e2apdf.pipeline import E2APipeline, PipelineConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Sidebar — Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Translation Settings")

    backend = st.selectbox(
        "Translation backend",
        options=["mock", "llm-anthropic", "llm-openai", "google", "deepl"],
        index=0,
        help=(
            "**mock** – instant, no API key needed (for testing)\n\n"
            "**llm-anthropic** – Claude (best quality)\n\n"
            "**llm-openai** – GPT (excellent quality)\n\n"
            "**google** – Google Cloud Translation\n\n"
            "**deepl** – DeepL API"
        ),
    )

    api_key = ""
    if backend != "mock":
        env_var_map = {
            "llm-anthropic": "ANTHROPIC_API_KEY",
            "llm-openai": "OPENAI_API_KEY",
            "google": "GOOGLE_TRANSLATE_API_KEY",
            "deepl": "DEEPL_API_KEY",
        }
        env_var = env_var_map.get(backend, "")
        env_val = os.environ.get(env_var, "")

        api_key = st.text_input(
            f"API Key ({env_var})",
            value=env_val,
            type="password",
            help=f"Your API key, or set the `{env_var}` environment variable.",
        )

    model_name = ""
    if backend in ("llm-anthropic", "llm-openai"):
        default_model = (
            "claude-sonnet-4-6" if backend == "llm-anthropic" else "gpt-4o"
        )
        model_name = st.text_input(
            "Model name (optional)",
            value="",
            placeholder=default_model,
            help=f"Leave blank to use the default ({default_model}).",
        )

    st.divider()
    st.subheader("Layout Options")

    mirror_layout = st.checkbox(
        "Mirror layout for RTL",
        value=True,
        help="Mirror x-positions so right becomes left in the Arabic output.",
    )
    preserve_positions = st.checkbox(
        "Preserve text positions",
        value=True,
        help="Keep approximate original positions. Uncheck for flowing text.",
    )
    add_page_numbers = st.checkbox("Add Arabic page numbers", value=True)

    line_spacing = st.slider(
        "Line spacing",
        min_value=1.0,
        max_value=2.5,
        value=1.4,
        step=0.1,
    )
    margin = st.slider(
        "Page margin (pts)",
        min_value=20,
        max_value=100,
        value=50,
        step=5,
    )

    st.divider()
    st.subheader("Cache")
    use_cache = st.checkbox(
        "Enable translation cache",
        value=True,
        help="Cache translations to avoid redundant API calls across runs.",
    )

# ---------------------------------------------------------------------------
# Main area — file upload
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an English PDF",
    type=["pdf"],
    help="Select the PDF file you want to translate into Arabic.",
)

if uploaded_file is None:
    st.info("Upload a PDF file to get started.")
    st.stop()

st.success(f"File loaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

# ---------------------------------------------------------------------------
# Validate API key before running
# ---------------------------------------------------------------------------
if backend != "mock" and not api_key.strip():
    st.warning(
        f"Please enter an API key for the **{backend}** backend (sidebar)."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Translate button
# ---------------------------------------------------------------------------
if st.button("Translate to Arabic", type="primary", use_container_width=True):

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / uploaded_file.name
        output_path = Path(tmpdir) / (Path(uploaded_file.name).stem + "_ar.pdf")

        # Write uploaded file to temp dir
        input_path.write_bytes(uploaded_file.read())

        # Build config
        cache_path = (
            str(Path(tmpdir) / "translations_cache.json") if use_cache else None
        )

        config = PipelineConfig(
            translation_backend=backend,
            api_key=api_key.strip() or None,
            model=model_name.strip() or None,
            mirror_layout=mirror_layout,
            preserve_positions=preserve_positions,
            add_page_numbers=add_page_numbers,
            line_spacing=line_spacing,
            margin=float(margin),
            cache_path=cache_path,
            verbose=False,
        )

        pipeline = E2APipeline(config)

        # Run translation with progress feedback
        status_placeholder = st.empty()
        with st.spinner("Translating… this may take a moment."):
            try:
                report = pipeline.translate(input_path, output_path)
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
                st.stop()

        # Show report
        st.subheader("Translation Report")

        if report.success:
            st.success("Translation completed successfully!")
        else:
            st.error("Translation encountered errors. See details below.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Pages", f"{report.translated_pages}/{report.total_pages}")
        col2.metric("Blocks translated", report.translated_blocks)
        col3.metric("Duration", f"{report.duration_seconds:.1f}s")

        col4, col5 = st.columns(2)
        col4.metric("From cache", report.cached_blocks)
        col5.metric("Skipped", report.skipped_blocks)

        if report.warnings:
            with st.expander(f"Warnings ({len(report.warnings)})"):
                for w in report.warnings:
                    st.warning(w)

        if report.errors:
            with st.expander(f"Errors ({len(report.errors)})", expanded=True):
                for e in report.errors:
                    st.error(e)

        with st.expander("Full report text"):
            st.code(report.summary(), language=None)

        # Download button
        if report.success and output_path.exists():
            st.divider()
            output_bytes = output_path.read_bytes()
            download_name = Path(uploaded_file.name).stem + "_arabic.pdf"
            st.download_button(
                label="Download Arabic PDF",
                data=output_bytes,
                file_name=download_name,
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
        elif not output_path.exists():
            st.warning("Output file was not created. Check the errors above.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "E2A PDF Translator — powered by ReportLab, pypdfium2, and your chosen "
    "translation backend. Arabic reshaping via arabic-reshaper + python-bidi."
)
