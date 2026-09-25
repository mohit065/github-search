import os
import math
import time
import requests
import streamlit as st

st.set_page_config(
    page_title="GitHub Semantic Search Engine",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = os.getenv("FASTAPI_URL", "http://localhost:8000/api")

# Custom styling for rich UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #f1f5f9;
        color: #334155;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .topic-tag {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        font-size: 0.72rem;
        border-radius: 6px;
        background-color: #eff6ff;
        color: #2563eb;
        border: 1px solid #dbeafe;
        margin-right: 0.35rem;
        margin-top: 0.35rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to fetch languages for sidebar filter
@st.cache_data(ttl=60)
def fetch_languages():
    try:
        resp = requests.get(f"{API_URL}/languages", timeout=3)
        if resp.status_code == 200:
            return ["All"] + resp.json()
    except Exception:
        pass
    return ["All", "Python", "TypeScript", "JavaScript", "Go", "Rust", "C++", "Java", "C#", "Ruby", "PHP", "HTML", "Shell"]

# Helper function to fetch backend stats
@st.cache_data(ttl=30)
def fetch_stats():
    try:
        resp = requests.get(f"{API_URL}/stats", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

languages_list = fetch_languages()
stats_data = fetch_stats()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Search Settings")
    
    mode_label = st.selectbox(
        "Search Algorithm",
        options=["Hybrid (Semantic + Lexical + Activity)", "Semantic Only (Dense Vectors)", "Lexical Only (BM25 Keywords)"],
        index=0,
        help="Choose retrieval model: Hybrid combines sentence transformers with BM25 and activity signals."
    )
    mode_map = {
        "Hybrid (Semantic + Lexical + Activity)": "hybrid",
        "Semantic Only (Dense Vectors)": "semantic",
        "Lexical Only (BM25 Keywords)": "lexical"
    }
    selected_mode = mode_map[mode_label]
    
    selected_lang = st.selectbox(
        "Primary Language",
        options=languages_list,
        index=0
    )
    
    min_stars = st.number_input(
        "Minimum Stars ⭐",
        min_value=0,
        max_value=100000,
        value=0,
        step=50
    )
    
    sort_by_label = st.selectbox(
        "Sort Results By",
        options=["Relevance Score", "Most Stars ⭐", "Highest Activity ⚡", "Most Recently Updated 🕒"],
        index=0
    )
    sort_map = {
        "Relevance Score": "relevance",
        "Most Stars ⭐": "stars",
        "Highest Activity ⚡": "activity",
        "Most Recently Updated 🕒": "recency"
    }
    selected_sort = sort_map[sort_by_label]
    
    st.divider()
    st.subheader("📄 Pagination & Display")
    page_size = st.selectbox(
        "Results Per Page",
        options=[10, 15, 25, 50, 100, 200],
        index=1,
        help="Select how many repositories to display per page."
    )

    if stats_data:
        st.divider()
        st.caption(f"📊 Total Indexed: **{stats_data.get('indexed_in_search', 0):,}** repos")
        st.caption(f"⭐ Avg Stars: **{stats_data.get('avg_stars', 0):,.1f}**")
        st.caption(f"⚡ Avg Activity: **{stats_data.get('avg_activity_score', 0):.2f}**")

# Main UI Header
st.markdown('<div class="main-header">🔎 GitHub Semantic Search Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Discover open-source repositories using natural language concepts, neural embeddings, and BM25 hybrid ranking.</div>', unsafe_allow_html=True)

# Example Chips
example_queries = [
    "fast async web framework in python",
    "web automation testing framework for games and apps",
    "computer vision deep learning for object detection",
    "distributed kubernetes cluster monitoring and logging",
    "export csv and excel spreadsheet in browser"
]

st.markdown("**💡 Try example queries:**")
chip_cols = st.columns(len(example_queries))
clicked_query = None
for i, col in enumerate(chip_cols):
    if col.button(example_queries[i], key=f"example_{i}", use_container_width=True):
        clicked_query = example_queries[i]

# State management for search input and pagination
if "current_query" not in st.session_state:
    st.session_state.current_query = ""
if "page" not in st.session_state:
    st.session_state.page = 1
if "filter_hash" not in st.session_state:
    st.session_state.filter_hash = ""

if clicked_query:
    st.session_state.current_query = clicked_query
    st.session_state.page = 1

query_input = st.text_input(
    "Search Repositories",
    value=st.session_state.current_query,
    placeholder="e.g. fast lightweight async web framework for microservices",
    label_visibility="collapsed"
)

# Detect change in filters or query to reset page to 1
current_filters = f"{query_input}|{selected_mode}|{selected_lang}|{min_stars}|{selected_sort}|{page_size}"
if current_filters != st.session_state.filter_hash:
    st.session_state.filter_hash = current_filters
    st.session_state.page = 1

# Show warming-up banner if backend is still indexing
try:
    _h = requests.get(f"{API_BASE}/health", timeout=3)
    if _h.status_code == 200:
        _hdata = _h.json()
        _indexed = _hdata.get("total_indexed", 0)
        if _hdata.get("status") == "warming_up":
            st.info(f"Search index is building in the background ? {_indexed:,} repositories indexed so far. Queries will return results once indexing completes.")
except Exception:
    pass

if query_input:
    st.session_state.current_query = query_input
    current_page = st.session_state.page
    offset = (current_page - 1) * page_size
    
    start_time = time.time()
    with st.spinner("Executing hybrid retrieval & ranking..."):
        try:
            params = {
                "q": query_input,
                "mode": selected_mode,
                "min_stars": min_stars,
                "sort_by": selected_sort,
                "limit": page_size,
                "offset": offset,
            }
            if selected_lang and selected_lang != "All":
                params["language"] = selected_lang

            response = requests.get(
                f"{API_URL}/search",
                params=params,
                timeout=20,
            )

            elapsed_ms = (time.time() - start_time) * 1000

            if response.status_code != 200:
                st.error(f"API error {response.status_code}: {response.text}")
                st.stop()

            data = response.json()

        except requests.exceptions.ConnectionError:
            st.error("⚠️ Could not connect to the search backend. Ensure FastAPI backend is running.")
            st.stop()
        except Exception as e:
            st.error(f"⚠️ Search failed: {e}")
            st.stop()

    total = data.get("total", 0)
    results = data.get("results", [])
    total_pages = max(1, math.ceil(total / page_size)) if total > 0 else 1

    # Status & Pagination bar
    col_status, col_pages_nav = st.columns([3, 3])
    with col_status:
        start_idx = min(offset + 1, total) if total > 0 else 0
        end_idx = min(offset + page_size, total)
        st.markdown(
            f"Showing **{start_idx:,} – {end_idx:,}** of **{total:,}** matched repositories &nbsp;·&nbsp; ⏱️ `{elapsed_ms:.1f} ms`"
        )

    with col_pages_nav:
        if total_pages > 1:
            p_col1, p_col2, p_col3, p_col4 = st.columns([1, 1.2, 1.2, 1])
            if p_col1.button("◀ Prev", disabled=(current_page <= 1), key="prev_top"):
                st.session_state.page = max(1, current_page - 1)
                st.rerun()
            
            p_col2.write(f"Page **{current_page}** of **{total_pages}**")
            
            # Direct page jumper
            new_page = p_col3.number_input("Go to page", min_value=1, max_value=total_pages, value=current_page, key="page_num_top", label_visibility="collapsed")
            if new_page != current_page:
                st.session_state.page = new_page
                st.rerun()
                
            if p_col4.button("Next ▶", disabled=(current_page >= total_pages), key="next_top"):
                st.session_state.page = min(total_pages, current_page + 1)
                st.rerun()

    if not results:
        st.info("No repositories found matching your query and filter criteria.")
        st.stop()

    st.markdown("---")

    # Render repository cards
    for idx, item in enumerate(results):
        repo = item["repo"]
        name = repo["name_with_owner"]
        description = repo.get("description") or "No description provided."
        language = repo.get("primary_language") or "Unknown"
        stars = repo.get("stars", 0)
        forks = repo.get("forks", 0)
        license_name = repo.get("license") or "No license"
        activity_score = repo.get("activity_score", 0.0)
        score = item["score"]
        dense = item.get("dense_score", 0.0)
        lexical = item.get("lexical_score", 0.0)
        metadata_score = item.get("metadata_score", 0.0)
        url = f"https://github.com/{name}"
        topics = repo.get("topics") or []

        with st.container(border=True):
            header_col, score_col = st.columns([5, 1.5])
            with header_col:
                st.subheader(f"[{name}]({url})")
            with score_col:
                st.metric("Relevance Score", f"{score:.3f}")

            st.write(description)

            # Metadata tags & badges
            meta_cols = st.columns([6, 1.5])
            with meta_cols[0]:
                tags_md = f"📌 `{language}` &nbsp; | &nbsp; ⭐ **{stars:,}** &nbsp; | &nbsp; 🍴 **{forks:,}** &nbsp; | &nbsp; 📜 {license_name} &nbsp; | &nbsp; ⚡ Activity: **{activity_score:.2f}**"
                st.markdown(tags_md)
                
                if topics:
                    topics_html = " ".join([f'<span class="topic-tag">#{t}</span>' for t in topics[:10]])
                    st.markdown(topics_html, unsafe_allow_html=True)
            
            with meta_cols[1]:
                st.link_button("View on GitHub ↗", url, use_container_width=True)

            with st.expander("🔍 Multi-Factor Score Breakdown"):
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Semantic Similarity", f"{dense:.3f}", help="Sentence Transformer dense embedding cosine similarity")
                sc2.metric("BM25 Lexical Match", f"{lexical:.3f}", help="BM25 Okapi exact keyword frequency & IDF score")
                sc3.metric("Metadata & Activity", f"{metadata_score:.3f}", help="Log-scaled stars, commit activity & recency boost")

    # Bottom Pagination bar
    if total_pages > 1:
        st.markdown("---")
        b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns([1, 1, 2, 1, 1])
        if b_col1.button("◀ Previous Page", disabled=(current_page <= 1), key="prev_bottom"):
            st.session_state.page = max(1, current_page - 1)
            st.rerun()
            
        b_col3.write(f"Showing Page **{current_page}** of **{total_pages}** ({total:,} repositories total)")
        
        if b_col5.button("Next Page ▶", disabled=(current_page >= total_pages), key="next_bottom"):
            st.session_state.page = min(total_pages, current_page + 1)
            st.rerun()
