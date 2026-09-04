import os
import time

import requests
import streamlit as st


st.set_page_config(
    page_title="GitHub Search Engine",
    page_icon="🔎",
    layout="wide",
)


API_URL = os.getenv(
    "FASTAPI_URL",
    "http://localhost:8000/api"
)


st.title("GitHub Search Engine")

st.caption(
    "Search GitHub repositories using natural language."
)


query = st.text_input(
    "Search",
    placeholder="e.g. fast async web framework in python",
)


if query:

    start_time = time.time()

    with st.spinner("Searching..."):
        try:
            response = requests.get(
                f"{API_URL}/search",
                params={
                    "q": query,
                    "mode": "hybrid",
                    "min_stars": 0,
                    "sort_by": "relevance",
                    "limit": 15,
                    "offset": 0,
                },
                timeout=15,
            )

            elapsed_ms = (
                time.time() - start_time
            ) * 1000

            if response.status_code != 200:
                st.error(
                    f"API error {response.status_code}: "
                    f"{response.text}"
                )
                st.stop()

            data = response.json()

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the search backend."
            )
            st.stop()

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
            st.stop()

        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()


    total = data.get("total", 0)
    results = data.get("results", [])


    st.caption(
        f"{total:,} repositories matched · "
        f"{elapsed_ms:.0f} ms"
    )


    if not results:
        st.info("No repositories found.")
        st.stop()


    for item in results:

        repo = item["repo"]

        name = repo["name_with_owner"]
        description = (
            repo.get("description")
            or "No description."
        )

        language = (
            repo.get("primary_language")
            or "Unknown"
        )

        stars = repo.get("stars", 0)
        forks = repo.get("forks", 0)
        license_name = (
            repo.get("license")
            or "No license"
        )

        score = item["score"]

        url = (
            f"https://github.com/{name}"
        )


        with st.container(border=True):

            title_col, score_col = st.columns(
                [5, 1]
            )

            with title_col:
                st.subheader(name)

            with score_col:
                st.metric(
                    "Score",
                    f"{score:.3f}",
                )


            st.write(description)


            st.caption(
                f"{language} · "
                f"{stars:,} stars · "
                f"{forks:,} forks · "
                f"{license_name}"
            )


            st.link_button(
                "Open Repository",
                url,
            )


            with st.expander("Details"):

                dense = item.get(
                    "dense_score",
                    0,
                )

                lexical = item.get(
                    "lexical_score",
                    0,
                )

                metadata = item.get(
                    "metadata_score",
                    0,
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Semantic",
                    f"{dense:.3f}",
                )

                col2.metric(
                    "Lexical",
                    f"{lexical:.3f}",
                )

                col3.metric(
                    "Metadata",
                    f"{metadata:.3f}",
                )


                topics = repo.get("topics") or []

                if topics:
                    st.write(
                        "Topics:",
                        ", ".join(topics),
                    )