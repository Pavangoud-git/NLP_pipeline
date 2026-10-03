from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

from nlp_pipeline import get_pipeline


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="NLP Intelligence Studio",
    page_icon="🧠",
    layout="wide"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 17px;
        color: #6b7280;
        margin-top: 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="title">🧠 NLP Intelligence Studio</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Text Preprocessing • Entity Extraction • '
    'Sentiment Analysis • REST API'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("⚙️ Configuration")

    mode = st.radio(
        "Processing mode",
        [
            "Local Pipeline",
            "REST API"
        ]
    )

    api_url = st.text_input(
        "REST API URL",
        "http://127.0.0.1:8000"
    )

    st.caption(
        "NLP analysis results are displayed in "
        "the Streamlit UI."
    )


# ---------------------------------------------------------
# SAMPLE TEXT
# ---------------------------------------------------------

examples = {

    "E-commerce review":
        "I bought an Apple iPhone in Hyderabad "
        "last week. The camera is excellent, "
        "but the battery life is disappointing.",

    "Movie review":
        "The movie was inspiring and beautifully "
        "directed. The actors delivered a fantastic "
        "performance.",

    "Business text":
        "Microsoft announced a new AI partnership "
        "with OpenAI in New York on Monday."
}


selected = st.selectbox(
    "Example text",
    list(examples.keys())
)


text = st.text_area(
    "Enter text",
    value=examples[selected],
    height=170,
    placeholder="Type or paste English text here..."
)


# ---------------------------------------------------------
# ANALYZE BUTTON
# ---------------------------------------------------------

col1, col2 = st.columns([1, 5])

with col1:

    analyze_btn = st.button(
        "🚀 Analyze",
        type="primary",
        use_container_width=True
    )

with col2:

    st.caption(
        "Pipeline: Input → Cleaning → Tokenization → "
        "Stop-word Removal → Lemmatization → NER → Sentiment"
    )


# ---------------------------------------------------------
# RESULT FUNCTION
# ---------------------------------------------------------

def get_result() -> dict:

    # Local processing
    if mode == "Local Pipeline":

        return get_pipeline().analyze(
            text
        )

    # REST API processing
    response = requests.post(
        f"{api_url.rstrip('/')}/analyze",
        json={
            "text": text
        },
        timeout=180
    )

    response.raise_for_status()

    return response.json()


# ---------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------

if analyze_btn:

    if not text.strip():

        st.warning(
            "Please enter some text."
        )

    else:

        try:

            with st.spinner(
                "Running NLP pipeline..."
            ):

                result = get_result()


            sentiment = result["sentiment"]

            entities = result["entities"]


            # -------------------------------------------------
            # OVERVIEW
            # -------------------------------------------------

            st.divider()

            st.subheader(
                "📊 Analysis Overview"
            )


            m1, m2, m3, m4 = st.columns(4)


            m1.metric(
                "Original chars",
                len(result["original_text"])
            )


            m2.metric(
                "Cleaned tokens",
                len(result["tokens"])
            )


            m3.metric(
                "Entities",
                len(entities)
            )


            m4.metric(
                "Sentiment",
                sentiment["label"]
            )


            # -------------------------------------------------
            # TABS
            # -------------------------------------------------

            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                [
                    "🧹 Preprocessing",
                    "🏷️ Entities",
                    "😊 Sentiment",
                    "🔗 REST JSON",
                    "ℹ️ Pipeline"
                ]
            )


            # -------------------------------------------------
            # PREPROCESSING TAB
            # -------------------------------------------------

            with tab1:

                st.markdown(
                    "**Cleaned text**"
                )

                st.code(
                    result["cleaned_text"]
                    or "(empty)"
                )


                a, b = st.columns(2)


                with a:

                    st.markdown(
                        "**Tokens**"
                    )

                    st.write(
                        result["tokens"]
                    )


                with b:

                    st.markdown(
                        "**Stop-word removed tokens**"
                    )

                    st.write(
                        result["filtered_tokens"]
                    )


                st.markdown(
                    "**Lemmas**"
                )

                st.write(
                    result["lemmas"]
                )


            # -------------------------------------------------
            # ENTITY TAB
            # -------------------------------------------------

            with tab2:

                if entities:

                    entity_df = (
                        pd.DataFrame(
                            entities
                        )
                        .rename(
                            columns={
                                "text": "Entity",
                                "label": "Label",
                                "description": "Meaning"
                            }
                        )
                    )


                    st.dataframe(
                        entity_df,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.info(
                        "No named entities were detected."
                    )


            # -------------------------------------------------
            # SENTIMENT TAB
            # -------------------------------------------------

            with tab3:

                score = sentiment["score"]


                st.metric(
                    "Predicted sentiment",
                    sentiment["label"]
                )


                st.progress(score)


                st.write(
                    f"Confidence score: "
                    f"**{score:.2%}**"
                )


                st.caption(
                    "Model: DistilBERT fine-tuned "
                    "for English SST-2 sentiment classification."
                )


            # -------------------------------------------------
            # REST JSON TAB
            # -------------------------------------------------

            with tab4:

                st.json(
                    result
                )


            # -------------------------------------------------
            # PIPELINE TAB
            # -------------------------------------------------

            with tab5:

                st.markdown(
                    """
                    ### 1. NLTK Preprocessing

                    NLTK performs:

                    - Text cleaning
                    - Tokenization
                    - Stop-word removal

                    ### 2. spaCy Entity Extraction

                    spaCy identifies entities such as:

                    - PERSON
                    - ORGANIZATION
                    - LOCATION
                    - DATE
                    - PRODUCT
                    - MONEY

                    ### 3. Hugging Face Sentiment Analysis

                    The Hugging Face model classifies text as:

                    - POSITIVE
                    - NEGATIVE

                    and provides a confidence score.

                    ### 4. FastAPI REST Service

                    The project provides:

                    - `/health`
                    - `/preprocess`
                    - `/entities`
                    - `/sentiment`
                    - `/analyze`
                    """
                )


        except requests.RequestException as exc:

            st.error(
                f"REST API connection failed: {exc}"
            )

            st.info(
                "Start the FastAPI server or "
                "switch to Local Pipeline mode."
            )


        except Exception as exc:

            st.error(
                f"NLP pipeline error: {exc}"
            )

            st.info(
                "Check that the spaCy model and "
                "Python dependencies are installed."
            )