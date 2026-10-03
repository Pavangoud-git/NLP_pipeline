from __future__ import annotations

import re
from typing import Any, Dict, List

import nltk
from nltk.tokenize import word_tokenize
import spacy
from transformers import pipeline


def ensure_nltk_resources() -> None:
    """
    Download required NLTK resources if they are not already available.
    """

    resources = [
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]

    for resource_path, package_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, quiet=True)


# Download required NLTK resources
ensure_nltk_resources()

# English stop words
STOP_WORDS = set(
    nltk.corpus.stopwords.words("english")
)


class NLPPipeline:
    """
    Complete NLP pipeline:

    1. NLTK -> preprocessing
    2. spaCy -> Named Entity Recognition
    3. Hugging Face -> sentiment analysis
    """

    def __init__(self) -> None:

        # Load spaCy English model
        self.nlp = spacy.load(
            "en_core_web_sm"
        )

        # Load Hugging Face sentiment model
        self.sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    # ---------------------------------------------------------
    # TEXT CLEANING
    # ---------------------------------------------------------

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean input text.
        """

        text = text.strip()

        # Remove URLs
        text = re.sub(
            r"https?://\S+|www\.\S+",
            " ",
            text
        )

        # Remove email addresses
        text = re.sub(
            r"\S+@\S+",
            " ",
            text
        )

        # Keep letters, numbers, spaces, apostrophes and hyphens
        text = re.sub(
            r"[^A-Za-z0-9\s'\-]",
            " ",
            text
        )

        # Remove extra spaces
        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # ---------------------------------------------------------
    # PREPROCESSING
    # ---------------------------------------------------------

    def preprocess(self, text: str) -> Dict[str, Any]:

        cleaned = self.clean_text(text)

        # spaCy document
        doc = self.nlp(cleaned)

        # NLTK tokenization
        tokens = word_tokenize(cleaned)

        # Stop word removal
        filtered_tokens = [
            token
            for token in tokens
            if token.isalpha()
            and token.lower() not in STOP_WORDS
        ]

        # Lemmatization using spaCy
        lemmas = [
            token.lemma_
            for token in doc
            if not token.is_space
            and token.is_alpha
            and token.lower_ not in STOP_WORDS
        ]

        return {
            "original_text": text,
            "cleaned_text": cleaned,
            "tokens": tokens,
            "filtered_tokens": filtered_tokens,
            "lemmas": lemmas
        }

    # ---------------------------------------------------------
    # ENTITY EXTRACTION
    # ---------------------------------------------------------

    def entities(
        self,
        text: str
    ) -> List[Dict[str, str]]:

        doc = self.nlp(text)

        entities = []

        for ent in doc.ents:

            entities.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "description": (
                        spacy.explain(ent.label_)
                        or ent.label_
                    )
                }
            )

        return entities

    # ---------------------------------------------------------
    # SENTIMENT ANALYSIS
    # ---------------------------------------------------------

    def sentiment_analysis(
        self,
        text: str
    ) -> Dict[str, Any]:

        result = self.sentiment(
            text,
            truncation=True,
            max_length=512
        )[0]

        return {
            "label": result["label"],
            "score": round(
                float(result["score"]),
                4
            )
        }

    # ---------------------------------------------------------
    # COMPLETE ANALYSIS
    # ---------------------------------------------------------

    def analyze(
        self,
        text: str
    ) -> Dict[str, Any]:

        if not text or not text.strip():
            raise ValueError(
                "Text cannot be empty."
            )

        preprocessing = self.preprocess(text)

        cleaned = preprocessing[
            "cleaned_text"
        ]

        return {
            **preprocessing,

            "entities": self.entities(
                cleaned
            ),

            "sentiment": self.sentiment_analysis(
                cleaned
            )
        }


# ---------------------------------------------------------
# SINGLETON PIPELINE
# ---------------------------------------------------------

_pipeline_instance: NLPPipeline | None = None


def get_pipeline() -> NLPPipeline:

    global _pipeline_instance

    if _pipeline_instance is None:

        _pipeline_instance = NLPPipeline()

    return _pipeline_instance