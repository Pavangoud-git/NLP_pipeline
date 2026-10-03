🧠 NLP Intelligence Studio

An interactive NLP application built with Streamlit that performs text preprocessing, tokenization, stop-word removal, lemmatization, Named Entity Recognition (NER), sentiment analysis, and REST API-based text preprocessing.

🚀 Project Overview

NLP Intelligence Studio provides a simple web interface to analyze text using Natural Language Processing techniques.

The application follows this pipeline:

Input Text
    ↓
Text Cleaning
    ↓
Tokenization
    ↓
Stop-word Removal
    ↓
Lemmatization
    ↓
Named Entity Recognition (NER)
    ↓
Sentiment Analysis

It also supports a REST API mode for connecting NLP preprocessing functionality with other applications.

✨ Features

🧹 Text cleaning and preprocessing

🔤 Tokenization

🛑 Stop-word removal

🌱 Lemmatization

🏷️ Named Entity Recognition (NER)

😊 Sentiment analysis

🔗 REST API integration

📊 Interactive analysis dashboard

🖥️ Streamlit-based user interface

📄 REST JSON output

🛠️ Technologies Used

Technology

Purpose

Python

Core programming language

Streamlit

Web application UI

spaCy

NLP processing and NER

NLTK

Tokenization and stop-word processing

Hugging Face

NLP/sentiment analysis models

FastAPI

REST API services

Uvicorn

API server

Pandas

Data handling

Git & GitHub

Version control

📁 Project Structure

NLP-Intelligence-Studio/
│
├── app.py
├── api.py
├── requirements.txt
├── README.md
└── models/
    └── ...

File names may be adjusted based on your final project structure.

⚙️ Installation

1. Clone the repository

git clone https://github.com/your-username/NLP-Intelligence-Studio.git
cd NLP-Intelligence-Studio

2. Install dependencies

pip install -r requirements.txt

3. Download the spaCy English model

python -m spacy download en_core_web_sm

4. Run the Streamlit application

streamlit run app.py

The application will open in your browser.

🔗 Running the REST API

Start the FastAPI server:

uvicorn api:app --reload --host 0.0.0.0 --port 8000

The API will be available at:

http://127.0.0.1:8000

FastAPI documentation:

http://127.0.0.1:8000/docs

🧪 Example Input

I bought an Apple iPhone in Hyderabad last week.
The camera is excellent, but the battery life is disappointing.

Example Analysis

Entities:

Apple → ORG

Hyderabad → GPE

iPhone → PRODUCT

Sentiment:

Negative

The exact entities and sentiment output may vary depending on the NLP model used.

📊 Application Output

The Streamlit dashboard displays:

Original character count

Cleaned token count

Extracted entities

Sentiment result

Preprocessed text

Token information

REST JSON response

NLP pipeline stages

🔌 REST API Example

A sample request can be sent to the preprocessing endpoint:

{
  "text": "This is an example sentence for NLP preprocessing."
}

A typical response contains processed text/token information in JSON format.

🎯 Use Cases

This project can be used for:

Customer review analysis

Social media text analysis

Sentiment monitoring

Entity extraction

Text preprocessing

NLP learning and demonstrations

Building NLP-enabled applications

Integrating NLP services through REST APIs

📚 NLP Concepts Demonstrated

Tokenization

Splits text into individual words/tokens.

Stop-word Removal

Removes common words that may not add much value for certain NLP tasks.

Lemmatization

Converts words into their meaningful base form.

Named Entity Recognition

Identifies entities such as organizations, locations, people, products, dates, and more.

Sentiment Analysis

Determines the emotional polarity of text, such as positive, negative, or neutral.

🌟 Future Enhancements

Support for multiple languages

Emotion detection

Text summarization

Keyword extraction

Topic modeling

Upload and analyze CSV/text files

User authentication

Cloud deployment

More NLP models from Hugging Face

👨‍💻 Author

Bheemagani Pavan

Data Analyst | Data Scientist | Machine Learning Enthusiast

GitHub: https://github.com/Pavangoud-git

LinkedIn: https://linkedin.com/in/pavan-bheemagani-632b4732b2
