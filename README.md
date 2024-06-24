# PDFInsightBot - Talk to your PDF
PDFInsightBot is an intelligent chatbot designed to engage in conversations about the content of your PDF documents. By uploading a PDF, users can ask questions and receive detailed, contextual answers based on the document's text. 
It utilizes the capabilities of Natural Language Processing (NLP), Retreival Augmented Generation (RAG) and Large Language Model (LLM). It has been built using state-of-the-art technologies and practices including LangChain, LLMs (Mistral), FastAPI and Streamlit.

## Table of Contents
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Mistral](#about-mistral)
  - [Requirements](#requirements)
  - [Usage](#usage)
  - [Future Work](#future-work)

## Features
1. Upload and process PDF documents
2. Ask questions and receive detailed answers based on the PDF content
3. Uses advanced language models and vector databases for efficient document querying
   
## About Mistral
Mistral is a state-of-the-art language model used in PDFInsightBot for generating embeddings and responses. It enhances the chatbot's ability to understand and process the content of PDF documents, providing accurate and contextually relevant answers to user queries. 

## Requirements
1. Python 3.8+
2. Streamlit
3. Langchain
4. PyPDFLoader
5. Chroma
6. Ollama
7. Mistral (Language model used for embeddings and responses)
   
## Usage
1. Clone the repository:
   ```
    git clone https://github.com/pranavdhawann/PDFInsightBot.git
    cd PDFInsightBot
   ```
2. Create a virtual environment:
   ```
    python3 -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```
3. Install dependencies:
   ```
    pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```
    streamlit run app.py
   ```
## Future Work
- [ ] Add Support for Different Document Types: Extend the capabilities of PDFInsightBot to support additional document types such as Excel (.xlsx), Word (.docx), and others.

- [ ] Higher Speed: Optimize the performance of PDFInsightBot to achieve higher processing and response speeds, providing a more efficient user experience.
