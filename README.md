## Overview

This project implements a Retrieval Augmented Generation (RAG) system that leverages Azure OpenAI and FAISS indexing to generate answers based on the content of multiple .docx documents. The system reads documents from a specified folder, extracts their text, generates embeddings using the Azure OpenAI "text-embedding-ada-002" model, and indexes them with FAISS. At query time, it retrieves the most relevant document(s) and uses GPT-4 (via Azure OpenAI chat completions) to generate an answer that includes references to the source documents.

## Architecture

- **Document Ingestion:**  
  Reads and extracts text from .docx files stored in a specified folder.

- **Embedding Generation:**  
  Uses the Azure OpenAI "text-embedding-ada-002" model to convert document text into vector embeddings.

- **Indexing:**  
  Utilizes FAISS (Facebook AI Similarity Search) to index the document embeddings for fast similarity search.

- **Retrieval:**  
  Implements a retrieval pipeline that finds the top-k most relevant documents for a user query. Each retrieved document includes both its text and its filename (used as a reference).

- **Answer Generation:**  
  Combines the context (document content with names) into a chat prompt and uses the GPT-4 model (via Azure OpenAI) to generate an answer.

## Prerequisites

- **Databricks Environment:**  
  This project is designed to run in a Databricks Notebook.

- **Python Packages:**  
  The following packages are required:
  - `sentence-transformers`
  - `faiss-cpu`
  - `openai`
  - `langchain`
  - `python-docx`

- **Azure OpenAI Access:**  
  You need valid credentials (endpoint and subscription key) for Azure OpenAI.

## Installation

In your Databricks Notebook, run the following cell to install the required modules:

```python
%pip install sentence-transformers faiss-cpu openai langchain python-docx
