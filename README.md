# 🧠 LangChain RAG Chatbot with ChromaDB

## Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload local documents (PDF, Markdown, or TXT) and ask questions about their content. The chatbot retrieves relevant information from the documents and generates answers using LangChain, ChromaDB, and OpenAI's GPT.

---

## Key Features

- **File Upload & Processing**:
    - Supports PDF, Markdown, and TXT file formats for document uploads.
    - Automatically indexes the uploaded documents for efficient information retrieval.

- **Retrieval-Augmented Generation**:
    - Uses ChromaDB for document embedding and retrieval.
    - Enables GPT-based contextual question answering by combining relevant document content with user questions.

- **User-Friendly Interface**:
    - Built with Streamlit for an intuitive interface, allowing users to:
        - Upload documents.
        - Ask natural language questions.
        - View dynamically generated answers.

- **API-Centric Backend**:
    - Communicates with backend endpoints to handle document uploads and answer generation.

---

## Technologies Used

1. **LangChain**: Framework for orchestrating retrieval-augmented language generation workflows.
2. **ChromaDB**: Vector database used for document embedding and fast retrieval of relevant contexts.
3. **OpenAI GPT**: Language model for generating contextual answers.
4. **Streamlit**: Frontend framework for building the user-facing interface.
5. **FastAPI**: Backend framework for building API endpoints (e.g., `/upload`, `/ask`).

---

## Installation and Setup

### Prerequisites
- **Python 3.10 or higher**
- Docker (Optional, for containerized deployment)

### Steps to Run Locally

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Start the Streamlit app:
    ```bash
    streamlit run app.py
    ```
4. Access the application in your browser at: `http://localhost:8501`.

---

## Running with Docker

1. Build the Docker image:
    ```bash
    docker build -t rag-chatbot .
    ```

2. Run the Docker container:
    ```bash
    docker run -p 8501:8501 rag-chatbot
    ```

3. Access the application in your browser at: `http://localhost:8501`.

---

## Usage

1. Open the application in your browser.
2. Upload a document (PDF, TXT, or Markdown format).
3. Ask questions related to the uploaded document in natural language.
4. View the generated answers based on the document's content.

---

## Project Structure

- **`app.py`**: The main Streamlit application for handling the frontend and API requests.
- **`Dockerfile`**: Configuration for containerizing the application.
- **`requirements.txt`**: All dependencies required for the project.
- **Backend Endpoints**:
    - `/upload`: Handles document uploads and indexing.
    - `/ask`: Accepts user questions and provides answers based on the uploaded document's content.

---

## Future Enhancements

Planned features include:
- Adding support for images or other document formats.
- Improving scalability by integrating distributed vector search engines.
- Enabling long-term document storage and retrieval.

---