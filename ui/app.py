import streamlit as st
import requests

st.set_page_config(page_title="RAG Chatbot", layout="centered")

st.title("🧠 LangChain RAG Chatbot")
st.markdown("Upload a document (PDF, Markdown, or TXT) and ask questions about it!")

uploaded_file = st.file_uploader("Upload your document", type=["txt", "pdf", "md"])
if uploaded_file:
    with st.spinner("Uploading and indexing..."):
        response = requests.post(
            "http://localhost:8000/upload",
            files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)},
        )
        st.success("File uploaded and indexed!" if response.ok else response.text)

question = st.text_input("Ask a question:")
if question:
    with st.spinner("Thinking..."):
        try:
            # Send the question to the backend
            response = requests.post("http://localhost:8000/ask", json={"question": question})
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Parse the response
            data = response.json()

            # Safely handle the backend response
            if "answer" in data:
                st.markdown(f"**Answer:** {data['answer']}")
            elif "error" in data:
                st.error(f"Error: {data['error']}")
            else:
                st.error("Unexpected response format from the server.")
        except Exception as e:
            # Handle any request exceptions
            st.error(f"Request failed: {str(e)}")
