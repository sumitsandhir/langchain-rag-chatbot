import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Initialize FastAPI app
app = FastAPI()

# Initialize global variables
retriever = None
hf_embeddings = None
use_openai = False

# Check for OpenAI API Key in the environment
if os.getenv("OPENAI_API_KEY"):
    try:
        # Prefer OpenAI embeddings if the API key exists
        print("🔑 OpenAI API key detected. Initializing OpenAI embeddings...")
        hf_embeddings = OpenAIEmbeddings()  # Leverage OpenAI for embeddings
        use_openai = True
        print("✅ OpenAI embeddings initialized.")
    except Exception as e:
        print(f"❌ Error initializing OpenAI embeddings: {e}")
else:
    try:
        # Fallback to HuggingFace embeddings if no OpenAI API key is found
        print("🔍 No OpenAI API key found. Using HuggingFace embeddings...")
        hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("✅ HuggingFace embeddings initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing HuggingFace embeddings: {e}")
        hf_embeddings = None


class QueryInput(BaseModel):
    """Schema for questions."""
    question: str


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to handle document upload and retriever setup.
    """
    global retriever

    if not hf_embeddings:
        return {"error": "Embeddings could not be initialized. Please check your environment settings."}

    # Ensure persistent storage for file storage
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save the uploaded file locally
    file_path = os.path.join(data_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Detect the file type and use the appropriate loader
    ext = file.filename.split(".")[-1].lower()
    if ext == "pdf":
        loader = PyPDFLoader(file_path)
    elif ext == "md":
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        loader = TextLoader(file_path)

    # Load the document
    documents = loader.load()
    print(f"✅ Loaded document: {file.filename}")

    # Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"✅ Document split into {len(texts)} chunks.")

    # Initialize Chroma database
    try:
        db = Chroma(
            collection_name="document_index",
            persist_directory="chroma_store",
            embedding_function=hf_embeddings,
        )
        db.add_texts([doc.page_content for doc in texts])
        db.persist()

        # Set up a retriever for querying
        retriever = db.as_retriever()
        print("✅ ChromaDB retriever initialized successfully.")
        return {"message": f"File '{file.filename}' has been uploaded and indexed."}

    except Exception as e:
        print(f"❌ Error initializing ChromaDB: {e}")
        retriever = None
        return {"error": f"ChromaDB initialization failed: {e}"}


@app.post("/ask")
async def ask_question(query: QueryInput):
    """
    Endpoint to handle user questions with a fallback between OpenAI and local LLaMA.
    """
    global retriever, use_openai

    if not retriever:
        return {"error": "No document has been uploaded or retriever is not initialized."}

    # Ensure OpenAI key fallback logic during question answering
    try:
        if use_openai:
            # Use OpenAI's GPT model if OpenAI embeddings are used
            print("🧠 Using OpenAI GPT for question answering...")
            openai_qa_model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
            qa_chain = RetrievalQA.from_chain_type(
                retriever=retriever,
                llm=openai_qa_model,
                chain_type="stuff",
            )
            answer = qa_chain.run(query.question)
        else:
            # Use fallback logic for local answers
            print("⚙️ Using local retriever for response...")
            retrieved_docs = retriever.get_relevant_documents(query.question)
            if not retrieved_docs:
                return {
                    "answer": "No relevant information found. Try uploading a relevant document or rephrasing your question."
                }
            context = " ".join([doc.page_content for doc in retrieved_docs[:3]])
            answer = f"Relevant context:\n{context}"

        return {"answer": answer}

    except Exception as e:
        print(f"❌ Error during question answering: {e}")
        return {"error": "An error occurred while processing your query. Please try again later."}