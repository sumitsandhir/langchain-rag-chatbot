import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import requests
import json

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LOCAL_LLAMA_ENDPOINT = "http://localhost:11434"  # Updated to correct URL

if not OPENAI_API_KEY and not LOCAL_LLAMA_ENDPOINT:
    raise EnvironmentError("❌ No valid LLM configuration found. Set OPENAI_API_KEY or a local Llama endpoint.")

app = FastAPI()

# Use OpenAIEmbeddings, if API key is available
embedding = OpenAIEmbeddings() if OPENAI_API_KEY else None
llm = ChatOpenAI(model="gpt-4", temperature=0) if OPENAI_API_KEY else None
retriever = None
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


class QueryInput(BaseModel):
    question: str


@app.post("/upload")
async def upload_file( file: UploadFile = File(...) ):
    global retriever
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    ext = file.filename.split('.')[-1].lower()
    if ext == 'pdf':
        loader = PyPDFLoader(file_path)
    elif ext == 'md':
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        loader = TextLoader(file_path)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    db = Chroma.from_documents(texts, embedding, persist_directory="chroma_store") if embedding else None
    if db:
        db.persist()
        retriever = db.as_retriever()
    else:
        retriever = None

    return {"message": f"Uploaded and indexed {file.filename}"}


@app.post("/ask")
async def ask_question(query: QueryInput):
    global retriever

    try:
        # Attempt to use OpenAI
        if retriever and llm:
            qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
            result = qa.run(query.question)
            return {"answer": result}  # Always return "answer"
    except Exception as e:
        # Log the OpenAI error and proceed to fallback
        print(f"OpenAI Error: {e}")

    try:
        # Query the Local Llama server
        response = requests.post(
            f"{LOCAL_LLAMA_ENDPOINT}/api/generate",
            json={
                "model": "llama2",  # Replace with your configured model name
                "prompt": f"You are an expert assistant. Answer the following question concisely:\n\n{query.question}",
                "temperature": 0.7,
            },
            stream=False,
        )

        # Print the raw response content for debugging
        print("Raw Response Content:", response.text)
        response.raise_for_status()  # Raise for HTTP error status codes

        # Parse JSONL response (handle multiple JSON objects)
        full_response = ""
        for line in response.text.splitlines():
            try:
                json_obj = json.loads(line)   # Parse each JSON object
                full_response += json_obj.get("response", "")  # Extract the "response" field
                if json_obj.get("done"):      # Stop if streaming is marked as "done"
                    break
            except json.JSONDecodeError:
                # Handle malformed JSON lines if needed
                print("Warning: Skipped invalid JSON line:", line)
                continue

        return {
            "answer": full_response.strip()  # Return the combined response
        }

    except requests.exceptions.RequestException as e:
        # Handle HTTP-related exceptions
        print("Request Exception:", e)
        return {"error": f"Failed to query local Llama: {str(e)}"}

    except ValueError as e:
        # Handle JSON decoding errors
        print("JSON Decode Error:", e)
        return {"error": f"Invalid JSON from Local Llama: {response.text}"}
