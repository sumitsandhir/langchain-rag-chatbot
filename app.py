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

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise EnvironmentError("❌ OPENAI_API_KEY not set in .env file")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

app = FastAPI()

embedding = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4", temperature=0)
retriever = None
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

class QueryInput(BaseModel):
    question: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global retriever
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

    db = Chroma.from_documents(texts, embedding, persist_directory="chroma_store")
    db.persist()
    retriever = db.as_retriever()

    return {"message": f"Uploaded and indexed {file.filename}"}

@app.post("/ask")
async def ask_question(query: QueryInput):
    if not retriever:
        return {"error": "No document uploaded yet."}
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
    result = qa.run(query.question)
    return {"answer": result}
