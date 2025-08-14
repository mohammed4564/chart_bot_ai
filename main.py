import os
from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

# Load env variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

app = Flask(__name__)

llm = ChatGroq(
    model_name="llama3-8b-8192",
    temperature=0.7,
    groq_api_key=groq_api_key
)

def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore.as_retriever()

PDF_PATH = "pdfs/smartphone_support_guide (2).pdf"

print("Current working directory:", os.getcwd())
print("File exists:", os.path.exists(PDF_PATH))

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF file not found at {PDF_PATH}")

retriever = process_pdf(PDF_PATH)
print(f"Loaded PDF from {PDF_PATH}")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    question = data["question"]

    if retriever is None:
        return jsonify({"error": "PDF not loaded yet"}), 500

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain({"query": question})

    return jsonify({
        "question": question,
        "answer": result["result"]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
