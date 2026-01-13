from flask import Flask, render_template, request, jsonify
from npmai import Ollama
import os, base64
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_classic.chains import RetrievalQA
import httpx


app = Flask(__name__)
UPLOAD_DIR = "uploads"
DB_DIR = "vector_dbs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

llm = Ollama(
    model="llama3.2",
    temperature=0.8
)
embeddings = FastEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    max_length=512
)

# ---------------- HELPERS ---------------- #

async def extract_pdf(path):
    async with httpx.AsyncClient() as client:
        with open(path,"rb") as f:
            file=f
            text=await client.post("https://npmai-api.onrender.com/uploadfile",files={
                "file":("document.pdf",file, "application/pdf")})
            return text.text

async def ocr_image(path):
    async with httpx.AsyncClient() as client:
        with open(path,"rb") as f:
            file=f
            text=await client.post("https://npmvoiceai.onrender.com/ocr",files={
                "file":("image.png",file,"image/png")})
            return text.text

def text_processes(path):
    with open(path,"r") as f:
        text=f.read()
        return text

# ---------------- ROUTES ---------------- #

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data["question"]
    db_name = data["db_name"]
    source_type = data.get("source_type")
    

    db_path = os.path.join(DB_DIR, db_name)

    if os.path.exists(db_path):
        vector_db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    else:
        text = ""

        if source_type == "pdf":
            file_path = data["file_path"]
            text = extract_pdf(file_path)

        elif source_type == "image":
            file_path = data["file_path"]
            text = ocr_image(file_path)

        elif source_type== "text":
            file_path=data["file_path"]
            text= text_processes(file_path)
            
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        vector_db = FAISS.from_texts(chunks, embeddings)
        vector_db.save_local(db_path)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(),
        chain_type="refine"
    )

    response = qa.invoke(question)
    return jsonify({"response": str(response)})

# ---------------- FILE UPLOAD ---------------- #

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files["file"]
    path = os.path.join(UPLOAD_DIR, f.filename)
    f.save(path)
    return jsonify({"path": path})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
