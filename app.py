from flask import Flask, render_template, request, jsonify
from npmai import Ollama
import os, fitz, cv2, base64
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA


app = Flask(__name__)
UPLOAD_DIR = "uploads"
DB_DIR = "vector_dbs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

llm = Ollama(
    model="llama3.2",
    temperature=0.8
)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

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

# ---------------- HELPERS ---------------- #

def extract_pdf(path):
    doc = fitz.open(path)
    text = ""
    for p in doc:
        t = p.get_text().strip()
        if t:
            text += t
    if text:
        return text
    images = convert_from_path(path)
    return "\n".join(pytesseract.image_to_string(img) for img in images)

def ocr_image(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray)

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
