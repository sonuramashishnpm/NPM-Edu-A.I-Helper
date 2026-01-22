from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

url = "https://sonuramashish22028704-npmeduai.hf.space/ingestion"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.form
    question = data["question"]
    db_name = data["db_name"]
    
    files = {}
    if "file" in request.files:
        file = request.files["file"]
        files = {"file": (file.filename, file.stream, file.mimetype)}

    payload = {"query": question, "DB_PATH": db_name}
    response = requests.post(url, data=payload, files=files)
    
    return jsonify({"response": response.json()["response"]})
