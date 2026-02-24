from flask import Flask, request, jsonify, render_template, session
from npmai import Memory
import requests
import json
import uuid
import os

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "979f83cfdc626eb35847c1ae3193c5accc3d0bfbb38991ee287ec108b2bd7739")

HF_API = "https://sonuramashish22028704-npmeduai.hf.space/ingestion"

@app.route("/")
def index():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_id = session.get('user_id', str(uuid.uuid4()))
    memory = Memory(user_id)
    
    files = {}
    data = {}

    # Required
    data["query"] = request.form.get("query")
    data["DB_PATH"] = request.form.get("DB_PATH")
    data["temperature"] = 0.5
    data["model"] = "llama3.2"

    # Optional
    if "file" in request.files:
        f = request.files["file"]
        files["file"] = (f.filename, f.stream, f.mimetype)

    if request.form.get("link"):
        data["link"] = request.form.get("link")
        data["output_path"] = request.form.get("output_path")

    try:
        history = memory.load_memory_variables()
        full_prompt = f"Context history:\n{history}\nHuman: {data}\nAI:"
        res = requests.post(HF_API, data=data, files=files if files else None, timeout=1200)
        response = str(res)
        
        memory.save_context(data, response)
        
        if "application/json" in res.headers.get("Content-Type", ""):
            return jsonify({"response": res.json().get("response")})
        else:
            return jsonify({"response": f"HF Error: {res.status_code}. API might be down or blocked."})
    
    except Exception as e:
        return jsonify({"response": f"Flask Error: {str(e)}"})
        
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
