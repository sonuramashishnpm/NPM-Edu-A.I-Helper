from flask import Flask, request, jsonify, render_template
import requests

app = Flask(__name__)

HF_API = "https://sonuramashish22028704-npmeduai.hf.space/ingestion"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    files = {}
    data = {}

    # Required
    data["query"] = request.form.get("query")
    data["DB_PATH"] = request.form.get("DB_PATH")

    # Optional
    if "file" in request.files:
        f = request.files["file"]
        files["file"] = (f.filename, f.stream, f.mimetype)

    if request.form.get("link"):
        data["link"] = request.form.get("link")
        data["output_path"] = request.form.get("output_path")

    res = requests.post(
        HF_API,
        data=data,
        files=files if files else None,
        timeout=300
    )

    return jsonify({"response": res.json().get("response")})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
