from flask import Flask,request,render_template,session,jsonify
from werkzeug.utils import secure_filename
from npmai import Memory,Rag
import requests
import json
import uuid
import os

app= Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY","979f83cfdc626eb35847c1ae3193c5accc3d0bfbb38991ee287ec108b2bd7739")

HF_API = "https://sonuramashish22028704-npmeduai.hf.space/ingestion"

@app.route("/")
def index_render():
  if 'user_id' not in session:
      session['user_id'] = str(uuid.uuid4())
  return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
  user_id = session.get('user_id', str(uuid.uuid4()))
  memory = Memory(user_id)

  data = {}

  data["query"] = request.form.get("query")
  data["DB_PATH"] = request.form.get("DB_PATH")
  data["temperature"] = 0.5
  data["model"] = "llama3.2"

  filess= request.files.getlist('file')
  files=[]
  folders="rag_db_upload"

  os.makedirs(folders,exist_ok=True)

  for f in filess:
    if f.filename != '':
      file_name=secure_filename(f.filename)
      file_paths=os.path.join(folders,file_name)
      f.save(file_paths)
      files.append(('file', open(file_paths, 'rb')))

  if request.form.get("link"):
    data["link"] = request.form.get("link")
    data["output_path"] = request.form.get("output_path")
    
  try:
    history = memory.load_memory_variables()
    full_prompt = f"Context history:\n{history}\nHuman: {data["query"]}\nAI:"
    res = requests.post(HF_API, data=data, files=files if files else None, timeout=1200)
    response = str(res)
        
    memory.save_context(data["query"], res.json().get("response"))
        
    if "application/json" in res.headers.get("Content-Type", ""):
      return jsonify({"response": res.json().get("response")})
    else:
      return jsonify({"response": f"HF Error: {res.status_code}. API might be down or blocked."})
    
  except Exception as e:
    return jsonify({"response": f"Flask Error: {str(e)}"})


@app.route("/use",methods=["POST"])
def use_vectordb():
  user_id1 = session.get('user_id', str(uuid.uuid4()))
  memory1 = Memory(user_id1)
  
  #DATA
  DB_Privacy = request.form.get("DB_Privacy")
  DB_PATH= request.form.get("DB_PATH")
  query= request.form.get("query")

  if DB_Privacy=="privat":
    secret_key=request.form.get("secret_key")

    load_memory= memory1.load_memory_variables()
    final_query= f"Context history:\n{load_memory}\nHuman: {query}\nAI:"
    rag=Rag(
        DB_PATH=DB_PATH,
        secret_key=secret_key,
        query=final_query
    )
  else:
    rag=Rag(
        DB_PATH=DB_PATH,
        public=True,
        query=query,
    )
    
  response=rag.vector_db_use()
  memory1.save_context(query,response)
  return jsonify({"response":response})

  
@app.route("/develop",methods=["POST"])
def develop_rag():
  DB_PATH=request.form.get("DB_PATH")
  DB_Privacy = request.form.get("DB_Privacy")
  link = request.form.get("link")
  output_path = request.form.get("output_path")
  
  filess=request.files.getlist("file")
  files=[]
  folders="rag_db_upload"

  os.makedirs(folders,exist_ok=True)

  for f in filess:
    if f.filename != '':
      file_name=secure_filename(f.filename)
      file_paths=os.path.join(folders,file_name)
      f.save(file_paths)
      files.append(file_paths)
      
  
  if DB_Privacy=="privat":
    secret_key=request.form.get("secret_key")
    
    rag=Rag(
      DB_PATH=DB_PATH,
      secret_key=secret_key,
      Upload=True,
      files=files,
      link=link,
      output_path=output_path
      )
  elif DB_Privacy=="publi":

    rag=Rag(
        DB_PATH=DB_PATH,
        public=True,
        Upload=True,
        files=files,
        link=link,
        output_path=output_path
    )

  response=rag.send()
  return jsonify({"response":"DONE check by using the Chat"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
  
