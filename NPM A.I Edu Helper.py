llms=input("Enter A.I from which you want to study so select and write exact from here:-{'ChatGPT','Grok','Perplexity','Gemini','GeminiAIMode'}:")
email=input("Do you want to send your question response to other person via email(gmail) if yes then just write'y' and if no just write 'n'")
if email.lower()=="y":
    to=input("write email id of that person you want to send just write email id")
whatsapp=input("Do you want to send your question response to other person via Whatsapp if yes then just write'y' and if no just write 'n'")
from pdf2image import convert_from_path
from PIL import Image
from sentence_transformers import SentenceTransformer
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from npmai import ChatGPT,Perplexity,Grok,Gemini
import youtube_transcript_api
import numpy as np
import pytesseract
import fitz
import os
import cv2
if email.lower()=="y":
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    import base64
    from email.mime.text import MIMEText

else:
    pass

DB_PATH=input("Enter the name you want for your knowledge base")

if os.path.exists(DB_PATH):
    pass

else:
    path=input("Enter file path if document like pdf or photo then just enter the file path and if youtube video then write just vide_id")
    if path.lower().endswith(".pdf"):
        number=int(input("Enter total page you are submitting"))
question=input("Enter Your Question")
   
def pdf_has_text(path):
    doc=fitz.open(path)
    for i in range(len(doc)):
        page=doc[i]
        text=page.get_text().strip()
        if text:
            return True
    return False

def extractable_text(path):
    doc=fitz.open(path)
    full=[]
    for page in doc:
        full.append(page.get_text())
    return "\n".join(full)

def pdf_scanned_to_text(pdf_path, dpi=300, tesseract_lang='eng'):
    pages = convert_from_path(pdf_path, dpi=dpi)
    full = []
    for img in pages:
        full.append(pytesseract.image_to_string(img, lang=tesseract_lang, config='--psm 6'))
    return "\n\n".join(full)

def preprocess_for_ocr(path):
    img=cv2.imread(path,cv2.IMREAD_COLOR)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    h,w=gray.shape
    if w < 1000:
        gray=cv2.resize(gray,(int(w*2),int(h*2)),interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def ocr(path,lang="eng"):
    proc = preprocess_for_ocr(path)
    pil = Image.fromarray(proc)
    full = pytesseract.image_to_string(pil, lang=lang, config='--psm 6')
    return full

def get_transcript(path):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(path)
        full = " ".join([item["text"] for item in transcript])
        return full
    except Exception as e:
        return None

def send_email(to,subject,response):
    SCOPES=["https://www.googleapis.com/auth/gmail.send"]
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            "credentials.json", SCOPES
        )
        creds = flow.run_local_server(port=0)
        with open("token.json", "w") as f:
            f.write(creds.to_json())
    gmail=build("gmail", "v1", credentials=creds)

    message = MIMEText(response)
    message["to"] = to
    message["subject"] = subject

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

    return gmail.users().messages().send(
        userId="me",
        body={"raw": raw}
    ).execute()


def ingest_file(path):
    if path.endswith(".pdf") and path.lower().endswith(".pdf"):
        if pdf_has_text(path):
            return extractable_text(path)
        else:
            return pdf_scanned_to_text(path)
    elif any(path.lower().endswith(ext) for ext in ('.png','.jpg','.jpeg')):
             return ocr(path)
    else:
        return get_transcript(path)


emb=HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

if os.path.exists(DB_PATH):
    vector_db=FAISS.load_local(
        DB_PATH,
        emb,
        allow_dangerous_deserialization=True
        )
else:
    document=ingest_file(path)

    docs_for_rag=document
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
        )
    chunks=text_splitter.split_text(docs_for_rag)
   
    vector_db=FAISS.from_texts(chunks,emb)
    vector_db.save_local(DB_PATH)


retriever=vector_db.as_retriever()

llm=globals()[llms]()

qa=RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="refine"
    )

   
response=qa.invoke(question)
print(response)
if email.lower()=="y":
    subject=f"Query:{question} and below answer"
    send_email(to,subject,response)
