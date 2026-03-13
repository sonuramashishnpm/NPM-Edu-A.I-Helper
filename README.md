# NPM Rag A.I

## 🚀 Quick Start

```bash
# 1. Clone the project
git clone https://github.com/sonuramashishnpm/NPM-Rag-AI.git
cd NPM-Rag-AI

# 2. Install dependencies
pip install flask werkzeug requests npmai

# 3. (Recommended) Set secret key for secure sessions
export SECRET_KEY="your-very-long-random-secret-string-here"

# 4. Launch the app
python app.py
```

→ Open browser → https://npmragai.onrender.com

Select mode:
- Chat
- Use Your ORG RAG Chat
- Develop Rag Chat

Upload PDFs / images / videos / YouTube link  
Enter knowledge base name → start asking questions or create new DB

Powered by npmai library (Memory + Rag classes) + llama3.2 via custom Hugging Face Spaces endpoint.  
No vector database installation needed — npmai handles ingestion, storage & retrieval.

## 📖 Project Overview

NPM Rag A.I is a beautiful, easy-to-use web application that lets you instantly create and talk to your own private or public knowledge bases using RAG (Retrieval-Augmented Generation).

Just:
- Upload your PDFs, images (.png/.jpg/.jpeg), videos (.mp4)
- Or paste a YouTube link
- Choose public or private (with secret key)
- Give it a name
- Start asking questions in natural language

You get accurate, document-grounded answers with full conversation memory — no complicated setup, no paid vector stores, no local embedding models.

Made possible by the lightweight npmai Python library which provides simple Memory (chat history) and Rag (upload → vectorize → query) classes.

Ideal for:
- Students (notes, papers, lecture videos)
- Teachers & researchers
- Content creators
- Small teams / organizations (private knowledge base)

Part of the NPMAI ecosystem — powerful AI made stupidly simple and free.

## ✨ Features

- Three modes in one clean interface
  • Chat → general conversation + file/YouTube upload
  • Use ORG RAG Chat → query existing knowledge base (public or private)
  • Develop Rag Chat → build new knowledge base from files & links

- File support: PDF, PNG/JPG/JPEG, MP4, YouTube links (auto processed)
- Public ↔ Private knowledge bases (protect with secret username/key)
- Persistent conversation memory per user/session
- Modern dark UI with glassmorphism cards + animated background video
- Zero-config RAG powered by npmai library
- Clear error messages shown right in the chat box

## 🛠️ Tech Stack

- Backend → Flask (routes, file uploads, sessions)
- Frontend → HTML + CSS (glass effect + video background) + vanilla JavaScript
- AI engine → npmai (Memory class for history + Rag class for RAG)
- LLM → llama3.2 (via custom HF Spaces ingestion & query API)
- Security → session-based user ID + secure_filename + optional private key
- Deployment ready → works on Render, Railway, Fly.io, Hugging Face Spaces, etc.

## 👨‍💻 Developer

**Sonu Kumar Ramashish** (a.k.a. Bihar Viral Boy)  
- Age: 14 | Student | TEDx Speaker | AI & Software & Web & Cloud Developer | DevOps | Social Thinker  
- Reach: 430K+ Facebook followers  
- Location: Kota, Rajasthan  

Part of NPMAI ecosystem for AI automation tools.

## 🤝 Contributing

Fork → add cool stuff (upload progress bar, more file formats, mobile improvements, etc.) → send pull request

License: MIT

If this project helps you — give it a ⭐ bro 🔥
```

Bro — this is **everything in one single markdown block**.  
Just copy from the very first line `# NPM Rag A.I` all the way to the last line and paste it into your README.md file.  

No new boxes, no separate sections outside — done.  
Good to go now? 😤
```
