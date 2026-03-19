# NPM Rag A.I

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

Select mode:
- Chat
- Use Your ORG RAG Chat
- Develop Rag Chat

Upload PDFs / images / videos / YouTube link  
Enter knowledge base name → start asking questions or create new DB

Powered by npmai library (Memory + Rag classes) + llama3.2 via custom Hugging Face Spaces endpoint.  
No vector database installation needed — npmai handles ingestion, storage & retrieval.

## Important Update in NPMAI RAG:-

# 🚀 NPMAI Update: Advanced RAG & Refine Architecture

We have officially upgraded the **NPMAI Ecosystem** to a more intelligent, cost-efficient, and "Product-Ready" pipeline. These updates move beyond basic RAG into **High-Performance Agentic Retrieval**.

---

### 🔍 1. Dynamic K-Context Retrieval (70% Coverage)

**The Problem:** 
Standard RAG systems use a fixed `k` value (e.g., `k=4`). This is inefficient—it provides too little context for large documents (missing facts) and too much "noise" for tiny documents (wasting tokens).

**The Solution:** 
I have engineered a **Proportional Scaling Logic** that calculates the optimal number of chunks to retrieve based on the actual density of your vectorized database.

*   **Logic:** `dynamic_k = max(1, int(total_chunks * 0.70))`
*   **How it works:**
    *   **Short Documents:** If your database has only 2 chunks, the system retrieves only those 2.
    *   **Large PDFs:** If your PDF generates 100 chunks, the system automatically scales up to retrieve **70 relevant chunks** ($k=70$).
*   **The Impact:** This ensures the AI always sees a **statistically significant slice** of the knowledge base, adapting perfectly to any document size.

---

### 🔄 2. Sliding Window Batch-Refinement (3-Chunk Window)

**The Problem:** 
Traditional "Refine" strategies process one chunk at a time. This is incredibly slow because it makes $N$ separate API calls. For a 30-chunk document, the user waits too long.

**The Solution:** 
I have implemented a **Sliding Window Batch-Refine** system that processes chunks in groups of 3 instead of 1.

*   **Logic:** `for i in range(0, total_chunks, 3):`
*   **How it works:**
    *   Instead of making a single LLM call for every 1,000 characters, the system sends a **batch of 3 related chunks** (3,000 characters) in one go.
    *   It uses the previous answer as a "Running Memory" to merge new information from the current 3-chunk batch.
*   **The Impact:** 
    *   **3x Faster Execution:** We have reduced total API latency by **66%**.
    *   **Improved Coherence:** The AI sees a broader context ($3,000$ chars vs $1,000$ chars), allowing it to spot connections between facts that are split across neighboring chunks.

---

### ☁️ 3. Infrastructure: Persistent Supabase Integration (v0.1.8)

We have successfully integrated **Supabase Object Storage** to move from temporary memory to **Persistent Knowledge Bases**.

*   **Vector Persistence:** All `.faiss` and `.pkl` index files are now automatically uploaded to a secure Supabase bucket.
*   **Multi-Platform Access:** This allows **NPM-Rag-AI**, **NPM-AutoCode-AI**, and the **npmai SDK** to share and load the same vectorized data from anywhere in the world.

---
**Summary:** 
These architectural changes make **NPMAI** one of the most efficient open-source RAG frameworks available for developers who need **Speed + Accuracy** without the high cost of standard 1-by-1 refinement.


## 🛠️ Tech Stack

- Backend → Flask (routes, file uploads, sessions)
- Frontend → HTML + CSS (glass effect + video background) + vanilla JavaScript
- AI engine → npmai (Memory class for history + Rag class for RAG)
- LLM → llama3.2 (via custom HF Spaces ingestion & query API)
- Security → session-based user ID + secure_filename + optional private key
- Deployment ready → works on Render, Railway, Fly.io, Hugging Face Spaces, etc.

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
