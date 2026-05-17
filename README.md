# AI Agents Portfolio

A collection of AI-powered agents built with Python, Groq LLM, and various APIs. These projects demonstrate practical skills in LLM integration, RAG pipelines, and agentic AI systems.

---

## Projects

### 1. `my_first_agent_groq.py` — Simple AI Chatbot
A minimal AI agent that sends a question to Groq's LLM and prints the response. The starting point for understanding how LLM APIs work.

**Skills demonstrated:** Groq API, environment variables, python-dotenv

**How to run:**
```bash
python my_first_agent_groq.py
```

---

### 2. `research_agent.py` — AI Research Agent (DuckDuckGo)
An agentic bot that takes any topic, searches DuckDuckGo for live results, and uses Groq's LLM to generate a readable summary. Saves the output as a `.txt` report automatically.

**Skills demonstrated:** REST APIs, HTTP requests, LLM summarization, file I/O, agent loops

**How to run:**
```bash
python research_agent.py
```

---

### 3. `research_agent_wiki.py` — AI Research Agent (Wikipedia)
An improved version of the research agent that uses Wikipedia's API instead of DuckDuckGo for more reliable and structured results. Includes proper HTTP headers and error handling.

**Skills demonstrated:** REST APIs, HTTP headers, error handling, LLM summarization, agent loops

**How to run:**
```bash
python research_agent_wiki.py
```

---

### 4. `pdf_qa_bot.py` — PDF Q&A Bot (RAG)
A Retrieval Augmented Generation (RAG) pipeline that lets you upload any PDF and ask natural language questions about its content. Uses sentence embeddings and FAISS vector search to find relevant sections before passing them to Groq's LLM for answering.

**Skills demonstrated:** RAG, vector embeddings, FAISS similarity search, PyMuPDF, NLP, LLM grounding

**How to run:**
```bash
python pdf_qa_bot.py
```
When prompted, enter the full path to any PDF file on your computer.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| Groq API | LLM inference (llama-3.3-70b-versatile) |
| PyMuPDF (fitz) | PDF text extraction |
| Sentence Transformers | Text embeddings (all-MiniLM-L6-v2) |
| FAISS | Vector similarity search |
| Requests | HTTP API calls |
| python-dotenv | Secure API key management |

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-agent-projects.git
cd ai-agent-projects
```

### 2. Install dependencies
```bash
pip install groq python-dotenv requests pymupdf sentence-transformers faiss-cpu
```

### 3. Set up your API key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 4. Run any project
```bash
python my_first_agent_groq.py
```

---

## What I Learned

- How to call LLM APIs and handle responses
- How RAG (Retrieval Augmented Generation) works in practice
- Vector embeddings and semantic similarity search
- Secure API key management using environment variables
- Debugging real API errors (403, 404, rate limits, deprecated models)
- Building agent loops that run continuously until user exits

---

## Author

**Manaswi Konatham** — learning AI development.
