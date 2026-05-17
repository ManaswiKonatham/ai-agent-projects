# pdf_qa_bot.py
# Ask questions about any PDF using RAG (Retrieval Augmented Generation)

import os
import fitz  # pymupdf
import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

# --- Setup ---
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, free model

# --- Step 1: Read the PDF ---
def read_pdf(pdf_path):
    """Extract all text from a PDF file"""
    print(f"Reading PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    print(f"Read {len(doc)} pages successfully.")
    return text

# --- Step 2: Split into chunks ---
def split_into_chunks(text, chunk_size=500):
    """Split text into small overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - 50):  # 50 word overlap
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

# --- Step 3: Create embeddings and index ---
def build_index(chunks):
    """Convert chunks to embeddings and store in FAISS index"""
    print("Building search index (this may take a moment)...")
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    print("Index built successfully!")
    return index, embeddings

# --- Step 4: Find relevant chunks for a question ---
def search_chunks(question, chunks, index, top_k=3):
    """Find the most relevant chunks for a question"""
    question_embedding = embedder.encode([question]).astype("float32")
    distances, indices = index.search(question_embedding, top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

# --- Step 5: Answer with Groq ---
def answer_question(question, relevant_chunks):
    """Use Groq to answer the question based on relevant chunks"""
    context = "\n\n".join(relevant_chunks)

    prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided document content.
If the answer is not in the document, say "I couldn't find that in the document."

Document content:
{context}

Question: {question}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- Main ---
def main():
    print("=" * 50)
    print("PDF Q&A Bot — Powered by RAG + Groq")
    print("=" * 50)

    # Get PDF path from user
    pdf_path = input("\nEnter the path to your PDF file: ").strip()

    if not os.path.exists(pdf_path):
        print("File not found! Make sure the path is correct.")
        return

    # Build the RAG pipeline
    text = read_pdf(pdf_path)
    chunks = split_into_chunks(text)
    index, embeddings = build_index(chunks)

    print("\nReady! Ask me anything about your PDF.")
    print("Type 'quit' to exit.\n")

    # Q&A loop
    while True:
        question = input("Your question: ").strip()

        if question.lower() == "quit":
            print("Goodbye!")
            break

        if not question:
            continue

        print("\nSearching document...")
        relevant_chunks = search_chunks(question, chunks, index)

        print("Generating answer...\n")
        answer = answer_question(question, relevant_chunks)

        print("Answer:", answer)
        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    main()
