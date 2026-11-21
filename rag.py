import os
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "pdf_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("Loading local free LLM...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)

def split_text(text, chunk_size=800, overlap=100):
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks

def get_collection():
    client_db = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client_db.get_or_create_collection(name=COLLECTION_NAME)
    return collection

def index_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        raise ValueError("No readable text was found in the PDF.")
    chunks = split_text(text)
    collection = get_collection()
    
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)
        
    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"{os.path.basename(pdf_path)}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": pdf_path, "chunk": i} for i in range(len(chunks))]
    
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )
    return len(chunks)

def search_documents(question, n_results=3):
    collection = get_collection()
    question_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=n_results
    )
    return results

def generate_answer(question, documents):
    context = "\n\n".join(documents)
    prompt = f"Read the context below and answer the question in a few words.\n\nContext: {context}\n\nQuestion: {question}"
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=100)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer