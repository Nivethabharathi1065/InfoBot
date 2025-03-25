import streamlit as st
import pdfplumber
from docx import Document
import pandas as pd
import pytesseract
import cv2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import cohere  # For CSV Processing
import google.generativeai as genai  # For PDF, DOCX, Image, Audio, Video
import os
from vosk import Model, KaldiRecognizer
import wave

# 👉 Configure API Keys
COHERE_API_KEY = "2xG3kIglq38EBSeQyzfLcaUny2e2sHYdpbhpXLmo"  # Cohere for CSV
GEMINI_API_KEY = "AIzaSyAUZG1FZp6Ti7n8t9MqRa1xb26L-l3b6pw"  # Gemini for all except CSV

genai.configure(api_key=GEMINI_API_KEY)
co = cohere.Client(COHERE_API_KEY)

# 👉 Initialize FAISS
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = embedding_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(embedding_dim)
document_store = {}

# 👉 Extract text from different file types
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df.to_string(index=False)  # Maintain tabular structure

def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    return pytesseract.image_to_string(image)

def transcribe_audio_with_vosk(audio_path):
    wf = wave.open(audio_path, "rb")
    rec = KaldiRecognizer(Model("vosk-model-small-en-us-0.15"), wf.getframerate())
    text = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            text += rec.Result()
    return text

# 👉 Store extracted text in FAISS
def store_text_in_faiss(text, doc_id):
    embedding = embedding_model.encode(text).astype(np.float32)
    index.add(np.array([embedding]))
    document_store[index.ntotal - 1] = text

# 👉 Search relevant text from FAISS (Retrieves only top 3 matches & limits words)
def search_in_faiss(query):
    query_embedding = embedding_model.encode(query).astype(np.float32)
    distances, indices = index.search(np.array([query_embedding]), k=3)  # Retrieve top 3 matches
    
    results = []
    for idx in indices[0]:  
        if idx in document_store:
            results.append(document_store[idx])  

    return [" ".join(result.split()[:100]) for result in results]  # Limit each result to 100 words

# 👉 Generate response using Gemini (for all except CSV)
def generate_response_gemini(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Gemini API Error: {str(e)}"

# 👉 Generate response using Cohere (for CSV) - Prevents token limit errors
def generate_response_cohere(prompt):
    MAX_TOKENS = 3500  # Keep token usage safe
    
    if len(prompt.split()) > MAX_TOKENS:
        prompt = "Summarize and answer concisely:\n\n" + " ".join(prompt.split()[:2000])  # Truncate safely

    try:
        response = co.generate(model="command", prompt=prompt, max_tokens=150)
        return response.generations[0].text.strip()
    except Exception as e:
        return f"❌ Cohere API Error: {str(e)}"

# 👉 Streamlit UI
st.title("🔹 PDF: Gemini | CSV: Cohere | FAISS Retrieval Chatbot")

uploaded_file = st.file_uploader("📂 Upload a document (PDF, DOCX, CSV, Image, Audio, Video)", type=["pdf", "docx", "csv", "jpg", "png", "mp4", "wav"])

if uploaded_file is not None:
    file_type = uploaded_file.type
    text_data = ""
    use_cohere = False  # Default to Gemini
    
    if "pdf" in file_type:
        text_data = extract_text_from_pdf(uploaded_file)
    elif "word" in file_type:
        text_data = extract_text_from_docx(uploaded_file)
    elif "csv" in file_type:
        text_data = extract_text_from_csv(uploaded_file)
        use_cohere = True  # Use Cohere for CSV
    elif "image" in file_type:
        text_data = extract_text_from_image(uploaded_file)
    elif "audio" in file_type or "video" in file_type:
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.read())
        text_data = transcribe_audio_with_vosk("temp_audio.wav")
    
    store_text_in_faiss(text_data, uploaded_file.name)
    st.success(f"✅ {uploaded_file.name} has been processed and stored!")

st.subheader("💬 Ask a Question!")
query = st.text_input("🔍 Enter your question:")

if query:
    retrieved_texts = search_in_faiss(query)
    prompt = f"Using the following information:\n\n{retrieved_texts}\n\nAnswer the question: {query}"
    
    response = generate_response_cohere(prompt) if use_cohere else generate_response_gemini(prompt)
    st.write(response)