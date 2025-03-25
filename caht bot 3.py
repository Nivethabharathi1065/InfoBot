import streamlit as st
import pdfplumber
from docx import Document
import pandas as pd
import pytesseract
import cv2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
from vosk import Model, KaldiRecognizer
import wave

# 👉 Step 1: Configure Google Gemini API
GEMINI_API_KEY = "AIzaSyAUZG1FZp6Ti7n8t9MqRa1xb26L-l3b6pw"  # Replace with your actual Gemini API key
genai.configure(api_key=GEMINI_API_KEY)

# 👉 Step 2: Initialize FAISS (vector storage)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = embedding_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(embedding_dim)
document_store = {}

# 👉 Step 3: Functions for extracting text from different file types
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
    return "\n".join(df.astype(str).values.flatten())

def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    text = pytesseract.image_to_string(image)
    return text

# Load Vosk model for speech-to-text
vosk_model = Model("C:/Users/pnb10/Desktop/rag/vosk_model/vosk-model-small-en-us-0.15")

def extract_text_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    text = ""

    frame_interval = 10  # Process every 10th frame
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            text += pytesseract.image_to_string(frame) + "\n"
        count += 1

    # Convert audio to text using Vosk
    text += transcribe_audio_with_vosk(video_path)
    cap.release()
    return text

def transcribe_audio_with_vosk(audio_path):
    wf = wave.open(audio_path, "rb")
    rec = KaldiRecognizer(vosk_model, wf.getframerate())

    text = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            text += rec.Result()

    return text

# 👉 Step 4: Store extracted text in FAISS
def store_text_in_faiss(text, doc_id):
    embedding = embedding_model.encode(text).astype(np.float32)
    index.add(np.array([embedding]))
    document_store[index.ntotal - 1] = text

# 👉 Step 5: Search relevant text from FAISS
def search_in_faiss(query):
    query_embedding = embedding_model.encode(query).astype(np.float32)
    distances, indices = index.search(np.array([query_embedding]), k=5)
    results = [document_store[idx] for idx in indices[0] if idx in document_store]
    return results

# 👉 Step 6: Generate response using Gemini API
def generate_response(prompt):
    model = genai.GenerativeModel("gemini-1.5-pro")  # Use the correct model name
    response = model.generate_content(prompt)
    return response.text  # Return the generated text

# 👉 Step 7: Build Streamlit Chatbot UI
st.title("🦾 RAG Chatbot with OCR & AI (FAISS + Vosk)")

uploaded_file = st.file_uploader("📂 Upload a document, image, video, or CSV", type=["pdf", "docx", "csv", "jpg", "png", "mp4", "wav"])

if uploaded_file is not None:
    file_type = uploaded_file.type
    text_data = ""

    # Process uploaded file
    if "pdf" in file_type:
        text_data = extract_text_from_pdf(uploaded_file)
    elif "word" in file_type:
        text_data = extract_text_from_docx(uploaded_file)
    elif "csv" in file_type:
        text_data = extract_text_from_csv(uploaded_file)
    elif "image" in file_type:
        text_data = extract_text_from_image(uploaded_file)
    elif "video" in file_type or "audio" in file_type:
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.read())
        text_data = transcribe_audio_with_vosk("temp_audio.wav")

    # Store extracted text in FAISS
    store_text_in_faiss(text_data, uploaded_file.name)
    st.success(f"✅ {uploaded_file.name} has been processed and stored!")

st.subheader("💬 Ask me anything!")
query = st.text_input("🔍 Enter your question:")

if query:
    retrieved_texts = search_in_faiss(query)  # Retrieve relevant info
    prompt = f"Using the following information:\n\n{retrieved_texts}\n\nAnswer the question: {query}"
    response = generate_response(prompt)  # Generate AI response
    st.write(response)