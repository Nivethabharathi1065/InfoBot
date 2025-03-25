# InfoBot
Retrieval-Augmented Generation (RAG)-based chatbot
[InfoBot ppt(document).pdf](https://github.com/user-attachments/files/19445640/InfoBot.ppt.document.pdf)


Retrieval-Augmented Generation (RAG)-based chatbot


Introduction
InfoBot is an AI-powered chatbot designed to efficiently retrieve and generate answers based on multiple document formats. This chatbot supports PDFs, DOCX, CSVs, images, videos, and audio files and leverages advanced machine learning techniques and APIs to ensure fast, accurate, and contextually relevant responses.
Key Features:
 Multi-format document processing
 Efficient retrieval & storage using FAISS
 Optimized model architecture without LangChain
 Interactive UI with Streamlit
Supports real-time, scalable knowledge retrieval

Introduction
InfoBot is an AI-powered chatbot designed to efficiently retrieve and generate answers based on multiple document formats. This chatbot supports PDFs, DOCX, CSVs, images, videos, and audio files and leverages advanced machine learning techniques and APIs to ensure fast, accurate, and contextually relevant responses.
Key Features:
Multi-format document processing
Efficient retrieval & storage using FAISS
Optimized model architecture without LangChain
Interactive UI with Streamlit
Supports real-time, scalable knowledge retrieval

What is RAG?
Retrieval-Augmented Generation (RAG) is an advanced approach that improves response generation by retrieving relevant document data before generating answers.
How It Works:
Retrieval: Searches for relevant information in stored documents using FAISS.
Generation: Uses AI models (Gemini API, Cohere API) to generate contextually appropriate answers.
Advantage: Ensures factual, high-accuracy responses without hallucination.



Technical Stack
Document Processing Techniques:
PDF: Extracted using pdfplumber
DOCX: Processed with python-docx
CSV: Structured data handled via Cohere API
Image OCR: Text extracted using EasyOCR
Audio/Video Transcription: Speech converted to text using Vosk

Model & APIs:
Gemini API → Handles PDFs, DOCX, images, and videos
Cohere API → Processes structured CSV data
Data Storage & Retrieval:
FAISS (Facebook AI Similarity Search) → Stores and retrieves document embeddings for fast similarity-based searches.
User Interface:
Streamlit → Provides an intuitive and interactive UI.
System Architecture
Step-by-Step Workflow:
User Uploads a Document (PDF, DOCX, CSV, Image, or Video/Audio).
Text Extraction: Extracts content using format-specific libraries.
Embedding Generation: Converts text into vector embeddings using Sentence Transformers.
Storage in FAISS: Stores embeddings for fast retrieval.
User Queries the System: Searches the FAISS index for relevant data.
Response Generation: Gemini or Cohere API generates answers based on retrieved content.
Output Displayed on Streamlit UI.



Conclusion
The InfoBot RAG-based chatbot is an efficient, scalable, and intelligent solution for answering document-based queries. By leveraging FAISS, LLM APIs, and advanced text extraction techniques, it ensures accurate and contextually relevant responses across different formats.
High Processing Speed – Optimized retrieval using vector search.
Improved Accuracy – AI-driven response generation based on real-time document search.
Scalability – Adaptable across industries and knowledge bases.
User-Friendly UI – Simple document upload and query processing via Streamlit.
Future Enhancements
Better Indexing Strategies – Improve FAISS-based search accuracy
Hybrid Search Models – Combine keyword + semantic search for refined retrieval.
Multimodal Support – Extend capabilities to extract tables, charts, and handwritten text.
Cloud Deployment – Scale the chatbot for enterprise-level usage.
This chatbot proves to be a highly efficient, scalable, and reliable tool for knowledge retrieval, offering real-time, accurate responses across multiple document formats. 




MY REPOSITORY LINK:
https://github.com/Nivethabharathi1065/InfoBot

MAIN CODE:
import streamlit as st 
import pdfplumber 
from docx import Document 
import pandas as pd 
import cv2 
import faiss 
import numpy as np 
import cohere # For CSV Processing 
import google.generativeai as genai # For PDF, DOCX, Image, Audio, Video 
import os 
import tempfile 
from vosk import Model, KaldiRecognizer 
from sentence_transformers import SentenceTransformer 
import wave 
import easyocr # Replacing Tesseract with EasyOCR 
# Configure API Keys 
COHERE_API_KEY = "2xG3kIglq38EBSeQyzfLcaUny2e2sHYdpbhpXLmo" # Cohere for CSV 
GEMINI_API_KEY = "AIzaSyAUZG1FZp6Ti7n8t9MqRa1xb26L-l3b6pw" # Gemini for all except CSV 
genai.configure(api_key=GEMINI_API_KEY) 
co = cohere.Client(COHERE_API_KEY) 
# Initialize FAISS 
embedding_model = SentenceTransformer("all-MiniLM-L6-v2") 
embedding_dim = embedding_model.get_sentence_embedding_dimension() 
index = faiss.IndexFlatL2(embedding_dim) 
document_store = {} 
# EasyOCR Reader Initialization 
ocr_reader = easyocr.Reader(["en"]) # Supports multiple languages 
# Extract text from different file types 
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
return df.to_string(index=False) # Maintain tabular structure 
# Using EasyOCR Instead of Tesseract 
def extract_text_from_image(uploaded_file): 
try: 
with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file: 
temp_file.write(uploaded_file.read()) 
temp_path = temp_file.name 
result = ocr_reader.readtext(temp_path, detail=0) # Extract text 
return "\n".join(result) if result else "No text found in the image." 
except Exception as e: 
return f"Image OCR Error: {str(e)}" 
# Fix for video/audio processing (Handles Vosk path properly) 
def transcribe_audio_with_vosk(audio_path): 
try: 
model_path = "vosk-model-small-en-us-0.15" # Ensure this model exists 
if not os.path.exists(model_path): 
return "Vosk model not found. Please download and set the correct path." 
wf = wave.open(audio_path, "rb") 
rec = KaldiRecognizer(Model(model_path), wf.getframerate()) 
text = "" 
while True: 
data = wf.readframes(4000) 
if len(data) == 0: 
break 
if rec.AcceptWaveform(data): 
text += rec.Result() 
return text 
except Exception as e: 
return f"Audio Processing Error: {str(e)}" 
# Store extracted text in FAISS 
def store_text_in_faiss(text, doc_id): 
embedding = embedding_model.encode(text).astype(np.float32) 
index.add(np.array([embedding])) 
document_store[index.ntotal - 1] = text 
# Search relevant text from FAISS (Retrieves only top 3 matches & limits words) 
def search_in_faiss(query): 
query_embedding = embedding_model.encode(query).astype(np.float32) 
distances, indices = index.search(np.array([query_embedding]), k=3) # Retrieve top 3 matches 
results = [] 
for idx in indices[0]: 
if idx in document_store: 
results.append(document_store[idx]) 
return [" ".join(result.split()[:100]) for result in results] # Limit each result to 100 words 
# Generate response using Gemini (for all except CSV) 
def generate_response_gemini(prompt): 
try: 
model = genai.GenerativeModel("gemini-1.5-pro") 
response = model.generate_content(prompt) 
return response.text.strip() 
except Exception as e: 
return f"Gemini API Error: {str(e)}" 
# Generate response using Cohere (for CSV) - Prevents token limit errors 
def generate_response_cohere(prompt): 
MAX_TOKENS = 3500 # Keep token usage safe 
if len(prompt.split()) > MAX_TOKENS: 
prompt = "Summarize and answer concisely:\n\n" + " ".join(prompt.split()[:2000]) # Truncate safely 
try: 
response = co.generate(model="command", prompt=prompt, max_tokens=150) 
return response.generations[0].text.strip() 
except Exception as e: 
return f"Cohere API Error: {str(e)}" 
# Streamlit UI 
st.title("PDF: Gemini | CSV: Cohere | FAISS Retrieval Chatbot") 
uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, CSV, Image, Audio, Video)", type=["pdf", "docx", "csv", "jpg", "png", "mp4", "wav"]) 
if uploaded_file is not None: 
file_type = uploaded_file.type 
text_data = "" 
use_cohere = False # Default to Gemini 
if "pdf" in file_type: 
text_data = extract_text_from_pdf(uploaded_file) 
elif "word" in file_type: 
text_data = extract_text_from_docx(uploaded_file) 
elif "csv" in file_type: 
text_data = extract_text_from_csv(uploaded_file) 
use_cohere = True # Use Cohere for CSV 
elif "image" in file_type: 
text_data = extract_text_from_image(uploaded_file) # Using EasyOCR 
elif "audio" in file_type or "video" in file_type: 
with open("temp_audio.wav", "wb") as f: 
f.write(uploaded_file.read()) 
text_data = transcribe_audio_with_vosk("temp_audio.wav") 
store_text_in_faiss(text_data, uploaded_file.name) 
st.success(f"{uploaded_file.name} has been processed and stored!") 
st.subheader("Ask a Question!") 
query = st.text_input("Enter your question:") 
if query: 
retrieved_texts = search_in_faiss(query) 
prompt = f"Using the following information:\n\n{retrieved_texts}\n\nAnswer the question: {query}" 
response = generate_response_cohere(prompt) if use_cohere else generate_response_gemini(prompt) 
st.write(response) 

OUTPUTS
PDF RETERIVAL

![Screenshot 2025-03-25 124942](https://github.com/user-attachments/assets/875c09fc-a98d-44e3-b74a-10c44dfaaac6)

CSV RETERIVAL

![Screenshot 2025-03-25 130833](https://github.com/user-attachments/assets/9d4c4050-b264-44e8-a71f-6e2c3b1bd4f8)

IMAGE RETERIVAL

![Screenshot 2025-03-25 131251](https://github.com/user-attachments/assets/3f893445-235a-4fd9-ac11-52aa6f6c6b28)

VIDEO RETERIVAL

![Screenshot 2025-03-25 144830](https://github.com/user-attachments/assets/60ca457b-0994-4483-ac85-3fc9ef66ad94)
