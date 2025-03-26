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



OUTPUTS
PDF RETERIVAL

![Screenshot 2025-03-25 124942](https://github.com/user-attachments/assets/875c09fc-a98d-44e3-b74a-10c44dfaaac6)

CSV RETERIVAL

![Screenshot 2025-03-25 130833](https://github.com/user-attachments/assets/9d4c4050-b264-44e8-a71f-6e2c3b1bd4f8)

IMAGE RETERIVAL

![Screenshot 2025-03-25 131251](https://github.com/user-attachments/assets/3f893445-235a-4fd9-ac11-52aa6f6c6b28)

VIDEO RETERIVAL

![Screenshot 2025-03-25 144830](https://github.com/user-attachments/assets/60ca457b-0994-4483-ac85-3fc9ef66ad94)
