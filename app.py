import os
import pickle
import faiss
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

from fastapi.responses import RedirectResponse, HTMLResponse

# --- Load environment variables ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

# --- Configure Gemini ---
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-pro")

# --- FastAPI App ---
app = FastAPI(
    title="Document Q&A RAG System",
    description="An API for uploading PDFs and querying them using a Gemini LLM.",
)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/upload", response_class=HTMLResponse)
async def upload_page():
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload Document & Ask Questions</title>
        <style>
            body {{ font-family: sans-serif; margin: 2rem; background-color: #f4f7f6; color: #333; }}
            h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
            div {{ margin-bottom: 15px; }}
            input[type="file"], textarea {{
                width: calc(100% - 20px);
                padding: 10px;
                margin-top: 5px;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-sizing: border-box;
            }}
            button {{
                background-color: #3498db;
                color: white;
                padding: 10px 15px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                margin-top: 10px;
                width: 100%;
            }}
            button:hover {{
                background-color: #2980b9;
            }}
            #response {{
                white-space: pre-wrap;
                background: #eef1f3;
                padding: 1em;
                margin-top: 1em;
                border-radius: 5px;
                border: 1px solid #d1d8dc;
                min-height: 50px; /* Ensure it's visible even when empty */
            }}
            .status-message {{
                font-weight: bold;
                color: #27ae60; /* Green for success */
                margin-top: 10px;
            }}
            .error-message {{
                font-weight: bold;
                color: #e74c3c; /* Red for error */
                margin-top: 10px;
            }}
        </style>
    </head>
    <body>
        <h1>📚 Document Q&A RAG System</h1>

        <div id="statusIndicator">
            <p><strong>Knowledge Base Status:</strong> <span id="kbStatus">Loading...</span></p>
        </div>

        <h2>📄 Upload PDF Document</h2>
        <div>
            <input type="file" id="fileInput" accept=".pdf" />
            <button onclick="uploadFile()">Upload PDF</button>
        </div>

        <h2>🤖 Ask a Question</h2>
        <div>
            <textarea id="query" rows="4" placeholder="e.g. Will my surgery be approved? What are the terms for dental coverage?"></textarea>
            <button onclick="sendQuery()">Ask Question</button>
        </div>

        <h3>Response:</h3>
        <div id="response"></div>

        <script>
            // Function to check and display KB status
            async function checkKBStatus() {{
                const statusElem = document.getElementById("kbStatus");
                try {{
                    const res = await fetch("/status"); // New endpoint
                    const json = await res.json();
                    if (json.faiss_index_loaded) {{
                        statusElem.textContent = `Loaded (${{json.num_chunks}} chunks)`;
                        statusElem.style.color = '#27ae60';
                    }} else {{
                        statusElem.textContent = "Empty (Please upload documents)";
                        statusElem.style.color = '#f39c12';
                    }}
                }} catch (error) {{
                    statusElem.textContent = "Error checking status";
                    statusElem.style.color = '#e74c3c';
                    console.error("Error checking KB status:", error);
                }}
            }}

            // Call status check on page load
            window.onload = checkKBStatus;


            async function uploadFile() {{
                const fileInput = document.getElementById("fileInput");
                const file = fileInput.files[0];
                const responseDiv = document.getElementById("response");
                responseDiv.className = ''; // Clear previous status classes
                responseDiv.innerText = "Uploading and processing...";

                if (!file) {{
                    responseDiv.innerText = "Please select a PDF file.";
                    responseDiv.className = 'error-message';
                    return;
                }}

                if (file.type !== "application/pdf") {{
                    responseDiv.innerText = "Only PDF files are supported.";
                    responseDiv.className = 'error-message';
                    return;
                }}

                const formData = new FormData();
                formData.append("file", file);

                try {{
                    const res = await fetch("/ingest", {{
                        method: "POST",
                        body: formData,
                    }});

                    const json = await res.json();
                    if (res.ok) {{
                        responseDiv.innerText = `✅ Document "${{json.filename}}" ingested successfully. Added ${{json.chunks_added}} chunks.`;
                        responseDiv.className = 'status-message';
                        fileInput.value = ''; // Clear file input
                        checkKBStatus(); // Update KB status after upload
                    }} else {{
                        responseDiv.innerText = `❌ Error: ${{json.detail || "An unknown error occurred."}}`;
                        responseDiv.className = 'error-message';
                    }}
                }} catch (error) {{
                    responseDiv.innerText = `❌ Network Error: ${{error.message}}`;
                    responseDiv.className = 'error-message';
                    console.error("Upload error:", error);
                }}
            }} // Added the missing closing curly brace for uploadFile()

            async function sendQuery() {{
                const query = document.getElementById("query").value;
                const responseDiv = document.getElementById("response");
                responseDiv.className = ''; // Clear previous status classes
                responseDiv.innerText = "Thinking...";

                if (!query) {{
                    responseDiv.innerText = "Please enter a question.";
                    responseDiv.className = 'error-message';
                    return;
                }}

                try {{
                    const res = await fetch("/query", {{
                        method: "POST",
                        headers: {{ "Content-Type": "application/json" }},
                        body: JSON.stringify({{ query }}),
                    }});

                    const json = await res.json();
                    if (res.ok) {{
                        if (json.answer) {{
                            responseDiv.innerText = `🤖 Answer: ${{json.answer}}`;
                            responseDiv.className = 'status-message';
                        }} else if (json.error) {{
                            responseDiv.innerText = `⚠️ LLM Error: ${{json.error}}`;
                            responseDiv.className = 'error-message';
                        }}
                    }} else {{
                        responseDiv.innerText = `❌ Error: ${{json.detail || "An unknown error occurred."}}`;
                        responseDiv.className = 'error-message';
                    }}
                }} catch (error) {{
                    responseDiv.innerText = `❌ Network Error: ${{error.message}}`;
                    responseDiv.className = 'error-message';
                    console.error("Query error:", error);
                }}
            }}
        </script>
    </body>
    </html>
    """

# --- Directories ---
DATA_DIR = "ingested_data"
UPLOAD_DIR = "uploaded_docs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
CHUNKS_MAP_PATH = os.path.join(DATA_DIR, "chunks_map.pkl")

# --- Global State ---
faiss_index = None
chunks_data = []

# --- Embedding Model ---
print("Loading embedding model...")
# Ensure you have 'sentence-transformers' installed: pip install sentence-transformers
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding model loaded.")

# --- Load Index on Startup ---
def load_knowledge_base():
    global faiss_index, chunks_data
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_MAP_PATH):
        print("Loading FAISS index and chunks...")
        try:
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            with open(CHUNKS_MAP_PATH, "rb") as f:
                chunks_data = pickle.load(f)
            print(f"Knowledge base loaded with {len(chunks_data)} chunks.")
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            faiss_index = None # Reset if loading fails
            chunks_data = []
    else:
        print("Knowledge base not found. Please upload documents to build it.")

@app.on_event("startup")
async def startup_event():
    load_knowledge_base()

# --- New Endpoint to check Knowledge Base Status ---
@app.get("/status", summary="Check the status of the knowledge base")
async def get_kb_status():
    global faiss_index, chunks_data
    return {
        "faiss_index_loaded": faiss_index is not None,
        "num_chunks": len(chunks_data)
    }


# --- LLM Function ---
def call_llm(prompt: str):
    try:
        response = gemini_model.generate_content(prompt)
        # Check if response has text attribute before accessing
        if hasattr(response, 'text') and response.text:
            return {"answer": response.text.strip()}
        else:
            return {"error": "LLM returned no content."}
    except Exception as e:
        print(f"Error calling LLM: {e}") # Log error for debugging
        return {"error": f"Failed to get response from LLM: {e}"}

# --- Ingest Endpoint ---
@app.post("/ingest", summary="Ingest and process a document")
async def ingest_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")


    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_chunks = splitter.split_documents(documents)
    except Exception as e:
        # Clean up the uploaded file if processing fails
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    if not split_chunks:
        # Clean up the uploaded file if no text is found
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail="No text found in PDF or document is empty.")

    global chunks_data
    current_chunks = [
        {"text": chunk.page_content, "metadata": {"source": file.filename, "chunk_num": i}}
        for i, chunk in enumerate(split_chunks)
    ]

    # Before extending, get the current number of chunks to adjust indices if needed
    start_index = len(chunks_data)
    chunks_data.extend(current_chunks)

    texts = [chunk["text"] for chunk in current_chunks]
    try:
        embeddings = EMBEDDING_MODEL.embed_documents(texts)
    except Exception as e:
        # Rollback chunks if embedding fails
        chunks_data = chunks_data[:start_index]
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {e}")


    global faiss_index
    if faiss_index is None:
        dim = len(embeddings[0])
        faiss_index = faiss.IndexFlatL2(dim)

    try:
        faiss_index.add(np.array(embeddings).astype('float32')) # Ensure embeddings are float32 for FAISS
        faiss.write_index(faiss_index, FAISS_INDEX_PATH)
        with open(CHUNKS_MAP_PATH, "wb") as f:
            pickle.dump(chunks_data, f)
    except Exception as e:
        # More robust rollback might be needed here for production,
        # but for this example, we'll just log and raise
        print(f"Error saving FAISS index or chunks data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to persist knowledge base: {e}")


    return {"status": "success", "filename": file.filename, "chunks_added": len(current_chunks)}

# --- Query Model ---
class QueryRequest(BaseModel):
    query: str

# --- Query Endpoint ---
@app.post("/query", summary="Ask a question to the documents")
async def handle_query(request: QueryRequest):
    global faiss_index, chunks_data

    if faiss_index is None or not chunks_data:
        raise HTTPException(status_code=503, detail="Knowledge base is empty. Please ingest documents first.")

    try:
        query_vector = EMBEDDING_MODEL.embed_query(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed query: {e}")

    try:
        # Ensure the query vector is 2D for faiss.search
        query_vector_2d = np.array([query_vector]).astype('float32') # FAISS expects float32
        distances, indices = faiss_index.search(query_vector_2d, 5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform FAISS search: {e}")


    retrieved_context = ""
    # Filter out invalid indices if any (can happen with corrupted index or data mismatch)
    valid_indices = [idx for idx in indices[0] if 0 <= idx < len(chunks_data)]
    if not valid_indices:
        print("Warning: No valid chunks retrieved for the query.")
        # Fallback: maybe allow LLM to respond without context or give specific error
        # For now, it will proceed with empty context
        pass


    for idx in valid_indices:
        chunk_info = chunks_data[idx]
        retrieved_context += (
            f"Source: {chunk_info['metadata']['source']}, Chunk {chunk_info['metadata']['chunk_num']}\n"
            f"Content: {chunk_info['text']}\n---\n"
        )

    # If no context is retrieved, inform the user or adjust prompt
    if not retrieved_context:
        prompt = f"""
        You are an AI assistant. The user asked: "{request.query}".
        I could not find any relevant information in the documents. Please state that you don't have enough information to answer.
        """
    else:
        prompt = f"""
You are an AI assistant for insurance claim analysis.
Respond based ONLY on the context below:

--- CONTEXT ---
{retrieved_context}
--- END CONTEXT ---

User's Query:
"{request.query}"

Respond in a single sentence:
- If covered: "Yes, [brief reason]."
- If not covered: "No, [brief reason]."
- If the answer is not in the context: "The provided context does not contain enough information to answer."
        """

    llm_response = call_llm(prompt)
    if "error" in llm_response:
        raise HTTPException(status_code=500, detail=llm_response["error"])

    return llm_response