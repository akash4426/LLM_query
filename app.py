import os
import pickle
import numpy as np
import re
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, status
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from together import Together

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from fastapi.responses import RedirectResponse, HTMLResponse

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str

class HackRxDocument(BaseModel):
    url: str
    filename: str

class HackRxRequest(BaseModel):
    documents: Dict[str, List[str]]
    questions: Dict[str, List[str]]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- Environment Variables ---
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY not found in .env")

HACKRX_API_KEY = os.getenv("HACKRX_API_KEY")
if not HACKRX_API_KEY:
    print("WARNING: HACKRX_API_KEY not found in .env. The /hackrx/run endpoint will not be secured.")

# --- Configure Together AI ---
together_client = Together(api_key=TOGETHER_API_KEY)
TOGETHER_LLM_MODEL = "meta-llama/Llama-3-8b-chat-hf"

# --- FastAPI App ---
app = FastAPI(
    title="Document Q&A RAG System",
    description="An API for uploading PDFs and querying them using a Together AI LLM.",
)

# --- FIX APPLIED HERE ---
# Changed from @app.get to @app.api_route to handle health checks from hosting platforms.
@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")
# -------------------------

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
                min-height: 50px;
            }}
            .status-message {{ font-weight: bold; color: #27ae60; margin-top: 10px; }}
            .error-message {{ font-weight: bold; color: #e74c3c; margin-top: 10px; }}
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
            async function checkKBStatus() {{
                const statusElem = document.getElementById("kbStatus");
                try {{
                    const res = await fetch("/status");
                    const json = await res.json();
                    if (json.vector_db_loaded) {{
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
            window.onload = checkKBStatus;
            async function uploadFile() {{
                const fileInput = document.getElementById("fileInput");
                const file = fileInput.files[0];
                const responseDiv = document.getElementById("response");
                responseDiv.className = '';
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
                        fileInput.value = '';
                        checkKBStatus();
                    }} else {{
                        responseDiv.innerText = `❌ Error: ${{json.detail || "An unknown error occurred."}}`;
                        responseDiv.className = 'error-message';
                    }}
                }} catch (error) {{
                    responseDiv.innerText = `❌ Network Error: ${{error.message}}`;
                    responseDiv.className = 'error-message';
                    console.error("Upload error:", error);
                }}
            }}
            async function sendQuery() {{
                const query = document.getElementById("query").value;
                const responseDiv = document.getElementById("response");
                responseDiv.className = '';
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

# --- Directories & Global State ---
DATA_DIR = "ingested_data"
UPLOAD_DIR = "uploaded_docs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
vector_db: Optional[Chroma] = None

# --- Models & Startup ---
print("Loading embedding model...")
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
print("Embedding model loaded.")

def load_knowledge_base():
    global vector_db
    print("Attempting to load ChromaDB knowledge base...")
    try:
        vector_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=EMBEDDING_MODEL)
        if vector_db._collection.count() > 0:
            print(f"ChromaDB knowledge base loaded with {vector_db._collection.count()} documents.")
        else:
            print("ChromaDB initialized but no documents found. Please upload documents.")
    except Exception as e:
        print(f"Error loading ChromaDB knowledge base: {e}")
        vector_db = None

@app.on_event("startup")
async def startup_event():
    load_knowledge_base()

# --- API Endpoints ---
@app.get("/status", summary="Check the status of the knowledge base")
async def get_kb_status():
    global vector_db
    num_chunks = 0
    vector_db_loaded = False
    if vector_db:
        try:
            num_chunks = vector_db._collection.count()
            vector_db_loaded = True
        except Exception as e:
            print(f"Error getting ChromaDB count: {e}")
            vector_db_loaded = False
    return {"vector_db_loaded": vector_db_loaded, "num_chunks": num_chunks}

def call_llm(prompt: str):
    try:
        print(f"DEBUG: Calling LLM with model: {TOGETHER_LLM_MODEL} (Together AI)")
        messages = [{"role": "user", "content": prompt}]
        response = together_client.chat.completions.create(
            model=TOGETHER_LLM_MODEL,
            messages=messages,
            max_tokens=500,
            temperature=0.0
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            return {"answer": response.choices[0].message.content.strip()}
        else:
            return {"error": "Together AI LLM returned no content."}
    except Exception as e:
        print(f"Error calling Together AI LLM: {e}")
        return {"error": f"Failed to get response from Together AI LLM: {e}"}

def clean_ocr_text(text: str) -> str:
    cleaned_text = re.sub(r'(?<=\w)\s(?=\w)', '', text)
    cleaned_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned_text)
    cleaned_text = re.sub(r'(\D)(\d)', r'\1 \2', cleaned_text)
    cleaned_text = re.sub(r'(\d)(\D)', r'\1 \2', cleaned_text)
    specific_replacements = {
        'Contractmeans': 'Contract means',
        'Policyandthe': 'Policy and the',
        'Co-paymentmeans': 'Co-payment means',
        'CumulativeBonusmeans': 'Cumulative Bonus means',
        'IncaseofanygrievancerelatedtothePolicy, theinsuredpersonmaysubmitinwritingtothePolicyIssuingOfficeorGrievancecellatRegionalOfficeoftheCompanyforredressal.': 'In case of any grievance related to the Policy, the insured person may submit in writing to the Policy Issuing Office or Grievance cell at Regional Office of the Company for redressal.',
        'Ifthegrievanceremainsunaddressed, theinsuredpersonmaysubmitinwritingtothePolicyIssuingOfficeorGrievancecellatRegionalOfficeoftheCompanyforredressal.': 'If the grievance remains unaddressed, the insured person may submit in writing to the Policy Issuing Office or Grievance cell at Regional Office of the Company for redressal.',
        'Formoreinformationongrievancemechanism, andtodownloadgrievanceform, visitour': 'For more information on grievance mechanism, and to download grievance form, visit our',
        'CustomerRelationshipManagementDept\.': 'Customer Relationship Management Dept.',
        'NationalInsuranceCompanyLimited, PremisesNo\. 18 -0374, Plotno\. CBD -81, New Town, Kolkata - 700156, email: customer\.relations@nic\.co\.in, griho@nic\.co\.in': 'National Insurance Company Limited, Premises No. 18-0374, Plot no. CBD-81, New Town, Kolkata - 700156, email: customer.relations@nic.co.in, griho@nic.co.in',
        'REDRESSALOFGRIEVANCE': 'REDRESSAL OF GRIEVANCE'
    }
    for old, new in specific_replacements.items():
        cleaned_text = re.sub(old, new, cleaned_text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', cleaned_text).strip()

def remove_boilerplate_text(text: str) -> str:
    patterns = [
        r'Page\s*\d+\s*of\s*\d+',
        r'NATIONAL INSURANCE COMPANY LIMITED',
        r'Policy No\.:\s*[\w-]+',
        r'UIN:\s*[\w-]+',
    ]
    cleaned_text = text
    for pattern in patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', cleaned_text).strip()

@app.post("/ingest", summary="Ingest and process a document")
async def ingest_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        processed_documents = []
        for doc in documents:
            final_content = remove_boilerplate_text(clean_ocr_text(doc.page_content))
            if final_content.strip():
                processed_documents.append(Document(page_content=final_content, metadata=doc.metadata))
        if not processed_documents:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="No meaningful text found in PDF after cleaning.")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
        split_chunks = splitter.split_documents(processed_documents)
        if not split_chunks:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="No text found in PDF after chunking.")
        
        global vector_db
        if vector_db is None:
            load_knowledge_base()
            if vector_db is None:
                raise HTTPException(status_code=500, detail="Failed to initialize vector database.")
        
        docs_to_add = []
        for i, chunk in enumerate(split_chunks):
            chunk.metadata["source"] = file.filename
            chunk.metadata["chunk_num"] = i
            docs_to_add.append(chunk)
        vector_db.add_documents(docs_to_add)
        return {"status": "success", "filename": file.filename, "chunks_added": len(split_chunks)}
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

@app.post("/query", summary="Ask a question to the documents")
async def handle_query(request: QueryRequest):
    global vector_db
    if vector_db is None or vector_db._collection.count() == 0:
        raise HTTPException(status_code=503, detail="Knowledge base is empty. Please ingest documents first.")
    try:
        retrieved_docs = vector_db.similarity_search(request.query, k=15)
        retrieved_context = ""
        if retrieved_docs:
            for doc in retrieved_docs:
                source_info = doc.metadata.get("source", "Unknown Source")
                chunk_num_info = doc.metadata.get("chunk_num", "N/A")
                retrieved_context += f"Source: {source_info}, Chunk {chunk_num_info}\nContent: {doc.page_content}\n---\n"
        print(f"\n--- DEBUG: Retrieved Context ---\n{retrieved_context}\n--------------------------------\n")
        
        if not retrieved_context:
            prompt = f'You are an AI assistant. The user asked: "{request.query}". I could not find any relevant information. State that you do not have enough information to answer.'
        else:
            prompt = f'You are an AI assistant. Your task is to answer the User\'s Query based ONLY on the provided CONTEXT.\n\n--- CONTEXT ---\n{retrieved_context}\n--- END CONTEXT ---\n\nUser\'s Query:\n"{request.query}"\n\nBased solely on the CONTEXT, provide a concise and direct answer. If the context does NOT contain the answer, state: "The provided context does not contain enough information to answer."'
        
        llm_response = call_llm(prompt)
        if "error" in llm_response:
            raise HTTPException(status_code=500, detail=llm_response["error"])
        return llm_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform search: {e}")

@app.post("/hackrx/run", response_model=HackRxResponse, summary="HackRx Submission Endpoint")
async def hackrx_run(request_body: HackRxRequest, authorization: Optional[str] = Header(None)):
    if HACKRX_API_KEY and (not authorization or f"Bearer {HACKRX_API_KEY}" != authorization):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key.")
    
    all_answers = []
    global vector_db
    if vector_db is None or vector_db._collection.count() == 0:
        for questions_list in request_body.questions.values():
            all_answers.extend(["The provided context does not contain enough information to answer."] * len(questions_list))
        return HackRxResponse(answers=all_answers)

    for doc_filename, questions_list in request_body.questions.items():
        for question_text in questions_list:
            try:
                retrieved_docs = vector_db.similarity_search(question_text, k=15)
                retrieved_context = ""
                if retrieved_docs:
                    for doc in retrieved_docs:
                        source_info = doc.metadata.get("source", "Unknown Source")
                        chunk_num_info = doc.metadata.get("chunk_num", "N/A")
                        retrieved_context += f"Source: {source_info}, Chunk {chunk_num_info}\nContent: {doc.page_content}\n---\n"
                
                if not retrieved_context:
                    prompt = f'You are an AI assistant. The user asked: "{question_text}". I could not find any relevant information. State that you do not have enough information to answer.'
                else:
                    prompt = f'You are an AI assistant. Your task is to answer the User\'s Query based ONLY on the provided CONTEXT.\n\n--- CONTEXT ---\n{retrieved_context}\n--- END CONTEXT ---\n\nUser\'s Query:\n"{question_text}"\n\nBased solely on the CONTEXT, provide a concise and direct answer. If the context does NOT contain the answer, state: "The provided context does not contain enough information to answer."'
                
                llm_response = call_llm(prompt)
                all_answers.append(llm_response.get('answer', f"Error processing question: {llm_response.get('error')}"))
            except Exception as e:
                print(f"Error processing question '{question_text}' for '{doc_filename}': {e}")
                all_answers.append("An internal error occurred while processing this question.")
    return HackRxResponse(answers=all_answers)
