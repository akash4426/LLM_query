
import os
import pickle
# import faiss # REMOVED: No longer directly used after switching to ChromaDB
import numpy as np
import re # Import regex module
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, status
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
# import google.generativeai as genai # Removed Gemini import
from together import Together # Added Together AI import

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma # Updated import for ChromaDB
from langchain_core.documents import Document

from fastapi.responses import RedirectResponse, HTMLResponse

# --- Pydantic Models (Moved to top for clear definition order) ---
class QueryRequest(BaseModel):
    query: str

class HackRxDocument(BaseModel):
    url: str
    filename: str # Assuming this filename maps to an already ingested document

class HackRxRequest(BaseModel):
    documents: Dict[str, List[str]] # Key is URL (unused for now), value is list of filenames
    questions: Dict[str, List[str]] # Key is filename, value is list of questions

class HackRxResponse(BaseModel):
    answers: List[str]
# --- End Pydantic Models ---


# --- Load environment variables ---
load_dotenv()
# Changed GEMINI_API_KEY to TOGETHER_API_KEY
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY not found in .env")

# New: API Key for the /hackrx/run endpoint
HACKRX_API_KEY = os.getenv("HACKRX_API_KEY")
if not HACKRX_API_KEY:
    print("WARNING: HACKRX_API_KEY not found in .env. The /hackrx/run endpoint will not be secured.")


# --- Configure Together AI ---
together_client = Together(api_key=TOGETHER_API_KEY)
# Define the model to use from Together AI.
# Changed to meta-llama/Llama-3-8b-chat-hf as a highly performant model.
# Other excellent options include: "mistralai/Mixtral-8x7B-Instruct-v0.1", "Qwen/Qwen1.5-72B-Chat"
TOGETHER_LLM_MODEL = "meta-llama/Llama-3-8b-chat-hf" 

# --- FastAPI App ---
app = FastAPI(
    title="Document Q&A RAG System",
    description="An API for uploading PDFs and querying them using a Together AI LLM.",
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
            }}

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

# Path for ChromaDB persistence
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")

# --- Global State ---
vector_db: Optional[Chroma] = None

# --- Embedding Model ---
print("Loading embedding model...")
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
print("Embedding model loaded.")

# --- Load Vector Database on Startup ---
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

# --- New Endpoint to check Knowledge Base Status ---
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
    return {
        "vector_db_loaded": vector_db_loaded,
        "num_chunks": num_chunks
    }


# --- LLM Function (Now uses Together AI) ---
def call_llm(prompt: str):
    try:
        print(f"DEBUG: Calling LLM with model: {TOGETHER_LLM_MODEL} (Together AI)")
        messages = [
            {"role": "user", "content": prompt}
        ]

        response = together_client.chat.completions.create(
            model=TOGETHER_LLM_MODEL,
            messages=messages,
            max_tokens=500, # Increased max_tokens for potentially longer answers
            temperature=0.0 # Set temperature to 0.0 for more deterministic answers
        )
        
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            return {"answer": response.choices[0].message.content.strip()}
        else:
            return {"error": "Together AI LLM returned no content."}
    except Exception as e:
        print(f"Error calling Together AI LLM: {e}")
        return {"error": f"Failed to get response from Together AI LLM: {e}"}

# --- Text Cleaning Function for OCR issues ---
def clean_ocr_text(text: str) -> str:
    """
    Cleans text extracted from PDFs that might have extra spaces between characters
    due to OCR issues (e.g., "T r e a t m e n t" -> "Treatment").
    This regex attempts to remove single spaces that appear between word characters.
    It also attempts to split concatenated words (e.g., "Contractmeans" -> "Contract means").
    """
    # 1. Remove single spaces between characters (original fix for "T r e a t m e n t")
    cleaned_text = re.sub(r'(?<=\w)\s(?=\w)', '', text)

    # 2. Attempt to split concatenated words:
    #    - Insert space before an uppercase letter followed by a lowercase letter (e.g., "WordStarts" -> "Word Starts")
    #    - Insert space before a digit that follows a letter (e.g., "text123" -> "text 123")
    #    - Insert space before a non-digit that follows a digit (e.g., "123text" -> "123 text")
    # This is heuristic and might not catch all cases or might introduce new errors.
    cleaned_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned_text) # lowercase followed by uppercase
    cleaned_text = re.sub(r'(\D)(\d)', r'\1 \2', cleaned_text)     # non-digit followed by digit
    cleaned_text = re.sub(r'(\d)(\D)', r'\1 \2', cleaned_text)     # digit followed by non-digit

    # Specific common merged words (can be expanded based on observation)
    # These are added to handle specific observed concatenations more robustly
    cleaned_text = re.sub(r'Contractmeans', 'Contract means', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'Policyandthe', 'Policy and the', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'Co-paymentmeans', 'Co-payment means', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'CumulativeBonusmeans', 'Cumulative Bonus means', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'IncaseofanygrievancerelatedtothePolicy, theinsuredpersonmaysubmitinwritingtothePolicyIssuingOfficeorGrievancecellatRegionalOfficeoftheCompanyforredressal.', 'In case of any grievance related to the Policy, the insured person may submit in writing to the Policy Issuing Office or Grievance cell at Regional Office of the Company for redressal.', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'Ifthegrievanceremainsunaddressed, theinsuredpersonmaysubmitinwritingtothePolicyIssuingOfficeorGrievancecellatRegionalOfficeoftheCompanyforredressal.', 'If the grievance remains unaddressed, the insured person may submit in writing to the Policy Issuing Office or Grievance cell at Regional Office of the Company for redressal.', cleaned_text, flags=re.IGNORECASE) # Duplicated, but harmless
    cleaned_text = re.sub(r'Formoreinformationongrievancemechanism, andtodownloadgrievanceform, visitour', 'For more information on grievance mechanism, and to download grievance form, visit our', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'CustomerRelationshipManagementDept\.', 'Customer Relationship Management Dept.', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'NationalInsuranceCompanyLimited, PremisesNo\. 18 -0374, Plotno\. CBD -81, New Town, Kolkata - 700156, email: customer\.relations@nic\.co\.in, griho@nic\.co\.in', 'National Insurance Company Limited, Premises No. 18-0374, Plot no. CBD-81, New Town, Kolkata - 700156, email: customer.relations@nic.co.in, griho@nic.co.in', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'REDRESSALOFGRIEVANCE', 'REDRESSAL OF GRIEVANCE', cleaned_text, flags=re.IGNORECASE)


    # Normalize spaces (replace multiple spaces with a single one)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text

# --- Boilerplate Removal Function (ADDED TO FIX THE ERROR) ---
def remove_boilerplate_text(text: str) -> str:
    """
    Removes common boilerplate text like headers, footers, page numbers,
    and specific policy identifiers from the document text.
    """
    # Patterns to remove (case-insensitive)
    patterns = [
        r'Page\s*\d+\s*of\s*\d+',                # "Page X of Y"
        r'NATIONAL INSURANCE COMPANY LIMITED',      # Company name header
        r'Policy No\.:\s*[\w-]+',                 # "Policy No.: ..."
        r'UIN:\s*[\w-]+',                        # "UIN: ..."
        # Add any other repeating headers or footers you observe in your documents
    ]
    
    cleaned_text = text
    for pattern in patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)

    # Normalize whitespace that may be left after removals
    return re.sub(r'\s+', ' ', cleaned_text).strip()


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
        
        processed_documents = []
        for doc in documents:
            # Apply OCR cleaning first
            cleaned_content = clean_ocr_text(doc.page_content)
            # Then apply boilerplate removal
            final_content = remove_boilerplate_text(cleaned_content)
            
            # Only add document if content is not empty after cleaning
            if final_content.strip():
                processed_documents.append(Document(page_content=final_content, metadata=doc.metadata))

        if not processed_documents:
            # If all pages became empty after cleaning, raise an error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=400, detail="No meaningful text found in PDF after cleaning boilerplate.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
        split_chunks = splitter.split_documents(processed_documents)

    except Exception as e:
        # Ensure file_path is used for cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    if not split_chunks:
        # Ensure file_path is used for cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail="No text found in PDF or document is empty after chunking.")

    global vector_db
    if vector_db is None:
        load_knowledge_base()
        if vector_db is None:
            raise HTTPException(status_code=500, detail="Failed to initialize vector database.")

    try:
        docs_to_add = []
        for i, chunk in enumerate(split_chunks):
            chunk.metadata["source"] = file.filename
            chunk.metadata["chunk_num"] = i
            docs_to_add.append(chunk)

        vector_db.add_documents(docs_to_add)

    except Exception as e:
        print(f"Error adding documents to ChromaDB: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add documents to knowledge base: {e}")

    return {"status": "success", "filename": file.filename, "chunks_added": len(split_chunks)}

# --- Query Endpoint (Existing) ---
@app.post("/query", summary="Ask a question to the documents")
async def handle_query(request: QueryRequest):
    global vector_db

    if vector_db is None or vector_db._collection.count() == 0:
        raise HTTPException(status_code=503, detail="Knowledge base is empty. Please ingest documents first.")

    try:
        retrieved_docs = vector_db.similarity_search(request.query, k=15)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform vector database search: {e}")

    retrieved_context = ""
    if not retrieved_docs:
        print("Warning: No documents retrieved for the query.")
    else:
        for doc in retrieved_docs:
            source_info = doc.metadata.get("source", "Unknown Source")
            chunk_num_info = doc.metadata.get("chunk_num", "N/A")
            retrieved_context += (
                f"Source: {source_info}, Chunk {chunk_num_info}\n"
                f"Content: {doc.page_content}\n---\n"
            )

    print("\n--- DEBUG: Retrieved Context sent to LLM ---")
    print(retrieved_context)
    print("-------------------------------------------\n")

    if not retrieved_context:
        prompt = f"""
        You are an AI assistant. The user asked: "{request.query}".
        I could not find any relevant information in the documents. Please state that you don't have enough information to answer.
        """
    else:
        prompt = f"""
You are an AI assistant.
Your task is to answer the User's Query based ONLY on the provided CONTEXT.
Do not use any outside knowledge.

--- CONTEXT ---
{retrieved_context}
--- END CONTEXT ---

User's Query:
"{request.query}"

Based solely on the CONTEXT, provide a concise and direct answer.
- If the context contains the answer: "[Your concise answer based on context]."
- If the context does NOT contain the answer: "The provided context does not contain enough information to answer."
        """

    llm_response = call_llm(prompt)
    if "error" in llm_response:
        raise HTTPException(status_code=500, detail=llm_response["error"])

    return llm_response

# --- New HackRx Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse, summary="HackRx Submission Endpoint")
async def hackrx_run(
    request_body: HackRxRequest,
    authorization: Optional[str] = Header(None)
):
    global vector_db

    # --- API Key Authentication ---
    if HACKRX_API_KEY:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header missing or malformed. Use 'Bearer <api_key>'."
            )
        token = authorization.split(" ")[1]
        if token != HACKRX_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API Key."
            )
    else:
        print("WARNING: HACKRX_API_KEY is not set in .env. /hackrx/run endpoint is unsecured.")

    all_answers = []

    # Check if knowledge base is ready before processing questions
    if vector_db is None or vector_db._collection.count() == 0:
        # If no documents are ingested, return a default "not enough info" for all questions
        for doc_filename, questions_list in request_body.questions.items():
            for question_text in questions_list:
                all_answers.append("The provided context does not contain enough information to answer.")
        return HackRxResponse(answers=all_answers)


    # Iterate through documents and their questions
    for doc_filename, questions_list in request_body.questions.items():
        # In this simplified RAG, we query the entire vector_db.
        # For a more advanced setup, you could filter retrieval by source if ChromaDB supported it easily
        # or if you had a more complex metadata indexing strategy.

        for question_text in questions_list:
            retrieved_context = ""
            try:
                # Retrieve relevant documents for the current question
                # We use k=15 as defined for the /query endpoint
                retrieved_docs = vector_db.similarity_search(question_text, k=15)

                if not retrieved_docs:
                    print(f"Warning: No documents retrieved for question: {question_text} from {doc_filename}")
                else:
                    for doc in retrieved_docs:
                        source_info = doc.metadata.get("source", "Unknown Source")
                        chunk_num_info = doc.metadata.get("chunk_num", "N/A")
                        retrieved_context += (
                            f"Source: {source_info}, Chunk {chunk_num_info}\n"
                            f"Content: {doc.page_content}\n---\n"
                        )

                # Construct the prompt for the LLM
                if not retrieved_context:
                    prompt = f"""
                    You are an AI assistant. The user asked: "{question_text}".
                    I could not find any relevant information in the documents. Please state that you don't have enough information to answer.
                    """
                else:
                    prompt = f"""
You are an AI assistant.
Your task is to answer the User's Query based ONLY on the provided CONTEXT.
Do not use any outside knowledge.

--- CONTEXT ---
{retrieved_context}
--- END CONTEXT ---

User's Query:
"{question_text}"

Based solely on the CONTEXT, provide a concise and direct answer.
- If the context contains the answer: "[Your concise answer based on context]."
- If the context does NOT contain the answer: "The provided context does not contain enough information to answer."
                    """
                
                # Call the LLM
                llm_response = call_llm(prompt)
                if "error" in llm_response:
                    all_answers.append(f"Error processing question: {llm_response['error']}")
                else:
                    all_answers.append(llm_response['answer'])

            except Exception as e:
                # Catch any unexpected errors during processing a question
                print(f"Error processing question '{question_text}' for document '{doc_filename}': {e}")
                all_answers.append(f"An internal error occurred while processing this question.")

    return HackRxResponse(answers=all_answers)
