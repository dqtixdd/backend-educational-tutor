import os
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uuid
from fastapi import UploadFile, File
from jose import jwt, JWTError
import logging
from fastapi import BackgroundTasks

# Auth imports
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from jose import jwt, JWTError 

# Internal modules
import chromadb
import answer_once as ao 
from db import Base, engine
from models import User, Conversation, Message
from memory_sqlalchemy import MemoryStore
import ingest

load_dotenv()

# --- CONFIG ---
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret-key-change-me")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

Base.metadata.create_all(bind=engine)

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma")
db = chromadb.PersistentClient(path=CHROMA_PATH)
col = db.get_or_create_collection("edu-tutor")

app = FastAPI()
memory = MemoryStore()

origins = [
    "http://localhost:5173",  # local dev
    "https://frontend-educational-tutor.vercel.app",  # your Vercel URL
    # if Vercel preview URLs matter later, you can use regex instead
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELS ---
class ChatRequest(BaseModel):
    question: str
    k: int = 6
    conversation_id: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    rewritten_question: str

class UserRegister(BaseModel):
    email: str
    username: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

# --- AUTH DEPENDENCY ---
def get_current_user(x_token: str = Header(None)):
    """
    Validates token. 
    1. Tries to decode as our Custom JWT (Email/Pass login).
    2. If that fails, tries to decode as Google Token.
    """
    if not x_token:
        raise HTTPException(401, "Missing X-Token header")
    
    # 1. Try Custom JWT
    try:
        payload = jwt.decode(x_token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub") # returns email
    except JWTError:
        pass # Not a custom token, fall through to Google check

    # 2. Try Google Token
    try:
        id_info = id_token.verify_oauth2_token(
            x_token, google_requests.Request(), GOOGLE_CLIENT_ID
        )
        return id_info['email']
    except ValueError:
        raise HTTPException(401, "Invalid Token (Neither Google nor Custom)")

# --- AUTH ENDPOINTS ---
logger = logging.getLogger("uvicorn.error")

@app.post("/register")
def register(user: UserRegister):
    try:
        success = memory.create_user(user.email, user.username, user.password)
    except Exception as e:
        # log full error to Render logs
        logger.exception("Error while registering user %s", user.email)
        raise HTTPException(status_code=500, detail="Registration failed")

    if not success:
        # email already exists
        raise HTTPException(status_code=400, detail="Email already registered")

    return {"message": "User created successfully"}
@app.post("/login")
def login(user: UserLogin):
    db_user = memory.verify_user(user.email, user.password)
    if not db_user:
        raise HTTPException(401, "Invalid credentials")
    
    # Create JWT Token
    token = jwt.encode({"sub": db_user["email"], "name": db_user["username"]}, SECRET_KEY, algorithm=ALGORITHM)
    return {
        "token": token, 
        "email": db_user["email"], 
        "name": db_user["username"],
        "picture": "" # No picture for custom auth
    }

# --- CHAT ENDPOINTS (Existing) ---

@app.get("/pdfs")
def list_pdfs(email: str = Depends(get_current_user)):
    """Return the list of PDF files currently available for the tutor."""
    pdfs = []
    if DATA_DIR.exists():
        for p in DATA_DIR.glob("*.pdf"):
            pdfs.append(p.name)
    return pdfs

@app.post("/upload_pdf")
async def upload_pdf(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    email: str = Depends(get_current_user),
):
    # Validate
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    # Make unique filename
    safe_name = f"{uuid.uuid4().hex}_{file.filename}"
    dest = DATA_DIR / safe_name

    # Save very fast
    contents = await file.read()
    with open(dest, "wb") as f:
        f.write(contents)

    # Queue ingest job (this runs AFTER response finishes)
    background.add_task(ingest_pdf_background, dest, email, safe_name)

    return {
        "ok": True,
        "status": "processing",
        "filename": safe_name,
    }

def ingest_pdf_background(dest: Path, user_email: str, filename: str):
    print(f"[BG TASK] Starting ingest for {filename}")
    try:
        stats = ingest.ingest_single_pdf(dest)
        print(f"[BG TASK] Finished ingest: {stats}")
    except Exception as e:
        print(f"[BG TASK] ERROR ingesting {filename}: {e}")

@app.delete("/pdfs/{filename}")
def delete_pdf(filename: str, email: str = Depends(get_current_user)):
    # 1. delete the file on disk
    path = DATA_DIR / filename
    if path.exists():
        path.unlink()

    # 2. delete all its vectors from Chroma
    db = chromadb.PersistentClient(path="chroma")
    col = db.get_or_create_collection("edu-tutor")
    col.delete(where={"source": filename})

    return {"ok": True}

@app.get("/conversations")
def get_conversations(email: str = Depends(get_current_user)):
    return memory.get_all_conversations(email)

@app.get("/conversations/{cid}")
def get_conversation_history(cid: str, email: str = Depends(get_current_user)):
    return memory.get_messages(cid)

@app.delete("/conversations/{cid}")
def delete_conversation(cid: str, email: str = Depends(get_current_user)):
    memory.delete_conversation(cid, email)
    return {"ok": True}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, email: str = Depends(get_current_user)):
    cid = req.conversation_id

    # 1. Create chat if new
    msgs = memory.get_messages(cid)
    if not msgs:
        title = req.question[:30] + "..." if len(req.question) > 30 else req.question
        memory.create_conversation(cid, email, title)

    # 2. Save User Msg
    memory.add_message(cid, "user", req.question)

        # 3. RAG + hybrid logic
    history = memory.transcript_text(cid, last_n=6)
    rewritten = req.question  # later you can plug back your rewriter here

    docs, metas, ids = ao.retrieve(rewritten, k=req.k)

    # Build a readable context string with citations
    blocks = [f"[{m['source']} p.{m['page']}]\n{d}" for d, m in zip(docs, metas)]
    context = "\n\n".join(blocks).strip()

    # System prompt for Option C (hybrid)
    system_prompt = (
        "You are an educational tutor. You are helping a student who uploaded PDFs. "
        "You will be given context excerpts from those PDFs, plus the student's question.\n\n"
        "Use the context as your primary reference *when it is relevant*.\n"
        "- If the answer is clearly supported by the context, base your explanation on it.\n"
        "- If the question requires more reasoning or goes beyond what is explicitly written, "
        "you may use your own general knowledge and problem-solving skills.\n"
        "- If the context is mostly irrelevant, rely on your own knowledge but you can mention "
        "that the materials do not directly cover the exact question.\n\n"
        "Structure your answer using MARKDOWN headings:\n"
        "## 📚 From the materials\n"
        "- What can be justified strictly from the provided excerpts.\n\n"
        "## 🧠 Beyond the materials (optional)\n"
        "- Additional reasoning, examples, background knowledge.\n"
        "- Include this section only if needed.\n"
    )

    # User message combining context + question
    user_content = f"""
Context excerpts from the student's PDFs:
{context if context else "(no relevant excerpts were retrieved)"}

Now answer the student's question.

Question: {req.question}

Remember to structure your answer like this (omit a section if it's empty):

From the materials:
- ...

Beyond the materials (optional):
- ...
""".strip()

    comp = ao.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    answer = comp.choices[0].message.content
    sources = [{"source": m["source"], "page": m["page"], "id": i} for m, i in zip(metas, ids)]

    answer = answer.replace("From the materials:", "## 📚 From the materials")
    answer = answer.replace("Beyond the materials:", "## 🧠 Beyond the materials")
    # 4. Save Assistant Msg
    memory.add_message(cid, "assistant", answer, sources)
    

    return ChatResponse(answer=answer, sources=sources, rewritten_question=rewritten)
