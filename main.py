import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from retriever import hybrid_rerank_retrieve

app = FastAPI(title="Enterprise Hybrid RAG Engine")

# Enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from React frontend (localhost:3000)
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, GET, OPTIONS requests
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"status": "Enterprise RAG Engine is running."}

@app.post("/api/v1/query")
async def process_query(request: QueryRequest):
    start_time = time.time()
    
    # Guardrail 1: Input Sanitization
    cleaned_prompt = request.prompt.strip()
    if len(cleaned_prompt) < 3:
        raise HTTPException(status_code=400, detail="Query is too short. Please provide a valid prompt.")
    
    if len(cleaned_prompt) > 1000:
        raise HTTPException(status_code=400, detail="Query exceeds maximum character limit of 1000.")

    # Step 1: Hybrid Retrieval & Reranking
    retrieved_docs = hybrid_rerank_retrieve(cleaned_prompt, top_k=3)
    
    # Extract context passages
    context = "\n---\n".join([doc["text"] for doc in retrieved_docs])
    
    # Context-aware Response
    response_answer = f"Based on the knowledge base context:\n\n{context[:400]}..."
    
    # Compute metrics
    latency_ms = round((time.time() - start_time) * 1000, 2)
    estimated_token_count = len((cleaned_prompt + context).split())

    return {
        "query": cleaned_prompt,
        "answer": response_answer,
        "retrieved_sources": retrieved_docs,
        "metrics": {
            "latency_ms": latency_ms,
            "estimated_tokens": estimated_token_count
        }
    }