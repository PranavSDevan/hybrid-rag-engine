import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient

# Load BM25 index and raw chunks saved during ingestion
with open("bm25_index.pkl", "rb") as f:
    data = pickle.load(f)
    bm25 = data["bm25"]
    all_chunks = data["chunks"]

# Initialize SentenceTransformers & Qdrant Client
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
qdrant_client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "enterprise_kb"

def bm25_search(query: str, top_k: int = 20):
    """Perform BM25 Lexical Keyword Search"""
    tokenized_query = query.lower().split(" ")
    scores = bm25.get_scores(tokenized_query)

    # Pair scores with chunk indices and sort descending
    scored_docs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"id": idx, "text": all_chunks[idx], "bm25_score": score} for idx, score in scored_docs]

def qdrant_search(query: str, top_k: int = 20):
    """Perform Qdrant Dense Vector Search"""
    query_vector = embedding_model.encode(query).tolist()
    
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )
    
    results = []
    for point in search_results.points:
        results.append({
            "id": point.id,
            "text": point.payload["text"],
            "vector_score": point.score
        })
    return results

def reciprocal_rank_fusion(bm25_results, vector_results, k: int = 60):
    """Combine keyword and vector results using Reciprocal Rank Fusion (RRF)"""
    rrf_scores = {}
    chunk_map = {}

    # Process BM25 rankings
    for rank, item in enumerate(bm25_results):
        chunk_id = item["id"]
        chunk_map[chunk_id] = item["text"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank + 1))

    # Process Vector rankings
    for rank, item in enumerate(vector_results):
        chunk_id = item["id"]
        chunk_map[chunk_id] = item["text"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank + 1))

    # Sort by combined RRF score
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"id": cid, "text": chunk_map[cid], "rrf_score": score} for cid, score in sorted_chunks]

def hybrid_rerank_retrieve(query: str, top_k: int = 5):
    """Full Pipeline: Lexical + Dense -> RRF Fusion -> Cross-Encoder Rerank"""
    bm25_res = bm25_search(query, top_k=20)
    vector_res = qdrant_search(query, top_k=20)
    
    fused_candidates = reciprocal_rank_fusion(bm25_res, vector_res)
    
    # Prepare candidate pairs for Cross-Encoder
    pairs = [[query, candidate["text"]] for candidate in fused_candidates]
    cross_scores = cross_encoder.predict(pairs)

    # Sort candidates by Cross-Encoder score
    for idx, candidate in enumerate(fused_candidates):
        candidate["cross_encoder_score"] = float(cross_scores[idx])

    reranked = sorted(fused_candidates, key=lambda x: x["cross_encoder_score"], reverse=True)
    return reranked[:top_k]