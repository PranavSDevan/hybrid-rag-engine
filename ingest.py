import os
import pickle
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

print("1. Loading PDFs from ./data folder...")
loader = PyPDFDirectoryLoader("./data")
documents = loader.load()

if not documents:
    print("No PDFs found in ./data! Please add at least one PDF file.")
    exit()

print(f"Loaded {len(documents)} document page(s). Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} text chunks.")

# Extract text strings for indexing
chunk_texts = [doc.page_content for doc in chunks]

# Save BM25 index locally for keyword search
print("2. Building BM25 lexical index...")
tokenized_corpus = [doc.lower().split(" ") for doc in chunk_texts]
bm25 = BM25Okapi(tokenized_corpus)

with open("bm25_index.pkl", "wb") as f:
    pickle.dump({"bm25": bm25, "chunks": chunk_texts}, f)
print("Saved BM25 index to bm25_index.pkl.")

# Generate Dense Embeddings and upload to Qdrant
print("3. Generating Dense Embeddings using SentenceTransformers...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(chunk_texts, show_progress_bar=True)

print("4. Connecting to Qdrant Vector Database...")
client = QdrantClient(host="localhost", port=6333)

collection_name = "enterprise_kb"
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

points = [
    PointStruct(
        id=idx,
        vector=embedding.tolist(),
        payload={"text": chunk_texts[idx]}
    )
    for idx, embedding in enumerate(embeddings)
]

client.upsert(collection_name=collection_name, points=points)
print(f"Successfully uploaded {len(points)} vectors to Qdrant collection '{collection_name}'!")