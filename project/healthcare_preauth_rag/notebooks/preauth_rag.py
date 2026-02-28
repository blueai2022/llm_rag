#!/usr/bin/env python
# coding: utf-8

# %% [markdown]
# # Healthcare PreAuth RAG Demo
# 
# Query prior authorization rules using semantic search, reranking, and LLM.

# %% Setup & Imports

import os
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

# Project paths
PROJECT_ROOT = Path.cwd().parent
DATA_DIR = PROJECT_ROOT / "data" / "preauth_rules"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"

print(f"Data directory: {DATA_DIR}")
print(f"Vector store: {VECTOR_STORE_DIR}")


# %% 1. Load and Chunk Documents

def load_documents(data_dir):
    """Load all text files from data directory."""
    documents = []

    for filepath in data_dir.glob("*.txt"):
        with open(filepath, 'r') as f:
            content = f.read()
            # Split by procedure sections (##)
            sections = content.split('\n##')

            for section in sections[1:]:  # Skip first empty section
                section = '##' + section  # Add back the ##
                documents.append({
                    'text': section.strip(),
                    'source': filepath.name
                })

    return documents

docs = load_documents(DATA_DIR)
print(f"Loaded {len(docs)} document chunks")
print(f"\nSample chunk:\n{docs[0]['text'][:300]}...")


# %% 2. Create Embeddings

# Load embedding model (runs locally, no API needed)
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")

# Create embeddings for all documents
print(f"\nCreating embeddings for {len(docs)} chunks...")
texts = [doc['text'] for doc in docs]
embeddings = embedding_model.encode(texts, show_progress_bar=True)

print(f"Embeddings shape: {embeddings.shape}")


# %% 3. Create FAISS Vector Store

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype('float32'))

print(f"FAISS index created with {index.ntotal} vectors")


# %% 4. Load Reranker Model

print("Loading reranker model...")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("Reranker loaded!")


# %% 5. Search Function with Reranking

def search_preauth_rules(query, top_k=3, initial_retrieval=10):
    """Search for relevant preauth rules with reranking.
    
    Args:
        query: Search query
        top_k: Number of final results to return
        initial_retrieval: Number of candidates to retrieve before reranking
    """
    # Step 1: Initial retrieval with embedding similarity
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(
        query_embedding.astype('float32'), 
        initial_retrieval
    )
    
    # Step 2: Get candidate documents
    candidates = []
    for idx, dist in zip(indices[0], distances[0]):
        candidates.append({
            'text': docs[idx]['text'],
            'source': docs[idx]['source'],
            'embedding_distance': float(dist),
            'idx': idx
        })
    
    # Step 3: Rerank using cross-encoder
    query_doc_pairs = [[query, cand['text']] for cand in candidates]
    rerank_scores = reranker.predict(query_doc_pairs)
    
    # Step 4: Add rerank scores and sort
    for cand, score in zip(candidates, rerank_scores):
        cand['rerank_score'] = float(score)
    
    # Sort by rerank score (higher is better)
    reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
    
    # Return top_k results
    return reranked[:top_k]


# %% 6. Test Queries

# Query 1: By CPT code
query1 = "Does CPT code 93000 require prior authorization?"
print(f"Query: {query1}\n")
print("=" * 80)

results = search_preauth_rules(query1, top_k=2, initial_retrieval=5)
for i, result in enumerate(results, 1):
    print(f"\nResult {i}")
    print(f"Rerank Score: {result['rerank_score']:.3f}")
    print(f"Embedding Distance: {result['embedding_distance']:.3f}")
    print(f"Source: {result['source']}")
    print("-" * 80)
    print(result['text'])


# %%

# Query 2: By condition/procedure
query2 = "What are the requirements for MRI of the lumbar spine?"
print(f"Query: {query2}\n")
print("=" * 80)

results = search_preauth_rules(query2, top_k=2, initial_retrieval=5)
for i, result in enumerate(results, 1):
    print(f"\nResult {i}")
    print(f"Rerank Score: {result['rerank_score']:.3f}")
    print(f"Embedding Distance: {result['embedding_distance']:.3f}")
    print(f"Source: {result['source']}")
    print("-" * 80)
    print(result['text'])


# %%

# Query 3: By medical condition
query3 = "What documentation do I need for knee replacement surgery?"
print(f"Query: {query3}\n")
print("=" * 80)

results = search_preauth_rules(query3, top_k=2, initial_retrieval=5)
for i, result in enumerate(results, 1):
    print(f"\nResult {i}")
    print(f"Rerank Score: {result['rerank_score']:.3f}")
    print(f"Embedding Distance: {result['embedding_distance']:.3f}")
    print(f"Source: {result['source']}")
    print("-" * 80)
    print(result['text'])


# %% [markdown]
# ## 7. LLM-Enhanced Response with Reranked Results

# %%

from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_response(query, retrieved_docs):
    """Generate natural language response using retrieved context."""
    # Use reranked documents
    context = "\n\n".join([
        f"[Source: {doc['source']} | Relevance: {doc['rerank_score']:.2f}]\n{doc['text']}" 
        for doc in retrieved_docs
    ])
    
    prompt = f"""Based on the following prior authorization rules, answer the question.
    
RULES:
{context}
    
QUESTION: {query}
    
Provide a clear, concise answer with specific requirements and cite the source.
"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

# Test LLM response with reranking
results = search_preauth_rules("Does an EKG need prior auth?", top_k=2, initial_retrieval=5)
print("Query: Does an EKG need prior auth?\n")
print("=" * 80)
print("\nRetrieved & Reranked Documents:")
for i, res in enumerate(results, 1):
    print(f"{i}. {res['source']} (score: {res['rerank_score']:.3f})")

print("\n" + "=" * 80)
print("\nLLM Response:")
print("-" * 80)
answer = generate_response("Does an EKG need prior auth?", results)
print(answer)


# %% [markdown]
# ## Next Steps
