# step7.py - 쿼리 확장 (llava로 검색어 확장) + 확장 전후 비교 실험
import ollama
import clip
import torch
import chromadb
from PIL import Image
from pathlib import Path

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

client = chromadb.PersistentClient(path="./db")
collection = client.get_collection("fashion")

def expand_query(query):
    prompt = f"""Fashion keywords only. No bullets, no quotes, no punctuation.
Expand: {query}
Response (single line, max 10 words):"""
    response = ollama.chat(
        model="llava",
        messages=[{"role": "user", "content": prompt}]
    )
    expanded = response["message"]["content"].strip()
    print(f"쿼리 확장: '{query}' → '{expanded}'")
    return expanded

def search_by_category(text_vec, category, top_k=3):
    results = collection.query(
        query_embeddings=[text_vec.cpu().numpy().flatten().tolist()],
        n_results=top_k,
        where={"category": category}
    )
    return results["ids"][0]

def search(query):
    expanded = expand_query(query)
    token = clip.tokenize([expanded]).to(device)
    with torch.no_grad():
        text_vec = model.encode_text(token)
        text_vec = text_vec / text_vec.norm(dim=-1, keepdim=True)
    tops    = search_by_category(text_vec, "top")
    bottoms = search_by_category(text_vec, "bottom")
    shoes   = search_by_category(text_vec, "shoes")
    bags    = search_by_category(text_vec, "bag")
    hats    = search_by_category(text_vec, "hat")
    return tops, bottoms, shoes, bags, hats

# 확장 전후 비교 테스트 (보고서용)
test_queries = ["cute", "formal", "street", "sporty"]

for q in test_queries:
    print(f"\n===== '{q}' =====")
    expanded = expand_query(q)
    
    token_before = clip.tokenize([q]).to(device)
    with torch.no_grad():
        vec_before = model.encode_text(token_before)
        vec_before = vec_before / vec_before.norm(dim=-1, keepdim=True)
    results_before = collection.query(
        query_embeddings=[vec_before.cpu().numpy().flatten().tolist()],
        n_results=3
    )
    
    token_after = clip.tokenize([expanded]).to(device)
    with torch.no_grad():
        vec_after = model.encode_text(token_after)
        vec_after = vec_after / vec_after.norm(dim=-1, keepdim=True)
    results_after = collection.query(
        query_embeddings=[vec_after.cpu().numpy().flatten().tolist()],
        n_results=3
    )
    
    print(f"확장 전: {results_before['ids'][0]}")
    print(f"확장 후: {results_after['ids'][0]}")