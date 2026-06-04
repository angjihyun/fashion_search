# step3.py - 텍스트로 검색
import clip
import torch
import chromadb
from PIL import Image
from pathlib import Path

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

client = chromadb.PersistentClient(path="./db")
collection = client.get_collection("fashion")

def search(query, top_k=5):
    # 텍스트 벡터화
    token = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_vec = model.encode_text(token)
        text_vec = text_vec / text_vec.norm(dim=-1, keepdim=True)

    # 검색
    results = collection.query(
        query_embeddings=[text_vec.cpu().numpy().flatten().tolist()],
        n_results=top_k
    )

    print(f"\n검색어: '{query}'")
    print("-" * 40)
    for i, (filename, distance) in enumerate(zip(
        results["ids"][0], results["distances"][0]
    )):
        similarity = (2 - distance) / 2
        print(f"{i+1}. {filename} (유사도: {similarity:.4f})")

# 테스트
search("casual white top")
search("black pants")
search("formal outfit")