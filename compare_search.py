# compare_search.py - 파인튜닝 전후 검색 품질 비교
import clip
import torch
import chromadb
import numpy as np
from pathlib import Path

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 두 DB 로드
db_orig = chromadb.PersistentClient(path="./db_original").get_collection("fashion")
db_ft = chromadb.PersistentClient(path="./db").get_collection("fashion")

# 색상 키워드 매핑
color_keywords = {
    "white": ["white", "off", "ivory", "cream"],
    "black": ["black", "dark"],
    "blue": ["blue", "navy", "dark blue", "light blue"],
    "gray": ["gray", "grey", "dark grey", "dark gray"],
    "pink": ["pink", "light pink"],
}

def get_text_vec(query):
    token = clip.tokenize([query]).to(device)
    with torch.no_grad():
        vec = model.encode_text(token)
        vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.cpu().numpy().flatten().tolist()

def color_match(result_color, query_color):
    result_color = result_color.lower()
    for keyword in color_keywords.get(query_color, [query_color]):
        if keyword in result_color:
            return True
    return False

test_queries = [
    ("white top", "top", "white"),
    ("black pants", "bottom", "black"),
    ("blue jeans", "bottom", "blue"),
    ("gray bag", "bag", "gray"),
    ("pink top", "top", "pink"),
    ("white skirt", "bottom", "white"),
    ("black shoes", "shoes", "black"),
    ("black bag", "bag", "black"),
    ("white hat", "hat", "white"),
    ("black top", "top", "black"),
]

print("=" * 60)
print(f"{'쿼리':<20} {'DB':<10} {'유사도평균':>10} {'색상일치율':>10}")
print("=" * 60)

for query, category, target_color in test_queries:
    vec = get_text_vec(query)
    
    for label, db in [("기존", db_orig), ("파인튜닝", db_ft)]:
        results = db.query(
            query_embeddings=[vec],
            n_results=3,
            where={"category": category}
        )
        
        distances = results["distances"][0]
        similarities = [(2 - d) / 2 for d in distances]
        avg_sim = np.mean(similarities)
        
        colors = [m.get("color", "") for m in results["metadatas"][0]]
        matches = sum(1 for c in colors if color_match(c, target_color))
        color_rate = matches / len(colors) * 100
        
        print(f"{query:<20} {label:<10} {avg_sim:>10.4f} {color_rate:>9.0f}%")
    print("-" * 60)


print("\n" + "=" * 60)
print("전체 요약")
print("=" * 60)

orig_sims = []
ft_sims = []
orig_colors = []
ft_colors = []

for query, category, target_color in test_queries:
    vec = get_text_vec(query)
    
    for label, db in [("기존", db_orig), ("파인튜닝", db_ft)]:
        results = db.query(
            query_embeddings=[vec],
            n_results=3,
            where={"category": category}
        )
        distances = results["distances"][0]
        similarities = [(2 - d) / 2 for d in distances]
        avg_sim = np.mean(similarities)
        colors = [m.get("color", "") for m in results["metadatas"][0]]
        matches = sum(1 for c in colors if color_match(c, target_color))
        color_rate = matches / len(colors) * 100

        if label == "기존":
            orig_sims.append(avg_sim)
            orig_colors.append(color_rate)
        else:
            ft_sims.append(avg_sim)
            ft_colors.append(color_rate)

print(f"{'':20} {'기존':>10} {'파인튜닝':>10} {'변화':>10}")
print(f"{'평균 유사도':20} {np.mean(orig_sims):>10.4f} {np.mean(ft_sims):>10.4f} {np.mean(ft_sims)-np.mean(orig_sims):>+10.4f}")
print(f"{'평균 색상일치율':20} {np.mean(orig_colors):>9.1f}% {np.mean(ft_colors):>9.1f}% {np.mean(ft_colors)-np.mean(orig_colors):>+9.1f}%")