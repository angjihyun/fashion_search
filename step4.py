# step4.py - 코디 추천 UI (MMR + 한국어 쿼리 확장)
import clip
import torch
import chromadb
import gradio as gr
import ollama
import numpy as np
from PIL import Image
from pathlib import Path

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

client = chromadb.PersistentClient(path="./db")
collection = client.get_collection("fashion")

def expand_query(query):
    prompt = f"""You are a fashion expert. 
Step 1: If the query is not in English, translate it to English first.
Step 2: Expand the translated query into specific fashion keywords.

Query: {query}

Respond with ONLY a single line of English fashion keywords, no bullets, no quotes, no punctuation, max 10 words."""
    response = ollama.chat(
        model="llava",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"].strip()

def search_by_category(text_vec, category, top_k=3):
    candidates = collection.query(
        query_embeddings=[text_vec.cpu().numpy().flatten().tolist()],
        n_results=min(top_k * 3, 15),
        where={"category": category}
    )

    if not candidates["ids"][0]:
        return []

    candidate_ids = candidates["ids"][0]
    candidate_vecs = []
    for filename in candidate_ids:
        img_path = Path("./data/processed") / filename
        img = Image.open(img_path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            vec = model.encode_image(tensor)
            vec = vec / vec.norm(dim=-1, keepdim=True)
        candidate_vecs.append(vec.cpu().numpy().flatten())

    query_vec = text_vec.cpu().numpy().flatten()
    selected = []
    selected_vecs = []
    remaining = list(range(len(candidate_ids)))

    for _ in range(min(top_k, len(candidate_ids))):
        mmr_scores = []
        for i in remaining:
            query_sim = np.dot(query_vec, candidate_vecs[i])
            if selected_vecs:
                max_sim = max(np.dot(candidate_vecs[i], sv) for sv in selected_vecs)
            else:
                max_sim = 0
            mmr = 0.7 * query_sim - 0.3 * max_sim
            mmr_scores.append((i, mmr))

        best = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(candidate_ids[best])
        selected_vecs.append(candidate_vecs[best])
        remaining.remove(best)

    images = []
    for filename in selected:
        img_path = Path("./data/processed") / filename
        metadata = collection.get(ids=[filename])["metadatas"][0]
        caption = f"{metadata.get('color','')} | {metadata.get('mood','')} | {metadata.get('occasion','')}"
        images.append((Image.open(img_path), caption))
    return images

def search(query):
    expanded = expand_query(query)
    print(f"확장: '{query}' → '{expanded}'")

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

with gr.Blocks() as demo:
    gr.Markdown("# 👗 내 옷장 코디 추천 시스템")
    with gr.Row():
        query_input = gr.Textbox(label="어떤 스타일?", placeholder="예: 결혼식 하객룩 / 비올때 / 소개팅 코디")
        search_btn = gr.Button("추천 받기")

    gr.Markdown("### 👚 상의")
    gallery_top = gr.Gallery(columns=3, height=250)
    gr.Markdown("### 👖 하의")
    gallery_bottom = gr.Gallery(columns=3, height=250)
    gr.Markdown("### 👟 신발")
    gallery_shoes = gr.Gallery(columns=3, height=250)
    gr.Markdown("### 👜 가방")
    gallery_bag = gr.Gallery(columns=3, height=250)
    gr.Markdown("### 🧢 모자")
    gallery_hat = gr.Gallery(columns=3, height=250)

    search_btn.click(
        fn=search,
        inputs=query_input,
        outputs=[gallery_top, gallery_bottom, gallery_shoes, gallery_bag, gallery_hat]
    )

demo.launch()