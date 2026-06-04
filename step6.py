# step6.py - 태그 벡터 + 이미지 벡터 결합해서 ChromaDB 업데이트
# data/processed사진과 tags.json보고 이미지벡터+태그벡터 평균냄
# db/ChromaDB 재구축
# step6.py 수정 - 새 태그 구조 반영
import clip
import torch
import chromadb
import json
import numpy as np
from PIL import Image
from pathlib import Path

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

client = chromadb.PersistentClient(path="./db")
client.delete_collection("fashion")
collection = client.get_or_create_collection("fashion")

with open("./results/tags_finetuned2.json", encoding="utf-8") as f:
    tags_data = json.load(f)

for img_file in Path("./data/processed").iterdir():
    if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    if img_file.name not in tags_data:
        continue

    tag = tags_data[img_file.name]

    # 이미지 벡터
    img = Image.open(img_file).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_vec = model.encode_image(tensor)
        img_vec = img_vec / img_vec.norm(dim=-1, keepdim=True)

    # 태그 텍스트 구성 (mood + occasion + detail + fit + color)
    mood = " ".join(tag.get("mood", []))
    occasion = " ".join(tag.get("occasion", []))
    detail = " ".join(tag.get("detail", []))
    fit = tag.get("fit", "")
    color = tag.get("color", "")
    tag_text = f"{color} {fit} {mood} {occasion} {detail}"

    token = clip.tokenize([tag_text]).to(device)
    with torch.no_grad():
        text_vec = model.encode_text(token)
        text_vec = text_vec / text_vec.norm(dim=-1, keepdim=True)

    combined_vec = (img_vec + text_vec) / 2
    combined_vec = combined_vec / combined_vec.norm(dim=-1, keepdim=True)

    collection.add(
        ids=[img_file.name],
        embeddings=[combined_vec.cpu().numpy().flatten().tolist()],
        metadatas=[{
            "filename": img_file.name,
            "category": tag["category"],
            "color": tag.get("color", ""),
            "mood": ", ".join(tag.get("mood", [])),
            "occasion": ", ".join(tag.get("occasion", [])),
            "detail": ", ".join(tag.get("detail", []))
        }]
    )
    print(f"저장 완료: {img_file.name}")

print(f"\nDB 업데이트 완료: 총 {collection.count()}개")