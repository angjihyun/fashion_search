# step2.py - CLIP 벡터화 + ChromaDB 저장
# step6으로 대체 (사진만 보고 벡터 만들어서 db저장-> 사진+태그 같이 보고 벡터 만들어서 db저장)
import clip
import torch
import chromadb
from PIL import Image
from pathlib import Path

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print("CLIP 로드 완료")

# ChromaDB 초기화
client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection("fashion")

for img_file in Path("./data/processed").iterdir():
    if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue

    img = Image.open(img_file).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model.encode_image(tensor)
        vec = vec / vec.norm(dim=-1, keepdim=True)

    collection.add(
        ids=[img_file.name],
        embeddings=[vec.cpu().numpy().flatten().tolist()],
        metadatas=[{"filename": img_file.name}]
    )
    print(f"저장 완료: {img_file.name}")

print(f"\nDB에 총 {collection.count()}개 저장 완료")