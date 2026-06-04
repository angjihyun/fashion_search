# test_fashionclip.py - FashionCLIP vs CLIP 비교 테스트
import clip
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# FashionCLIP 로드
print("FashionCLIP 로딩 중...")
fclip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
fclip_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
print("FashionCLIP 로드 완료")

# 일반 CLIP 로드
device = "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# 여러 이미지로 테스트
test_images = {
    "top_pinkknit.jpg": "cute 확실",
    "topouter_charcol.jpg": "formal 확실",
    "top_adidas.jpg": "sporty 확실",
    "top_blackmeshsleeve.jpg": "edgy 확실",
}

queries = ["cute", "formal", "sporty", "edgy", "casual"]

for img_file, label in test_images.items():
    img = Image.open(f"./data/processed/{img_file}").convert("RGB")
    print(f"\n===== {img_file} ({label}) =====")

    print("일반 CLIP:")
    for q in queries:
        token = clip.tokenize([q]).to(device)
        with torch.no_grad():
            img_tensor = clip_preprocess(img).unsqueeze(0).to(device)
            img_vec = clip_model.encode_image(img_tensor)
            text_vec = clip_model.encode_text(token)
            img_vec = img_vec / img_vec.norm(dim=-1, keepdim=True)
            text_vec = text_vec / text_vec.norm(dim=-1, keepdim=True)
            sim = (img_vec @ text_vec.T).item()
        print(f"  {q}: {sim:.4f}")

    print("FashionCLIP:")
    for q in queries:
        inputs = fclip_processor(text=[q], images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = fclip_model(**inputs)
            img_vec = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_vec = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            sim = (img_vec @ text_vec.T).item()
        print(f"  {q}: {sim:.4f}")