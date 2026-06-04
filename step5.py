# step5.py - llava 태그 생성 (새 파일만)
# data/processed 사진을 llava에 이미지와 프롬프트-->태그 생성함
# results/tags.json 저장
# step5.py - llava 태그 생성 (상세 버전)
import ollama
import json
import base64
from pathlib import Path

tags_path = Path("./results/tags.json")
tags_data = {}  # 기존 태그 무시하고 전부 새로 뽑기

def generate_tags(img_path):
    with open(img_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()
    
    prompt = prompt = """You are a professional fashion stylist. Analyze this clothing item in extreme detail.
Respond ONLY in this JSON format, nothing else:
{
  "category": "one of: top/bottom/shoes/bag/hat",
  "color": "specific color name (e.g. ivory, charcoal, cobalt blue, sage green)",
  "fit": "one of: oversized/fitted/cropped/relaxed/slim/flared/straight",
  "length": "one of: mini/midi/maxi/cropped/regular (for tops: short/medium/long)",
  "neckline": "one of: v-neck/crew/collar/off-shoulder/halter/turtleneck/none (for non-tops write none)",
  "sleeve": "one of: sleeveless/short/long/3quarter/none (for non-tops write none)",
  "mood": ["2-3 mood tags from: feminine/edgy/minimal/romantic/sporty/casual/formal/vintage/streetwear/bohemian/preppy/grunge/chic/cute/elegant/classic/retro"],
  "occasion": ["1-2 tags from: daily/formal/party/outdoor/office/date/workout/school"],
  "material": "one of: cotton/denim/leather/knit/mesh/wool/silk/polyester/lace/chiffon",
  "detail": ["2-3 VERY SPECIFIC detail tags - describe exact visual features you can see. Examples: 'three white stripes on sleeve', 'NY Yankees embroidered logo', 'gold button closure', 'frayed hem', 'cargo pockets on thigh', 'sheer mesh overlay', 'ribbon tie at waist' - NEVER write just 'logo' or 'pattern'"],
  "description": "2 detailed English sentences describing the item's appearance and style"
}"""
    
    response = ollama.chat(
        model="llava",
        messages=[{"role": "user", "content": prompt, "images": [img_data]}]
    )
    text = response["message"]["content"].strip()
    text = text[text.find("{"):text.rfind("}")+1]
    return json.loads(text)

for img_file in Path("./data/processed").iterdir():
    if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    try:
        tags = generate_tags(img_file)
        tags_data[img_file.name] = tags
        print(f"완료: {img_file.name} → {tags}")
    except Exception as e:
        print(f"실패: {img_file.name} → {e}")

with open(tags_path, "w", encoding="utf-8") as f:
    json.dump(tags_data, f, ensure_ascii=False, indent=2)

print(f"\n총 {len(tags_data)}개 태그 생성 완료")