# generate_tags.py - 파인튜닝된 모델로 태그 생성
import torch
import json
import os
from PIL import Image
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

# 베이스 모델 로드
print("모델 로딩 중...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
base_model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# LoRA 가중치 로드
model = PeftModel.from_pretrained(base_model, "/workspace/llava_fashion_lora")
model.eval()
print("로드 완료!")

def generate_tag(img_path):
    img = Image.open(img_path).convert("RGB")
    prompt = """USER: <image>
Analyze this clothing item and respond ONLY in this JSON format, nothing else:
{
  "category": "one of: top/bottom/shoes/bag/hat",
  "color": "specific color",
  "fit": "one of: oversized/fitted/cropped/relaxed/slim",
  "mood": ["2-3 mood tags"],
  "occasion": ["1-2 occasion tags"],
  "material": "material type",
  "detail": ["2-3 specific detail tags"]
}
ASSISTANT:"""

    inputs = processor(text=prompt, images=img, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    result = processor.decode(output[0], skip_special_tokens=True)
    result = result.split("ASSISTANT:")[-1].strip()
    try:
        start = result.find("{")
        end = result.rfind("}") + 1
        return json.loads(result[start:end])
    except:
        return {"error": result}

# 우리 옷 사진 태그 생성
tags = {}
processed_dir = "/workspace/processed"

for img_file in tqdm(os.listdir(processed_dir)):
    if not img_file.endswith(('.jpg', '.jpeg', '.png')):
        continue
    img_path = os.path.join(processed_dir, img_file)
    tags[img_file] = generate_tag(img_path)
    print(f"{img_file}: {tags[img_file]}")

with open("/workspace/tags_finetuned.json", "w", encoding="utf-8") as f:
    json.dump(tags, f, ensure_ascii=False, indent=2)

print(f"\n완료! {len(tags)}개 태그 생성")
