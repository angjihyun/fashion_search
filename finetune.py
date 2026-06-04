# finetune.py - LLaVA LoRA 파인튜닝
import torch
import json
import os
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

# 1. 모델 로드
print("LLaVA 로딩 중...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor.tokenizer.pad_token = processor.tokenizer.eos_token
print(f"로드 완료! VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

# 2. LoRA 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3. 데이터셋
class FashionDataset(Dataset):
    def __init__(self):
        with open("/workspace/train_data2.json") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(f"/workspace/{item['image']}").convert("RGB")
        prompt = f"USER: <image>\n{item['instruction']}\nASSISTANT: {item['output']}"
        return img, prompt

def collate_fn(batch):
    images, prompts = zip(*batch)
    inputs = processor(
        text=list(prompts),
        images=list(images),
        return_tensors="pt",
        padding=True,
        truncation=False
    )
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

# 4. 학습
dataset = FashionDataset()
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
model.train()

print("파인튜닝 시작!")
for epoch in range(5):
    total_loss = 0
    for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

# 5. 저장
model.save_pretrained("/workspace/llava_fashion_lora")
print("파인튜닝 완료! 저장: /workspace/llava_fashion_lora")