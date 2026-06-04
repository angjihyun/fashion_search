# step1.py - 배경 제거
# data/raw에서 rembg로 배경제거해 data/processed에 저장
# step1.py - 배경 제거 (새 파일만)
from rembg import remove, new_session
from PIL import Image
from pathlib import Path

session = new_session("u2net")
print("모델 로드 완료")

for img_file in Path("./data/raw").iterdir():
    if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        continue
    
    out_path = Path("./data/processed") / (img_file.stem + ".jpg")
    
    # 이미 processed에 있으면 스킵
    if out_path.exists():
        print(f"스킵: {img_file.name}")
        continue
    
    img = Image.open(img_file).convert("RGBA")
    result = remove(img, session=session)
    
    # 투명 배경 → 흰 배경
    background = Image.new("RGB", result.size, (255, 255, 255))
    background.paste(result, mask=result.split()[3])
    background.save(out_path)
    print(f"완료: {img_file.name}")

print("전체 완료!")