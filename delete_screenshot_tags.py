# delete_screenshot_tags.py
import json

with open("./results/tags.json", encoding="utf-8") as f:
    tags = json.load(f)

before = len(tags)
cleaned = {k: v for k, v in tags.items() if not k.startswith("스크린샷")}
print(f"삭제된 키: {before - len(cleaned)}개")
print(f"남은 키: {len(cleaned)}개")

with open("./results/tags.json", "w", encoding="utf-8") as f:
    json.dump(cleaned, f, ensure_ascii=False, indent=2)

print("완료!")