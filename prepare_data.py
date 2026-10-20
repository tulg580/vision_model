import numpy as np
from pathlib import Path
from PIL import Image

SRC = Path("cityscapes")
DST = Path("seg_data")

# Cityscapes labelIds: road=7, sidewalk=8, parking=9, terrain=22
# 용도에 맞게 바꾸세요 (예: 인도 전용이면 {8})
WALKABLE = {7, 8, 9, 22}

CROP_H = 896          # 아래쪽 보닛이 나오는 영역 제거 (2048x1024 -> 2048x896)
OUT_W, OUT_H = 512, 224

for split in ["train", "val"]:
    (DST / split / "images").mkdir(parents=True, exist_ok=True)
    (DST / split / "masks").mkdir(parents=True, exist_ok=True)
    files = sorted((SRC / "gtFine" / split).rglob("*_gtFine_labelIds.png"))
    print(split, len(files), "장 처리")
    for lab_path in files:
        stem = lab_path.name.replace("_gtFine_labelIds.png", "")
        city = lab_path.parent.name
        img_path = SRC / "leftImg8bit" / split / city / f"{stem}_leftImg8bit.png"

        img = Image.open(img_path).convert("RGB").crop((0, 0, 2048, CROP_H))
        img = img.resize((OUT_W, OUT_H), Image.Resampling.BILINEAR)

        lab = np.array(Image.open(lab_path))[:CROP_H]
        mask = np.isin(lab, list(WALKABLE)).astype(np.uint8) * 255
        mask = Image.fromarray(mask).resize((OUT_W, OUT_H), Image.Resampling.NEAREST)

        img.save(DST / split / "images" / f"{stem}.jpg", quality=95)
        mask.save(DST / split / "masks" / f"{stem}.png")
print("완료")
