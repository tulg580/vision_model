import numpy as np
from pathlib import Path
from PIL import Image

SRC = Path("cityscapes")
DST = Path("seg_data")        # prepare_data.py가 만든 이미지 폴더를 그대로 재사용

CROP_H = 896                  # prepare_data.py와 동일하게 아래쪽을 잘라냄
OUT_W, OUT_H = 512, 224

# 0 기타(막힘) / 1 도로 / 2 인도 / 3 횡단보도 / 4 사람 / 5 차량 / 6 기둥·차단봉·표지판
# Cityscapes에는 횡단보도 클래스가 없어서(도로에 포함됨) 3번은 비어 있습니다
LUT = np.zeros(256, np.uint8)
for i in (7, 9, 10):
    LUT[i] = 1                # road, parking, rail track
LUT[8] = 2                    # sidewalk
for i in (24, 25):
    LUT[i] = 4                # person, rider
for i in range(26, 34):
    LUT[i] = 5                # car, truck, bus, caravan, trailer, train, motorcycle, bicycle
for i in (17, 18, 19, 20):
    LUT[i] = 6                # pole, polegroup, traffic light, traffic sign

NAMES = ["other", "road", "sidewalk", "crosswalk", "person", "vehicle", "pole_sign"]

for split in ["train", "val"]:
    out_dir = DST / split / "masks_multi"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted((SRC / "gtFine" / split).rglob("*_gtFine_labelIds.png"))
    counts = np.zeros(len(NAMES), np.int64)
    done = 0
    for lab_path in files:
        stem = lab_path.name.replace("_gtFine_labelIds.png", "")
        if not (DST / split / "images" / f"{stem}.jpg").exists():
            print("이미지 없음, 건너뜀:", stem)
            continue
        lab = np.array(Image.open(lab_path))[:CROP_H]
        mask = Image.fromarray(LUT[lab]).resize((OUT_W, OUT_H), Image.Resampling.NEAREST)
        mask.save(out_dir / f"{stem}.png")
        counts += np.bincount(np.array(mask).ravel(), minlength=len(NAMES))
        done += 1
    print(f"[{split}] {done}장 완료")
    for n, c in zip(NAMES, counts):
        print(f"   {n:10s} {100 * c / max(counts.sum(), 1):5.1f}%")
