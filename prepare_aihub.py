import os
import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

SRC_DIR = Path("Surface_1")
DST_DIR = Path("seg_data_aihub")

OUT_W, OUT_H = 512, 224
MAX_SAMPLES = 2000      # 파인튜닝용 2,000장 샘플링
VAL_RATIO = 0.15        # 검증용 15%

# AI Hub 인도보행 공식 RGB 색상 -> 우리 모델 7개 클래스 번호 매핑
COLOR_MAP = {
    # 2: 인도 (보도블록, 아스팔트 인도, 점자블록, 시멘트, 자전거길 등)
    (0, 0, 255): 2,        # blocks5 (보도블럭!)
    (128, 128, 128): 2,    # asphalt5 (아스팔트 인도)
    (217, 217, 217): 2,    # cement5 (시멘트 인도)
    (55, 86, 35): 2,       # damaged5 (파손 인도)
    (110, 168, 70): 2,     # other5 (기타 인도)
    (255, 230, 153): 2,    # soil_stone5 (흙/돌 보행로)
    (198, 89, 17): 2,      # urethane (우레탄 인도)
    (255, 255, 0): 2,      # normal2 (점자블록)
    (128, 96, 0): 2,       # damaged2 (파손 점자블록)
    (255, 155, 155): 2,    # bike_lane1 (자전거 도로)

    # 3: 횡단보도
    (255, 0, 255): 3,      # crosswalk4 (차도 내 횡단보도)
    (208, 88, 255): 3,     # crosswalk0 (골목길 횡단보도)

    # 1: 도로 (차도 및 골목길)
    (255, 128, 255): 1,    # normal4 (차도)
    (230, 170, 255): 1,    # normal0 (골목길 노면)
    (138, 60, 200): 1,     # speed_bump0 (과속방지턱)
    (88, 38, 128): 1,      # damaged0 (파손 도로)

    # 그 외 색상(배경 [0,0,0], 맨홀, 계단, 가로수구역 등)은 전부 기본값 0 (other)
}


def rgb_to_class_id(mask_rgb):
    """(H, W, 3) RGB 마스크를 (H, W) 클래스 번호(0~6)로 변환"""
    out = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    for (r, g, b), cid in COLOR_MAP.items():
        match = (mask_rgb[:, :, 0] == r) & (mask_rgb[:, :, 1] == g) & (mask_rgb[:, :, 2] == b)
        out[match] = cid
    return out


def main():
    # MASK 폴더 안의 모든 png 마스크 수집
    mask_files = sorted(SRC_DIR.rglob("MASK/*.png"))
    print(f"발견된 PNG 마스크: {len(mask_files)}개")

    # 원본 jpg와 짝맞추기 (MASK 폴더의 부모 폴더에 원본 jpg가 위치함)
    pairs = []
    for mf in mask_files:
        img_f = mf.parent.parent / f"{mf.stem}.jpg"
        if img_f.exists():
            pairs.append((img_f, mf))

    print(f"유효한 (이미지, 마스크) 쌍: {len(pairs)}개")
    random.seed(42)
    random.shuffle(pairs)
    if MAX_SAMPLES:
        pairs = pairs[:MAX_SAMPLES]

    val_cnt = int(len(pairs) * VAL_RATIO)
    splits = {
        "val": pairs[:val_cnt],
        "train": pairs[val_cnt:]
    }

    class_names = ["other", "road", "sidewalk", "crosswalk", "person", "vehicle", "pole_sign"]

    for split, items in splits.items():
        img_out = DST_DIR / split / "images"
        msk_out = DST_DIR / split / "masks_multi"
        img_out.mkdir(parents=True, exist_ok=True)
        msk_out.mkdir(parents=True, exist_ok=True)

        counts = np.zeros(7, np.int64)
        print(f"\n[{split}] {len(items)}장 변환 시작...")

        for img_p, msk_p in items:
            stem = img_p.stem
            
            # RGB 형태로 로드
            img = np.array(Image.open(img_p).convert("RGB"))
            mask = np.array(Image.open(msk_p).convert("RGB"))
            h, w = img.shape[:2]

            # 모델 화면비(512:224)에 맞게 상단 하늘은 자르고 하단 노면을 유지
            target_h = int(w * OUT_H / OUT_W)
            if target_h <= h:
                y0 = h - target_h
                img = img[y0:h, :]
                mask = mask[y0:h, :]
            else:
                target_w = int(h * OUT_W / OUT_H)
                x0 = (w - target_w) // 2
                img = img[:, x0:x0 + target_w]
                mask = mask[:, x0:x0 + target_w]

            # 리사이즈
            img_res = cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
            mask_res = cv2.resize(mask, (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST)

            # 색상 -> 클래스 ID(0~6) 변환
            class_mask = rgb_to_class_id(mask_res)

            # 저장
            Image.fromarray(img_res).save(img_out / f"{stem}.jpg", quality=95)
            Image.fromarray(class_mask).save(msk_out / f"{stem}.png")

            counts += np.bincount(class_mask.ravel(), minlength=7)

        print(f"[{split}] 변환 완료! 클래스별 픽셀 분포:")
        for name, c in zip(class_names[:4], counts[:4]):
            print(f"   {name:10s}: {100 * c / max(counts.sum(), 1):5.1f}%")

    print("\n전체 변환 성공! seg_data_aihub/ 폴더가 정상 생성되었습니다.")


if __name__ == "__main__":
    main()