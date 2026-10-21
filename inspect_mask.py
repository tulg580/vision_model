from pathlib import Path
from PIL import Image
import numpy as np

# 1. MASK 폴더 안의 PNG 파일 하나 확인
png_files = list(Path("Surface_1").rglob("*.png"))
print("총 발견된 PNG 마스크 수:", len(png_files))

if png_files:
    sample_png = png_files[0]
    print("샘플 PNG 경로:", sample_png)
    im = Image.open(sample_png)
    print("이미지 모드(Mode):", im.mode, "/ 크기:", im.size)
    arr = np.array(im)
    print("배열 차원(Shape):", arr.shape)
    
    if arr.ndim == 2:
        print("고유 픽셀값 (클래스 ID):", np.unique(arr))
    else:
        unique_colors = np.unique(arr.reshape(-1, arr.shape[-1]), axis=0)
        print(f"고유 색상 수: {len(unique_colors)}")
        print("색상 샘플 (상위 8개):")
        for c in unique_colors[:8]:
            print(" ", list(c))

# 2. XML 파일에 클래스/색상 정의가 있는지 확인
xml_files = list(Path("Surface_1").rglob("*.xml"))
if xml_files:
    sample_xml = xml_files[0]
    print("\n샘플 XML 경로:", sample_xml)
    with open(sample_xml, "r", encoding="utf-8", errors="ignore") as f:
        print("--- XML 상위 내용 ---")
        for _ in range(25):
            line = f.readline()
            if not line:
                break
            print(line, end="")