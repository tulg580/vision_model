from seg_multi import *
import numpy as np
from PIL import Image

model = keras.models.load_model("walkable_seg_multi.keras", compile=False)
N = NUM_CLASSES

cm = np.zeros((N, N), np.int64)                 # 행=실제, 열=예측
for x, y in make_ds("val", False):
    pred = np.argmax(model.predict(x, verbose=0), axis=-1).ravel()
    true = y.numpy().ravel()
    cm += np.bincount(true * N + pred, minlength=N * N).reshape(N, N)

print("\n[클래스별 IoU]")
for i, name in enumerate(CLASS_NAMES):
    union = cm[i].sum() + cm[:, i].sum() - cm[i, i]
    share = 100 * cm[i].sum() / cm.sum()
    iou = f"{cm[i, i] / union:.3f}" if cm[i].sum() > 0 else "  -  (데이터 없음)"
    print(f"  {name:10s} IoU {iou}   (전체 픽셀의 {share:.1f}%)")

# 보행 가능(인도+횡단보도) vs 불가능으로 합쳐서 안전 지표 계산
walk = np.zeros(N, bool)
walk[WALKABLE] = True
tp = cm[walk][:, walk].sum()
fp = cm[~walk][:, walk].sum()                   # 막힌 곳을 걸을 수 있다고 한 픽셀
fn = cm[walk][:, ~walk].sum()
tn = cm[~walk][:, ~walk].sum()
print("\n[보행 가능 / 불가능으로 합친 결과]")
print(f"  보행가능 정밀도 : {tp / max(tp + fp, 1):.3f}  (걸을 수 있다고 한 곳이 실제로 맞는 비율)")
print(f"  보행가능 재현율 : {tp / max(tp + fn, 1):.3f}")
print(f"  막힌 곳 재현율  : {tn / max(tn + fp, 1):.3f}  (막힌 곳을 막혔다고 맞힌 비율, 안전상 중요)")

# 예측을 색으로 덧씌운 미리보기 저장
PALETTE = np.array([[0, 0, 0], [128, 64, 128], [244, 35, 232], [255, 255, 0],
                    [255, 0, 0], [0, 0, 142], [255, 165, 0]], np.uint8)   # RGB
Path("seg_preview_multi").mkdir(exist_ok=True)
for p in sorted(Path("seg_data/val/images").glob("*.jpg"))[::60][:8]:
    im = Image.open(p).convert("RGB")
    cls = np.argmax(model.predict(np.array(im, dtype="float32")[None], verbose=0)[0], axis=-1)
    arr = np.array(im)
    m = cls != 0
    arr[m] = (0.5 * arr[m] + 0.5 * PALETTE[cls][m]).astype("uint8")
    Image.fromarray(arr).save(f"seg_preview_multi/{p.stem}.png")
print("\nseg_preview_multi 폴더에서 결과 이미지를 확인하세요")
print("색: 도로=보라, 인도=분홍, 횡단보도=노랑, 사람=빨강, 차량=파랑, 기둥/표지판=주황")
