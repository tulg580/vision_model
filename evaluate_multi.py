from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from seg_multi import *

# 1. 파인튜닝 모델이 있으면 우선 로드, 없으면 기본 모델 로드
MODEL_NAME = "walkable_seg_finetuned.keras" if Path("walkable_seg_finetuned.keras").exists() else "walkable_seg_multi.keras"
print(f">>> 평가 대상 모델 로드 중: {MODEL_NAME}")
model = keras.models.load_model(MODEL_NAME, compile=False)

N = NUM_CLASSES
PALETTE = np.array([[0, 0, 0], [255, 57, 57], [57, 255, 57], [57, 57, 255],
                    [255, 255, 57], [255, 57, 255], [57, 255, 255]], np.uint8)


def make_eval_ds(data_root, split="val"):
    """평가 대상 폴더(Cityscapes 또는 AI Hub)의 데이터셋 로더"""
    imgs = sorted(Path(f"{data_root}/{split}/images").glob("*.jpg"))
    img_paths = [str(p) for p in imgs]
    msk_paths = [str(p.parent.parent / "masks_multi" / (p.stem + ".png")) for p in imgs]
    
    ds = tf.data.Dataset.from_tensor_slices((img_paths, msk_paths))

    def load(ip, mp):
        img = tf.cast(tf.image.decode_jpeg(tf.io.read_file(ip), channels=3), tf.float32)
        m = tf.cast(tf.image.decode_png(tf.io.read_file(mp), channels=1), tf.int32)
        img.set_shape((H, W, 3))
        m.set_shape((H, W, 1))
        return img, tf.squeeze(m, -1)

    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)


def run_evaluation(data_root, title, preview_folder):
    """지정된 데이터셋에 대해 혼동 행렬, 클래스별 IoU, 전체 mIoU, 안전 지표 산출"""
    val_path = Path(f"{data_root}/val/images")
    if not val_path.exists() or len(list(val_path.glob("*.jpg"))) == 0:
        print(f"\n[건너뜀] {data_root}/val 폴더가 비어 있거나 존재하지 않습니다.")
        return

    print(f"\n{'='*60}")
    print(f"▶ [{title}] 성능 평가 진행 중... ({data_root})")
    print(f"{'='*60}")

    cm = np.zeros((N, N), np.int64)  # 행=실제, 열=예측
    ds = make_eval_ds(data_root, "val")

    for x, y in ds:
        pred = np.argmax(model.predict(x, verbose=0), axis=-1).ravel()
        true = y.numpy().ravel()
        cm += np.bincount(true * N + pred, minlength=N * N).reshape(N, N)

    # 1. 클래스별 IoU 및 전체 mIoU 계산
    ious = []
    print("\n[클래스별 IoU]")
    for i, name in enumerate(CLASS_NAMES):
        total_pixels = cm[i].sum()
        pred_pixels = cm[:, i].sum()
        
        # 데이터셋에 해당 클래스가 실제로 1픽셀이라도 존재하는 경우만 IoU 계산
        if total_pixels > 0 or pred_pixels > 0:
            union = total_pixels + pred_pixels - cm[i, i]
            iou = cm[i, i] / max(union, 1)
            ious.append(iou)
            share = 100 * total_pixels / cm.sum()
            print(f"  {name:10s} IoU: {iou:.3f}   (실제 정답 비율: {share:5.1f}%)")
        else:
            print(f"  {name:10s} IoU:   -    (데이터셋에 없음)")

    # 유효 클래스들의 산술 평균 = mIoU
    miou = np.mean(ious) if ious else 0.0
    print("  " + "-" * 42)
    print(f"  ⭐ 전체 Mean IoU (mIoU) : {miou:.3f} ⭐")
    print("  " + "-" * 42)

    # 2. 인도(sidewalk) 혼동 분석
    SW = CLASS_NAMES.index("sidewalk")
    if cm[SW].sum() > 0:
        print("\n[실제 인도 픽셀이 예측된 비율 (%)]")
        for n, v in zip(CLASS_NAMES, 100 * cm[SW] / max(cm[SW].sum(), 1)):
            print(f"  -> {n:10s} {v:5.1f}%")
        print("\n[인도로 예측된 픽셀의 실제 클래스 (%)]")
        for n, v in zip(CLASS_NAMES, 100 * cm[:, SW] / max(cm[:, SW].sum(), 1)):
            print(f"  <- {n:10s} {v:5.1f}%")

    # 3. 보행 안전 지표 (보행 가능 = 인도 + 횡단보도)
    walk = np.zeros(N, bool)
    walk[WALKABLE] = True
    tp = cm[walk][:, walk].sum()
    fp = cm[~walk][:, walk].sum()  # 장애물인데 걸을 수 있다고 오인한 픽셀
    fn = cm[walk][:, ~walk].sum()
    tn = cm[~walk][:, ~walk].sum()

    print("\n[보행 안전 통합 지표]")
    print(f"  보행가능 정밀도 : {tp / max(tp + fp, 1):.3f} (인도/횡단보도라고 한 곳이 실제 맞을 확률)")
    print(f"  보행가능 재현율 : {tp / max(tp + fn, 1):.3f} (실제 인도/횡단보도를 놓치지 않고 찾은 확률)")
    print(f"  막힌 곳 재현율  : {tn / max(tn + fp, 1):.3f} (장애물/도로를 막혔다고 잘 피한 확률, 안전상 중요)")

    # 4. 시각화 미리보기 저장
    preview_path = Path(preview_folder)
    preview_path.mkdir(parents=True, exist_ok=True)
    val_files = sorted(Path(f"{data_root}/val/images").glob("*.jpg"))
    sample_step = max(1, len(val_files) // 8)

    for p in val_files[::sample_step][:8]:
        im = Image.open(p).convert("RGB")
        cls = np.argmax(model.predict(np.array(im, dtype="float32")[None], verbose=0)[0], axis=-1)
        arr = np.array(im)
        m = cls != 0
        arr[m] = (0.5 * arr[m] + 0.5 * PALETTE[cls][m]).astype("uint8")
        Image.fromarray(arr).save(preview_path / f"{p.stem}.png")

    print(f"\n미리보기 결과 저장 완료: '{preview_folder}/' 폴더를 확인하세요.")


if __name__ == "__main__":
    # 1. 한국 보행로 검증셋 평가 (보도블록, 횡단보도 mIoU 확인)
    run_evaluation(
        data_root="seg_data_aihub",
        title="AI Hub 한국 보행로 검증셋",
        preview_folder="seg_preview_aihub"
    )

    # 2. 기존 Cityscapes 검증셋 평가 (차량, 볼라드 지식이 잊혀지지 않았는지 확인)
    run_evaluation(
        data_root="seg_data",
        title="Cityscapes 검증셋 (기존 지식 보존 평가)",
        preview_folder="seg_preview_city"
    )
