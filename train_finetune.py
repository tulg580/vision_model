from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from seg_multi import *

BATCH = 8
EPOCHS = 15
LR = 3e-4             # 인코더 동결 시 디코더 최적 학습률


def make_split_ds(data_root, split, training=True):
    imgs = sorted(Path(f"{data_root}/{split}/images").glob("*.jpg"))
    img_paths = [str(p) for p in imgs]
    msk_paths = [str(p.parent.parent / "masks_multi" / (p.stem + ".png")) for p in imgs]

    ds = tf.data.Dataset.from_tensor_slices((img_paths, msk_paths))
    if training:
        ds = ds.shuffle(len(img_paths)).repeat()

    def load(ip, mp):
        img = tf.cast(tf.image.decode_jpeg(tf.io.read_file(ip), channels=3), tf.float32)
        m = tf.cast(tf.image.decode_png(tf.io.read_file(mp), channels=1), tf.int32)
        img.set_shape((H, W, 3))
        m.set_shape((H, W, 1))
        return img, m

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        def aug(img, m):
            flip = tf.random.uniform(()) < 0.5
            img = tf.cond(flip, lambda: tf.image.flip_left_right(img), lambda: img)
            m = tf.cond(flip, lambda: tf.image.flip_left_right(m), lambda: m)
            img = tf.image.random_brightness(img, 25.0)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            return tf.clip_by_value(img, 0.0, 255.0), m
        ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(lambda i, m: (i, tf.squeeze(m, -1)))
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


# 1. 원본 깨끗한 모델 로드 (지식 손상 없는 walkable_seg_multi.keras)
base_model_path = Path("walkable_seg_multi.keras")
if not base_model_path.exists():
    raise FileNotFoundError("원본 모델인 walkable_seg_multi.keras 파일을 찾을 수 없습니다!")

print(f">>> 원본 베이스 모델 로드: {base_model_path}")
model = keras.models.load_model(str(base_model_path), compile=False)

# 2. [핵심] 인코더 완전 동결 (차량/사람/기둥 지식 영구 보존)
encoder = model.get_layer("encoder")
encoder.trainable = False
print(f">>> 인코더 레이어 {len(encoder.layers)}개 완전 동결 완료 (디코더만 학습)")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss=ce_dice,
    metrics=[keras.metrics.MeanIoU(num_classes=NUM_CLASSES, sparse_y_true=True,
                                   sparse_y_pred=False, name="miou")]
)

# 3. 데이터셋 로드 및 혼합
print("\n데이터셋 구성 중...")
ds_city_train = make_split_ds("seg_data", "train", training=True)
ds_aihub_train = make_split_ds("seg_data_aihub", "train", training=True)

# Cityscapes 55% + AI Hub 45% (기존 사물 지식 방어 비율)
mixed_train_ds = tf.data.Dataset.sample_from_datasets(
    [ds_city_train, ds_aihub_train], weights=[0.55, 0.45]
)

# [핵심] 검증셋: Cityscapes val + AI Hub val 두 데이터를 합쳐서 통합 검증
ds_city_val = make_split_ds("seg_data", "val", training=False)
ds_aihub_val = make_split_ds("seg_data_aihub", "val", training=False)
combined_val_ds = ds_city_val.concatenate(ds_aihub_val)

# 4. 학습 시작
steps_per_epoch = 250   # 에폭당 2,000장
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_miou", mode="max", patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_miou", mode="max", factor=0.5, patience=2),
    keras.callbacks.ModelCheckpoint("walkable_seg_finetuned.keras", monitor="val_miou", mode="max", save_best_only=True)
]

print("\n>>> 지식 보존형 파인튜닝을 시작합니다...")
model.fit(
    mixed_train_ds,
    steps_per_epoch=steps_per_epoch,
    validation_data=combined_val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# 최적 가중치 저장
model.save("walkable_seg_finetuned.keras")
print("\n파인튜닝 완료! 균형 잡힌 모델 저장: walkable_seg_finetuned.keras")