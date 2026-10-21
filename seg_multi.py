import os
from pathlib import Path
os.chdir(Path(__file__).parent)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

H, W = 224, 512
BATCH = 8

CLASS_NAMES = ["other", "road", "sidewalk", "crosswalk", "person", "vehicle", "pole_sign"]
NUM_CLASSES = len(CLASS_NAMES)
WALKABLE = [2, 3]             # 인도, 횡단보도만 걸을 수 있음 (도로는 제외)


def make_ds(split, training):
    imgs = sorted(Path(f"seg_data/{split}/images").glob("*.jpg"))
    img_paths = [str(p) for p in imgs]
    msk_paths = [str(p.parent.parent / "masks_multi" / (p.stem + ".png")) for p in imgs]
    ds = tf.data.Dataset.from_tensor_slices((img_paths, msk_paths))
    if training:
        ds = ds.shuffle(len(img_paths))

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

    ds = ds.map(lambda i, m: (i, tf.squeeze(m, -1)))      # 마스크: (H, W) 클래스 번호
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


def build_model(weights="imagenet"):
    # 입력 크기가 정사각형이 아니라는 경고가 떠도 정상입니다
    base = keras.applications.MobileNetV2(
        input_shape=(H, W, 3), include_top=False, weights=weights)
    names = ["block_1_expand_relu", "block_3_expand_relu",
             "block_6_expand_relu", "block_13_expand_relu", "block_16_project"]
    encoder = keras.Model(base.input, [base.get_layer(n).output for n in names],
                          name="encoder")

    inputs = keras.Input((H, W, 3))                      # 0~255 입력
    x = layers.Rescaling(1 / 127.5, offset=-1)(inputs)   # MobileNetV2는 -1~1 입력
    *skips, x = encoder(x)
    for f, s in zip([256, 128, 64, 32], reversed(skips)):
        x = layers.Conv2DTranspose(f, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.Concatenate()([x, s])
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")(x)
    out = layers.Conv2D(NUM_CLASSES, 1, activation="softmax")(x)   # (H, W, 클래스 수)
    return keras.Model(inputs, out)


def ce_dice(y_true, y_pred):
    ce = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(y_true, y_pred))
    onehot = tf.one_hot(tf.cast(y_true, tf.int32), NUM_CLASSES)
    inter = tf.reduce_sum(onehot * y_pred, axis=(1, 2))
    denom = tf.reduce_sum(onehot, axis=(1, 2)) + tf.reduce_sum(y_pred, axis=(1, 2))
    dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
    return ce + tf.reduce_mean(dice)
