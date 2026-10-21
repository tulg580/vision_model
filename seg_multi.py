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
    base = keras.applications.MobileNetV2(
        input_shape=(H, W, 3), include_top=False, weights=weights)
    
    # 크기(해상도)가 정확히 1/2씩 맞아떨어지는 레이어들 선택
    names = ["block_1_expand_relu", "block_3_expand_relu",
             "block_6_expand_relu", "block_13_expand_relu", "block_16_project"]
    encoder = keras.Model(base.input, [base.get_layer(n).output for n in names],
                          name="encoder")

    inputs = keras.Input((H, W, 3))
    x = layers.Rescaling(1 / 127.5, offset=-1)(inputs)
    *skips, x = encoder(x)

    # 채널 수를 가볍게 축소 (128, 64, 32, 16)
    # UpSampling2D + 1x1 채널 압축 + SeparableConv2D 조합 (모바일 최적화)
    for f, s in zip([128, 64, 32, 16], reversed(skips)):
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
        # 스킵 레이어의 뚱뚱한 채널을 1x1 Conv로 32~64개로 가볍게 다이어트
        s_reduced = layers.Conv2D(f // 2, 1, padding="same", use_bias=False)(s)
        x = layers.Concatenate()([x, s_reduced])
        x = layers.SeparableConv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = layers.SeparableConv2D(16, 3, padding="same", activation="relu")(x)
    out = layers.Conv2D(NUM_CLASSES, 1, activation="softmax")(x)

    return keras.Model(inputs, out)


def ce_dice(y_true, y_pred):
    ce = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(y_true, y_pred))
    onehot = tf.one_hot(tf.cast(y_true, tf.int32), NUM_CLASSES)
    inter = tf.reduce_sum(onehot * y_pred, axis=(1, 2))
    denom = tf.reduce_sum(onehot, axis=(1, 2)) + tf.reduce_sum(y_pred, axis=(1, 2))
    dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
    return ce + tf.reduce_mean(dice)
