from seg_multi import *

train_ds = make_ds("train", True)
val_ds = make_ds("val", False)
model = build_model()


def compile_model(lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=ce_dice,
        metrics=[keras.metrics.MeanIoU(num_classes=NUM_CLASSES, sparse_y_true=True,
                                       sparse_y_pred=False, name="miou")])


# 1단계: 인코더를 얼리고 디코더만 학습
model.get_layer("encoder").trainable = False
compile_model(1e-3)
model.fit(train_ds, validation_data=val_ds, epochs=5)

# 2단계: 전체 파인튜닝
model.get_layer("encoder").trainable = True
compile_model(1e-4)
model.fit(train_ds, validation_data=val_ds, epochs=40, callbacks=[
    keras.callbacks.EarlyStopping(monitor="val_miou", mode="max", patience=6,
                                  restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_miou", mode="max", factor=0.5, patience=3),
    keras.callbacks.ModelCheckpoint("walkable_seg_multi.keras", monitor="val_miou",
                                    mode="max", save_best_only=True),
])

model.save("walkable_seg_multi.keras")
print("저장 완료: walkable_seg_multi.keras")
