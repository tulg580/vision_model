"""ARiA 앱 규격(512x288, 0~1 입력, FP16)으로 TFLite 모델을 내보냅니다."""
import numpy as np
from pathlib import Path
import seg_multi as sm
from seg_multi import *

APP_H, APP_W = 288, 512
USE_FP16 = True

# 파인튜닝된 가중치가 있으면 우선 로드, 없으면 기본 모델 로드
weights_path = Path("walkable_seg_finetuned.keras")
if not weights_path.exists():
    weights_path = Path("walkable_seg_multi.keras")

print(f"로드할 가중치: {weights_path}")
old = keras.models.load_model(str(weights_path), compile=False)

sm.H, sm.W = APP_H, APP_W
new = sm.build_model(weights=None)
new.set_weights(old.get_weights())

inp = keras.Input((APP_H, APP_W, 3), batch_size=1)
out = new(layers.Rescaling(255.0)(inp))
export = keras.Model(inp, out)

converter = tf.lite.TFLiteConverter.from_keras_model(export)
if USE_FP16:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()
output_name = "walkable_seg_aria5.tflite"
Path(output_name).write_bytes(tflite_model)
print(f"저장 완료: {output_name} ({len(tflite_model) / 1024 / 1024:.2f} MB)")

# 검증
interp = tf.lite.Interpreter(model_path=output_name)
interp.allocate_tensors()
i, o = interp.get_input_details()[0], interp.get_output_details()[0]
print("입력 텐서:", i["shape"], i["dtype"].__name__, "/ 출력 텐서:", o["shape"], o["dtype"].__name__)

x01 = np.random.default_rng(0).random((1, APP_H, APP_W, 3)).astype("float32")
ref = new.predict(x01 * 255.0, verbose=0)[0]
interp.set_tensor(i["index"], x01)
interp.invoke()
res = interp.get_tensor(o["index"])[0]
print("최대 오차:", float(np.abs(ref - res).max()))
print("클래스 일치율:", float((ref.argmax(-1) == res.argmax(-1)).mean()))
