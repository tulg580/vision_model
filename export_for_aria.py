"""ARiA 테스트 앱 규격(입력 512x288, 0~1 입력)에 맞춰 모델을 다시 내보냅니다.
재학습은 필요 없고, 학습된 가중치를 그대로 새 크기의 모델에 옮깁니다."""
import numpy as np
import seg_multi as sm
from seg_multi import *

APP_H, APP_W = 288, 512
USE_FP16 = False        # True면 파일 크기가 절반 정도로 줄어듭니다 (GPU에서 유리). 처음엔 False로 테스트

old = keras.models.load_model("walkable_seg_multi.keras", compile=False)

# 같은 구조를 288x512 입력으로 새로 만들고 가중치만 복사 (합성곱 모델이라 크기와 무관)
sm.H, sm.W = APP_H, APP_W
new = sm.build_model(weights=None)
new.set_weights(old.get_weights())

# 앱은 0~1로 정규화해서 넣으므로, 모델 맨 앞에서 0~255로 되돌려 줌
inp = keras.Input((APP_H, APP_W, 3))
out = new(layers.Rescaling(255.0)(inp))
export = keras.Model(inp, out)

converter = tf.lite.TFLiteConverter.from_keras_model(export)
if USE_FP16:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
Path("walkable_seg_aria.tflite").write_bytes(tflite_model)
print(f"저장 완료: walkable_seg_aria.tflite ({len(tflite_model) / 1024 / 1024:.1f} MB)")

# 검증: 0~1 입력으로 돌린 TFLite 결과 == 0~255 입력으로 돌린 원래 Keras 결과
interp = tf.lite.Interpreter(model_path="walkable_seg_aria.tflite")
interp.allocate_tensors()
i, o = interp.get_input_details()[0], interp.get_output_details()[0]
print("입력:", i["shape"], i["dtype"].__name__, "/ 출력:", o["shape"], o["dtype"].__name__)

x01 = np.random.default_rng(0).random((1, APP_H, APP_W, 3)).astype("float32")
ref = new.predict(x01 * 255.0, verbose=0)[0]
interp.set_tensor(i["index"], x01)
interp.invoke()
res = interp.get_tensor(o["index"])[0]
print("최대 차이:", float(np.abs(ref - res).max()))
print("클래스 일치율:", float((ref.argmax(-1) == res.argmax(-1)).mean()))
