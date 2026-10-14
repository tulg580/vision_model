from seg_multi import *
import numpy as np
from PIL import Image

model = keras.models.load_model("walkable_seg_multi.keras", compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]   # 결과가 이상하면 이 줄을 지우고 비교
tflite_model = converter.convert()
Path("walkable_seg_multi.tflite").write_bytes(tflite_model)
print(f"저장 완료: walkable_seg_multi.tflite ({len(tflite_model) / 1024 / 1024:.1f} MB)")

# Keras와 TFLite 결과 비교
interp = tf.lite.Interpreter(model_path="walkable_seg_multi.tflite")
interp.allocate_tensors()
inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
print("입력:", inp["shape"], inp["dtype"].__name__, "/ 출력:", out["shape"], out["dtype"].__name__)

p = sorted(Path("seg_data/val/images").glob("*.jpg"))[0]
x = np.array(Image.open(p).convert("RGB"), dtype="float32")[None]
k = model.predict(x, verbose=0)[0]
interp.set_tensor(inp["index"], x)
interp.invoke()
t = interp.get_tensor(out["index"])[0]
print("최대 차이:", float(np.abs(k - t).max()))
print("클래스 일치율:", float((k.argmax(-1) == t.argmax(-1)).mean()))
