import os
import time
from pathlib import Path
os.chdir(Path(__file__).parent)

import cv2
import numpy as np
import tensorflow as tf

SOURCE = "test.mp4"      # 영상 경로. 웹캠이면 0
OUTPUT = "result_multi.mp4"
MODEL = "walkable_seg_multi.tflite"
FIT = "stretch"          # "stretch": 프레임 전체를 512x224로 늘림 / "crop": 아래쪽을 2.29:1로 잘라 사용
FREE_THRESH = 0.6        # 가운데 보행 가능 비율이 이 값보다 낮으면 OBSTACLE
SHOW = True              # 미리보기 창 (q 키로 종료)
W, H = 512, 224

NAMES = ["other", "road", "sidewalk", "crosswalk", "person", "vehicle", "pole_sign"]
WALKABLE = [2, 3]
# 앱(ImageUtils.CLASS_COLORS)과 같은 색 (BGR): 도로=빨강, 인도=초록, 횡단보도=파랑, 사람=노랑, 차량=자홍, 기둥/표지판=청록
PALETTE = np.array([[0, 0, 0], [57, 57, 255], [57, 255, 57], [255, 57, 57],
                    [57, 255, 255], [255, 57, 255], [255, 255, 57]], np.uint8)

interp = tf.lite.Interpreter(model_path=MODEL)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]


def prepare(frame):
    h, w = frame.shape[:2]
    if FIT == "crop":
        target_h = int(w * H / W)
        if target_h <= h:
            frame = frame[h - target_h:, :]          # 아래쪽(길)을 남김
        else:
            target_w = int(h * W / H)
            x0 = (w - target_w) // 2
            frame = frame[:, x0:x0 + target_w]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)


cap = cv2.VideoCapture(SOURCE)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
writer = None
times = []

while True:
    ok, frame = cap.read()
    if not ok:
        break
    shown, small = prepare(frame)

    t0 = time.time()
    interp.set_tensor(inp["index"], small.astype(np.float32)[None])
    interp.invoke()
    cls = interp.get_tensor(out["index"])[0].argmax(-1).astype(np.uint8)   # (H, W) 클래스 번호
    times.append((time.time() - t0) * 1000)

    # 화면 아래 45% 영역을 좌/중/우로 나눠 보행 가능 비율 계산
    roi = cls[int(H * 0.55):, :]
    walk = np.isin(roi, WALKABLE)
    t = W // 3
    r = [walk[:, :t].mean(), walk[:, t:2 * t].mean(), walk[:, 2 * t:].mean()]
    obstacle = r[1] < FREE_THRESH

    # 가운데 영역에서 막힌 픽셀이 주로 어떤 클래스인지
    center = roi[:, t:2 * t][~walk[:, t:2 * t]]
    blocked_by = NAMES[int(np.bincount(center, minlength=len(NAMES)).argmax())] if center.size else "-"

    # 시각화
    fh, fw = shown.shape[:2]
    big = cv2.resize(cls, (fw, fh), interpolation=cv2.INTER_NEAREST)
    color = PALETTE[big]
    m = big != 0
    vis = shown.copy()
    vis[m] = (0.5 * shown[m] + 0.5 * color[m]).astype(np.uint8)

    y0 = int(fh * 0.55)
    cv2.line(vis, (0, y0), (fw, y0), (255, 255, 255), 1)
    cv2.line(vis, (fw // 3, y0), (fw // 3, fh), (255, 255, 255), 1)
    cv2.line(vis, (2 * fw // 3, y0), (2 * fw // 3, fh), (255, 255, 255), 1)

    scale = max(0.6, fw / 1280)
    status_color = (0, 0, 255) if obstacle else (0, 200, 0)
    cv2.putText(vis, f"L {r[0]:.2f}  C {r[1]:.2f}  R {r[2]:.2f}", (10, int(35 * scale) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
    label = f"OBSTACLE ({blocked_by})" if obstacle else "CLEAR"
    cv2.putText(vis, label, (10, int(80 * scale) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, scale * 1.2, status_color, 3)

    if writer is None:
        writer = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))
    writer.write(vis)

    if SHOW:
        cv2.imshow("walkable", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()
if times:
    print(f"평균 추론 시간(PC): {np.mean(times):.1f} ms, 처리 프레임 {len(times)}장")
print(f"결과 영상 저장: {OUTPUT}")
