# 보행 환경 인식 세그멘테이션 모델 (AI 모델 파트)

카메라 영상에서 **걸을 수 있는 영역(인도·횡단보도)과 막힌 영역(도로·사람·차량·기둥/표지판 등)을 픽셀 단위로 구분**하는 경량 시맨틱 세그멘테이션 모델
안드로이드 앱에서 온디바이스(LiteRT/TFLite)로 실행하는 것을 목표로 하며, 이 문서는 모델 학습부터 앱 연동까지 AI 파트의 작업 내용을 정리


---

## 1. 한눈에 보기

| 항목 | 내용 |
|---|---|
| 태스크 | 시맨틱 세그멘테이션 (7클래스) |
| 모델 | MobileNetV2 인코더(ImageNet 사전학습) + 경량 U-Net 디코더 |
| 학습 데이터 | Cityscapes (라벨을 7클래스로 합쳐서 사용) |
| 학습 입력 | 224 × 512 (H × W), RGB, 0~255 |
| 앱용 입력 | 288 × 512 (H × W), RGB, 0~1 float32 |
| 출력 | `[1, H, W, 7]` 클래스별 확률(softmax) → 픽셀별 argmax |
| 배포 형식 | `.tflite` (LiteRT 호환), 앱에서 직접 불러와 테스트 |

### 파이프라인

```mermaid
flowchart LR
  A["Cityscapes"] --> B["prepare_data.py<br/>크롭 · 리사이즈"]
  B --> C["prepare_data_multi.py<br/>7클래스 마스크"]
  C --> D["train_multi.py<br/>학습"]
  D --> E["evaluate_multi.py<br/>평가"]
  D --> F["convert_multi.py<br/>TFLite 변환"]
  F --> G["export_for_aria.py<br/>앱 규격으로 내보내기"]
  G --> H["ARiA 앱<br/>Load Model"]
```

---

## 2. 설계 결정 기록

| 결정 | 이유 |
|---|---|
| 이미지 단위 이진분류 → **픽셀 단위 세그멘테이션** | 분류는 "장애물 있음/없음"만 알려주지만, 세그멘테이션은 길이 어느 쪽에 열려 있는지(좌/중/우)까지 알 수 있음. 학습하지 않은 물체도 "걸을 수 있는 바닥이 아니면" 막힌 곳으로 처리됨 |
| 객체 검출(OD) 모델은 제외 | 박스/이름보다 "걸을 수 있는 영역"이 이 프로젝트에 더 직접적이고, 모델 하나로 단순하게 유지하기 위함 |
| **도로는 보행 불가로 분류** | 보행 가능 = 인도 + 횡단보도 |
| 공개 사전학습 세그멘테이션 모델을 그대로 쓰지 않음 | 공개 모델은 자기 데이터셋의 클래스 체계(예: Cityscapes 19클래스)를 따라서 원하는 6클래스와 맞지 않음. 라벨을 합쳐서 직접 학습 |
| 시맨틱(영역) 세그멘테이션 | 같은 클래스의 객체는 하나의 영역으로 묶임 (사람 2명 → 하나의 `person` 영역). 개별 객체 구분(인스턴스)은 모바일에서 무거워 제외 |
| 인코더로 MobileNetV2 사용 | U-Net 스킵 연결 레이어가 잘 알려져 있고 TFLite 변환이 안정적 |
| 학습 이미지 아래쪽 크롭 | Cityscapes 일부 장면에 자동차 보닛이 찍혀 있어, 그대로 두면 "화면 아래 가운데는 항상 막힘"으로 학습됨 |

---

## 3. 클래스 정의

| ID | 클래스 | Cityscapes 원본 라벨 | 보행 가능 |
|---|---|---|---|
| 0 | `other` (기타·막힘) | 위에 해당하지 않는 모든 것 (건물, 벽, 울타리, 식생, 지형, 하늘 등) | ✗ |
| 1 | `road` (도로) | road, parking, rail track | ✗ |
| 2 | `sidewalk` (인도) | sidewalk | ✓ |
| 3 | `crosswalk` (횡단보도) | **없음** (Cityscapes는 도로에 포함) | ✓ |
| 4 | `person` (사람) | person, rider | ✗ |
| 5 | `vehicle` (차량) | car, truck, bus, caravan, trailer, train, motorcycle, bicycle | ✗ |
| 6 | `pole_sign` (기둥·차단봉·표지판) | pole, polegroup, traffic light, traffic sign | ✗ |

- **보행 가능 클래스는 `{2, 3}`** (`seg_multi.py`의 `WALKABLE`).
- 클래스 개수와 순서는 앱 코드(`WalkableSegmenter.kt`의 `NAMES`)와 반드시 같아야 함.
- 횡단보도(3)는 Cityscapes에 별도 라벨이 없어서 **현재 학습 데이터에는 비어 있음** 구조는 7클래스로 고정해 두었기 때문에, 데이터를 추가해 재학습해도 앱 코드는 바뀌지 않음.

---

## 4. 데이터

### 4.1 Cityscapes

- 학습 2,975장 / 검증 500장 사용. 시험(test) 셋은 정답이 공개되지 않아 사용하지 않음.
- 다운로드: [cityscapes-dataset.com](https://www.cityscapes-dataset.com) 로그인 후 `gtFine_trainvaltest.zip`, `leftImg8bit_trainvaltest.zip`
- 압축을 품 ("현재 폴더에 풀기"로 풀어서 폴더가 한 단계 더 생기지 않게 주의).

```
cityscapes/
  leftImg8bit/{train,val,test}/<도시>/*_leftImg8bit.png
  gtFine/{train,val,test}/<도시>/*_gtFine_labelIds.png
```

> ⚠️ **라이선스**: Cityscapes는 비상업적 연구 목적의 라이선스. 상업적 사용 가능 여부는 약관을 확인

### 4.2 전처리

| 단계 | 내용 |
|---|---|
| 크롭 | 2048×1024 → 위쪽 2048×896 (아래쪽 보닛 영역 제거) |
| 리사이즈 | 512×224 (이미지는 bilinear, 마스크는 nearest) |
| 라벨 변환 | `labelIds` → 7클래스 ID (위 표) |

---

## 5. 모델

```
입력 (224×512×3, 0~255)
  → Rescaling(1/127.5, offset=-1)            # MobileNetV2 입력 범위 [-1, 1]
  → MobileNetV2 인코더 (ImageNet 사전학습)
       스킵: block_1_expand_relu / block_3_expand_relu / block_6_expand_relu
             block_13_expand_relu / block_16_project
  → 디코더: [Conv2DTranspose → 스킵 Concat → Conv2D] × 4단계 (256/128/64/32)
  → Conv2DTranspose(16) → Conv2D(1×1, 7, softmax)
출력 (224×512×7)
```

구조는 합성곱으로만 이뤄져 있어서 가중치를 그대로 두고 입력 크기만 바꿔 내보낼 수 있음, (`export_for_aria.py`, 288×512). 입력 크기는 **32의 배수**

---

## 6. 학습

| 항목 | 값 |
|---|---|
| 손실 | Sparse Categorical Cross-Entropy + Multi-class Dice |
| 지표 | Mean IoU |
| 배치 | 8 |
| 증강 | 좌우 반전, 밝기(±25), 대비(0.8~1.2) |
| 1단계 | 인코더 동결, Adam 1e-3, 5 에폭 |
| 2단계 | 전체 파인튜닝, Adam 1e-4, 최대 40 에폭 |
| 콜백 | EarlyStopping(`val_miou`, patience 6, 최적 가중치 복원), ReduceLROnPlateau(×0.5, patience 3), ModelCheckpoint |
| 환경 | Windows CPU (이진 베이스라인(`legacy/`) 기준 약 5분/에폭). 오래 걸리면 Google Colab GPU 권장 |

> 학습은 최종 `model.save()`가 끝에서 한 번 실행되므로 Ctrl+C로 중간에 끄면 저장된 모델이 없을 수 있음. (ModelCheckpoint가 최고 모델을 따로 저장).

---

## 7. 평가

`evaluate_multi.py`가 아래를 출력하고, `seg_preview_multi/`에 예측 오버레이 이미지를 저장.

- **클래스별 IoU** (데이터가 없는 클래스는 `-`로 표시)
- **인도 혼동 분석**: 실제 인도 픽셀이 어느 클래스로 예측됐는지 / 인도로 예측된 픽셀의 실제 클래스
- 인도+횡단보도를 "보행 가능", 나머지를 "불가능"으로 합친 **안전 지표**
  - 보행가능 정밀도: 걸을 수 있다고 한 곳이 실제로 맞는 비율
  - 보행가능 재현율
  - **막힌 곳 재현율**: 막힌 곳을 막혔다고 맞힌 비율 → 장애물을 길로 착각하는 가장 위험한 오류와 직결되므로 우선 확인

미리보기 이미지 색은 앱과 동일: 도로=빨강, 인도=초록, 횡단보도=파랑, 사람=노랑, 차량=자홍, 기둥/표지판=청록, 기타=표시 안 함

### 결과

**이진(보행 가능/불가능) 베이스라인 (`legacy/`)** — 7클래스 모델 이전에 학습한 모델, 학습 중 로그(Epoch 14/40 시점) 기준:

| val IoU | val 보행가능 정밀도 | train IoU |
|---|---|---|
| 0.925 | 0.947 | 0.970 |

**7클래스 모델** — Cityscapes 검증 셋(500장) 기준, `evaluate_multi.py` 결과

| 클래스 | IoU | 전체 픽셀 비율 |
|---|---|---|
| other | 0.904 | 50.7% |
| road | 0.922 | 32.1% |
| sidewalk | 0.641 | 5.2% |
| crosswalk | 데이터 없음 | 0.0% |
| person | 0.617 | 1.5% |
| vehicle | 0.847 | 8.1% |
| pole_sign | 0.412 | 2.3% |

| 보행가능 정밀도 | 보행가능 재현율 | 막힌 곳 재현율 |
|---|---|---|
| 0.762 | 0.801 | 0.986 |

인도 혼동 분석 (검증 셋):

| 실제 인도 픽셀의 예측 | 비율 | 인도로 예측된 픽셀의 실제 클래스 | 비율 |
|---|---|---|---|
| 인도 | 80.1% | 인도 | 76.2% |
| 도로 | 9.8% | 도로 | 10.8% |
| 기타 | 8.0% | 기타 | 11.9% |
| 그 외 | 2.1% | 그 외 | 1.1% |

- 막힌 곳은 거의 다 막힌 곳으로 맞히지만(0.986), 걸을 수 있다고 한 픽셀의 약 24%는 실제로 보행 불가. 막힌 영역이 전체의 약 95%라서 작은 오류율도 정밀도를 크게 낮춤
- 인도, 사람, 기둥/표지판 IoU가 상대적으로 낮아 개선 여지 있음

---

## 8. 실행 방법 (Windows / PowerShell)

### 8.1 환경

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass   # 활성화 스크립트 오류가 나면
venv\Scripts\Activate.ps1
pip install tensorflow pillow opencv-python
```

> Keras 3 기준. `&&`는 PowerShell 5에서 동작하지 않으니 명령을 한 줄씩 실행.

### 8.2 데이터 준비 → 학습 → 평가 → 변환

```powershell
python prepare_data.py          # 이미지 크롭/리사이즈 → seg_data/ (이진 마스크 포함)
python prepare_data_multi.py    # 7클래스 마스크 → seg_data/*/masks_multi/
python train_multi.py           # 학습 → walkable_seg_multi.keras
python evaluate_multi.py        # 평가 + 미리보기 이미지
python convert_multi.py         # TFLite 변환 → walkable_seg_multi.tflite
```

`prepare_data_multi.py`는 `prepare_data.py`가 만든 `seg_data/*/images`를 재사용하므로 **반드시 `prepare_data.py`를 먼저** 실행.

`evaluate_multi.py`의 미리보기 색과 `test_video_multi.py`의 색은 앱(`ImageUtils.CLASS_COLORS`)과 동일.

### 8.3 영상으로 테스트 (PC)

```powershell
python test_video_multi.py      # test.mp4 → result_multi.mp4
```

`test_video_multi.py` 상단의 설정을 조정.

| 설정 | 설명 |
|---|---|
| `SOURCE` | 영상 경로, 웹캠이면 `0` |
| `FIT` | `"stretch"`(프레임 전체를 늘림) / `"crop"`(아래쪽을 모델 비율로 자름). 둘 다 돌려 비교 |
| `FREE_THRESH` | 가운데 보행 가능 비율이 이 값보다 낮으면 `OBSTACLE` (기본 0.6) |

화면 아래 45% 영역을 좌/중/우로 나눠 보행 가능 비율을 계산하고, 막힌 경우 어떤 클래스가 막았는지(`blocked by`)도 표시

---

## 9. ARiA 앱 연동

### 9.1 앱용 모델 내보내기

```powershell
python export_for_aria.py       # → walkable_seg_aria.tflite (약 21MB, float32)
```

| 항목 | 규격 |
|---|---|
| 입력 | `[1, 288, 512, 3]` float32, RGB, **0~1** (앱 정규화 방식에 맞춤) |
| 출력 | `[1, 288, 512, 7]` float32 (softmax) |
| 최소 런타임 | TFLite 2.13.0 (변환 환경 TF 2.21 기준) |

`USE_FP16 = True`로 바꾸면 파일 크기가 줄어듬 (GPU에서 유리할 수 있음, 기본은 False).

- 입력 배치를 1로 고정(`batch_size=1`)해서 내보냄. 모델 안의 shape 계산 연산(`SHAPE`/`STRIDED_SLICE`/`PACK`)이 사라져 연산 수가 94개에서 79개로 줄고 출력은 동일. 모바일(특히 GPU 위임)에서 연산이 덜 쪼개지는 효과 기대
- 입력 크기는 `APP_H`, `APP_W`로 변경 가능 (32의 배수), 재학습 불필요

### 9.2 앱에서 모델 불러오기

1. 모델 파일(`walkable_seg_aria.tflite`)을 폰 저장소로 복사
2. 앱 실행 → ⚙️ 설정 → **Load Model (.tflite)** → 파일 선택
3. 상태가 `Model Loaded`로 바뀌면 결과 마스크 표시

> 앱 소스에 `loadModelFromAssets`가 있지만 호출되는 곳이 없어서 `assets` 폴더에 넣어도 자동 로드되지 않음.

마스크 색상(앱 `CLASS_COLORS` 기준): 투명=기타 / 빨강=도로 / **초록=인도** / **파랑=횡단보도** / 노랑=사람 / 자홍=차량 / 청록=기둥·표지판

### 9.3 앱 수정 사항 (`ariaCamTest`)

아래 **3개 파일을 함께** 교체해야 함 (`LiteRtSegmenter.kt`가 `ImageUtils.kt`의 새 함수를 호출). 저장소 `android/` 폴더에 있음.

| 파일 (`app/src/main/java/com/ariateam/ariacamtest/` 아래) | 변경 |
|---|---|
| `ml/LiteRtSegmenter.kt` | 입출력 버퍼를 한 번만 만들어 재사용, 출력을 평평하게 한 번에 읽어 argmax, 구간별 시간 측정, GPU 실패 시 CPU로 자동 전환 |
| `util/ImageUtils.kt` | 프레임을 정사각형 중앙 크롭 대신 **모델 비율(512:288)로 자르기**, 세로가 남으면 아래쪽(길)을 남김. RGBA 프레임을 `toBitmap()`으로 변환, `colorForClass()` 추가 |
| `ui/CameraScreen.kt` | 분석 해상도 요청 288×288 → 640×480, 입력 폭 512, 카메라 분석 출력을 RGBA로 변경, 상단 바 아래에 구간별 시간 한 줄 표시, 표시 문구 수정 |

**폰을 가로로 들고** 테스트 (세로로 들면 16:9로 잘린 가는 띠만 입력됨).

상단 바 아래 한 줄 읽는 법 (숫자는 예시):

```
cam 18 | pre 4 run 62 post 9 | GPU
```

| 항목 | 의미 |
|---|---|
| `cam` | 카메라 프레임을 비트맵으로 변환하는 시간(ms) |
| `pre` | 입력 버퍼 준비 |
| `run` | 순수 모델 실행 (상단 `Inference` 값과 동일) |
| `post` | argmax 및 마스크 생성 |
| 마지막 | 실행 경로 (`GPU` / `CPUx4` / `CPUx4 (GPU failed)`) |


### 9.4 앱 판단 로직 (참고 구현)

`WalkableSegmenter.kt`는 마스크에서 화면 하단 영역을 좌/중/우로 나눠 **보행 가능 비율**과 **가운데를 막은 클래스**를 돌려주는 샘플.

```kotlin
val r = segmenter.analyze(bitmap)
val obstacle = r.free[1] < 0.6f      // 가운데 보행 가능 비율이 낮으면 장애물
```

> 이 앱의 `LiteRtSegmenter`는 `org.tensorflow.lite`(TFLite 2.16.1)를 사용함. 모델 로드 시 `Didn't find op for builtin opcode` 오류가 나면 LiteRT(`com.google.ai.edge.litert`) 의존성으로 올리고 import 패키지를 바꿔야 함.

---

## 10. 저장소 구조

```
.
├── README.md
├── prepare_data.py            # 이미지 크롭·리사이즈 (7클래스 마스크 생성 전에 먼저 실행)
├── prepare_data_multi.py      # 7클래스 마스크 생성
├── seg_multi.py               # 데이터 로더 · 모델 · 손실 · 클래스 정의
├── train_multi.py             # 학습
├── evaluate_multi.py          # 평가
├── convert_multi.py           # TFLite 변환
├── export_for_aria.py         # 앱 규격으로 내보내기
├── test_video_multi.py        # PC 영상 테스트
├── android/                   # 앱 코드 (ARiA 테스트 앱 수정본 + 후처리 샘플)
└── legacy/                    # 이전 단계 (현재 파이프라인에서는 사용 안 함)
```

### 현재 파이프라인 (루트)

| 파일 | 설명 |
|---|---|
| `prepare_data.py` | Cityscapes 이미지 크롭·리사이즈 + 이진 마스크 생성 (`seg_data/*/images`, `masks/`). `prepare_data_multi.py`가 이 이미지를 재사용하므로 먼저 실행 |
| `prepare_data_multi.py` | 7클래스 마스크 생성 (`masks_multi/`) |
| `seg_multi.py` | 데이터 로더, 모델 정의, 손실 함수 (클래스 목록·`WALKABLE` 포함) |
| `train_multi.py` | 2단계 학습 |
| `evaluate_multi.py` | 클래스별 IoU, 인도 혼동 분석, 안전 지표, 미리보기 |
| `convert_multi.py` | Keras → TFLite 변환 및 결과 비교 |
| `export_for_aria.py` | 앱 규격(288×512, 0~1 입력)으로 내보내기 |
| `test_video_multi.py` | 영상/웹캠으로 PC에서 결과 확인 |

### `android/`

| 파일 | 설명 |
|---|---|
| `WalkableSegmenter.kt` | 앱용 후처리 샘플 (좌/중/우 비율, 막은 클래스) |
| `LiteRtSegmenter.kt`, `ImageUtils.kt`, `CameraScreen.kt` | ARiA 테스트 앱(`ariaCamTest`) 수정본 (9.3 참고) |

### `legacy/` (이전 단계, 참고용)

| 파일 | 설명 |
|---|---|
| `seg_common.py`, `train_seg.py`, `evaluate_seg.py`, `convert_seg.py`, `test_video.py` | 이진(보행 가능/불가능) 세그멘테이션 베이스라인 |
| `train.py`, `evaluate.py`, `convert.py` | 초기 이진분류(이미지 단위, `data/` 폴더) 실험, 현재 방식으로 대체됨 |

`legacy/` 스크립트는 **저장소 루트에서** 실행 (데이터·모델 파일 경로가 루트 기준).

```powershell
python legacy\train_seg.py     # 이진 베이스라인 학습 (seg_data/ 필요)
python legacy\test_video.py    # 이진 모델 영상 테스트
```

### `.gitignore` 권장

```gitignore
venv/
__pycache__/
cityscapes/        # 데이터셋 원본은 올리지 않음
seg_data/
seg_preview*/
*.keras
saved_model_tmp/
result*.mp4
test.mp4
```


---

## 11. 알려진 한계

- **횡단보도 미학습**: Cityscapes에서는 도로에 포함되어 있어 현재 모델은 횡단보도를 도로(막힘)로 판단.
- **차단봉(볼라드) 별도 클래스 없음**: 기둥·표지판 계열(`pole_sign`)로 매핑되며, 볼라드가 정확히 이 클래스로 잡힌다고 보장되지 않음.
- **시점 차이**: Cityscapes는 자동차 앞유리 높이에서 찍은 유럽 도시 장면임. 사람 눈높이·낮은 위치·실내 환경에서는 성능이 떨어질 수 있음.
- **인스턴스 구분 불가**: 같은 클래스의 객체는 하나의 영역으로 나옴.
- **위험 지형**: 구덩이, 계단 아래 같은 지형은 색만으로는 "바닥"으로 보여 놓칠 수 있음.
- **오인 사례**: 앱 테스트에서 라바콘/볼라드는 `pole_sign`으로 잘 잡혔지만 가드레일 일부가 `vehicle`로 나오는 경우 있음.
- **낮은 카메라 위치**: 앱 테스트 카메라가 지면에서 낮아 자동차 시점인 학습 데이터와 차이가 큼.
- **속도(최적화 전)**: 병목 수정 전 앱 표시 값은 추론 약 440~450ms, 3 FPS 수준. 수정본으로 재측정 필요.
- **앱 패치/온디바이스 검증**: 앱 수정 사항은 빌드·실기기 검증이 필요함. 현재 테스트 앱은 폰 내장 카메라만 사용하며 외부 카메라 연결은 구현되어 있지 않음.

---

## 12. TODO

- [x] 7클래스 모델 학습 및 Cityscapes 검증 평가 (결과는 7장)
- [ ] 직접 촬영한 영상으로 PC 테스트 (`FIT` stretch/crop 비교)
- [ ] 앱 수정본 3개 파일 적용 후 구간별 시간(cam/pre/run/post) 재측정 (GPU/CPU 경로 확인)
- [ ] 인도가 있는 장면(인도+차도) 영상으로 인도/도로 구분 확인
- [ ] 횡단보도 데이터 추가 (예: Mapillary Vistas 등) 후 재학습 — 클래스 구조 유지
- [ ] 자체 촬영 영상으로 자동 라벨링 후 보정 → 파인튜닝
- [ ] 막힌 곳 재현율 기준 임계값(`FREE_THRESH`) 조정
- [ ] 속도 최적화 (FP16/양자화, GPU 위임 확인)
- [ ] 외부 카메라 연결 연동

---

## 13. 변경 이력

| 버전 | 내용 |
|---|---|
| v1 | 보행 가능/불가능 이진 세그멘테이션 (Cityscapes), TFLite 변환 |
| v2 | 7클래스 확장 (도로를 보행 불가로 분리, 횡단보도 클래스 자리 확보), 영상 테스트 스크립트 |
| v2.1 | ARiA 앱 연동: 앱 규격 내보내기(`export_for_aria.py`), 비율 크롭 패치 |
| v2.2 | 앱 병목 개선(버퍼 재사용·RGBA 변환·구간별 시간 표시·GPU 폴백), 내보내기 배치 고정, 평가에 인도 혼동 분석 추가, PC 미리보기 색을 앱과 통일 |

모델 학습 코드(`prepare_data_multi.py`, `seg_multi.py`, `train_multi.py`)는 v2 이후 변경 없음. v2.1, v2.2 변경은 **재학습 없이** 적용 가능 (모델 재내보내기와 앱 재빌드만 필요).
