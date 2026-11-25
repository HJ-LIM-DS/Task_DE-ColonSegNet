# DE-ColonSegNet: 대장내시경 용종 실시간 분할

대장내시경 영상에서 용종(Polyp)을 실시간으로 분할(Segmentation)하는 **DE-ColonSegNet(Dual-Encoder ColonSegNet)** 구현 저장소입니다. [IEEE 논문](https://ieeexplore.ieee.org/document/10590125)의 Dual-Encoder 구조를 PyTorch로 바닥부터 구현했고, 손실 함수, 데이터 통합, 샘플링, 평가, 배포 파이프라인은 이 프로젝트에서 직접 설계했습니다.

- 프로젝트 정리 (Notion): [대장내시경 용종 실시간 Segmentation](https://plant-mat-0ba.notion.site/Segmentation-3a34de2bd3e380e2bbd5c92265f6aa9c)
- 추론 결과 영상: [Google Drive](https://drive.google.com/file/d/1nKwSvz7DuSSz10ALnJ3kL_enZF1W9hfP/view)
- 학습 기록: [W&B 리포트](https://wandb.ai/mildchocolate1-none/DE-ColonSegNet_after_finetuning/reports/DE-ColonSegNet-fine-tuning--VmlldzoxNzU0NDU1NA)

## 배경

- 처음에는 무거운 EAT 모델을 ResNet50으로 지식 증류(Knowledge Distillation)했지만, Soft/Hard Loss 불균형으로 학습이 붕괴했고 작고 경계가 모호한 용종의 특징도 제대로 옮겨지지 않았습니다.
- 대장내시경에서는 작은 용종을 놓치는 미탐(False Negative)이 가장 치명적입니다. 그래서 증류 대신 DE-ColonSegNet 구조를 처음부터 구현하고, 작은 병변의 미탐을 줄이는 방향으로 학습과 평가를 설계했습니다.
- 전동식 내시경의 제어 루프에 탑재하려면 30 FPS 이상의 추론 속도가 필요해서, TensorRT 변환과 INT8 양자화로 추론을 가속했습니다.

## 결과

검증셋(Validation) 기준이며, 검증 Dice가 가장 높았던 Epoch 54의 지표입니다.

| Recall | Dice | IoU |
| :---: | :---: | :---: |
| 0.9194 | 0.8370 | 0.7527 |

추론 속도 (GeForce RTX 3060 12GB, 입력 1x3x352x352)

| 추론 환경 | 프레임당 시간 | FPS |
| --- | :---: | :---: |
| PyTorch | 43.95 ms | 22.76 |
| TensorRT FP32 | 약 30.00 ms | 33.33 |
| TensorRT INT8 (PTQ) | 약 6.80 ms | 147.13 |

TensorRT INT8 엔진은 PyTorch보다 약 6.5배 빠릅니다.

## 주요 특징

### 모델 구조

- ColonSegNet 2개를 직렬로 쌓은 구조입니다. 1단계(net1)가 예측한 마스크를 원본 이미지와 채널 방향으로 결합(3+1채널)해 2단계(net2)에 다시 넣고, 세부 병변을 보정합니다.
- 각 ColonSegNet은 Residual + SE(Squeeze-and-Excitation) 블록으로 된 인코더 블록 2개, ASPP 병목(1x1, dilation 6/12/18, 전역 평균 풀링), 스킵 연결이 있는 U-Net식 디코더로 구성됩니다.
- 입력 해상도는 352x352입니다.

### 작은 병변 대응

용종 픽셀이 영상 전체의 2% 미만이면 작은 병변(`is_small`)으로 분류하고, 학습과 평가 전반에 이 정보를 사용합니다.

- **크기 적응형 Tversky Loss**: 작은 병변은 미탐 가중치 β=0.85(α=0.15), 일반 병변은 β=0.7(α=0.3)
- **Focal Loss** (α=0.25, γ=2.0): 배경 픽셀이 대부분인 클래스 불균형 완화
- **보조 손실**: 1단계(net1) 출력에 대한 손실을 0.4배 가중해 더함
- **오버샘플링**: WeightedRandomSampler로 작은 병변 이미지를 3배 자주 샘플링
- **데이터 증강**: 뒤집기, 90도 회전, 밝기·대비·색상 변화, 가우시안 블러, 병변을 확대해 보여 주는 RandomResizedCrop(scale 0.5~1.0)
- **크기별 분리 평가**: `evaluate.py`가 전체, 일반 병변, 작은 병변으로 나눠 Dice, Recall, IoU를 집계

### 추론 가속 (TensorRT, INT8 양자화)

- PyTorch → ONNX(opset 14) → TensorRT 순서로 변환합니다. 벤치마크에서는 PyTorch CUDA 텐서의 `data_ptr()`를 TensorRT에 직접 바인딩해 불필요한 메모리 복사를 없앴습니다.
- ONNX Runtime으로 정적 INT8 PTQ(Post-Training Quantization)를 수행합니다. 가중치와 활성값 모두 대칭 INT8이며, SUN-SEG 이미지 50장으로 캘리브레이션합니다.
- TensorRT 11.x 환경에서 INT32로 양자화된 Conv Bias의 DequantizeLinear 노드를 파싱하지 못하는 오류가 발생했습니다. `fix_int8_onnx.py`는 Conv, ConvTranspose, Gemm의 Bias로 들어가는 DequantizeLinear 노드만 제거하고, (정수값 - 영점) x 스케일로 되돌린 FP32 상수를 원래 이름으로 넣어 이 문제를 해결합니다. 가중치와 활성값은 INT8 그대로 유지됩니다.

## 데이터셋

6개 공개 데이터셋을 하나의 MS COCO 형식 데이터셋으로 통합합니다 (`scripts/build_dataset.py`).

| 데이터셋 | 사용 범위 |
| --- | --- |
| Kvasir-SEG | 전체 |
| CVC-ClinicDB | 전체 (PNG 버전) |
| ETIS-LaribPolypDB | 전체 |
| BKAI-IGH NeoPolyp | train 세트 |
| PolypGen | 6개 기관 데이터 (C1~C6) |
| SUN-SEG | TrainDataset, 10프레임마다 1장 샘플링 |

- Train / Val / Test를 8 : 1 : 1로 무작위 분할합니다 (`random_state=42`).
- 같은 분할 안의 중복 이미지는 MD5 해시로 제거하고, 마스크가 비어 있는 이미지와 면적 10픽셀 미만의 작은 영역은 제외합니다.
- 데이터셋은 이 저장소에 포함되어 있지 않으며, 각 데이터셋의 라이선스와 이용 조건을 따릅니다.

데이터셋 폴더 구조 (`build_dataset.py`의 `DATASETS_BASE` 기준)

```
datasets/
├── kvasir-seg/
│   ├── images/
│   └── masks/
├── CVC-ClinicDB/PNG/
│   ├── Original/
│   └── Ground Truth/
├── ETIS-LaribPolypDB/
│   ├── images/
│   └── masks/
├── BKAI-IGH NeoPolyp/
│   ├── train/train/
│   └── train_gt/train_gt/
├── PolypGen/PolypGen2021_MultiCenterData_v3/
│   └── data_C1 ~ data_C6/
│       ├── images_C1 ~ images_C6/
│       └── masks_C1 ~ masks_C6/      # 마스크 파일명: <이미지 이름>_mask.jpg
└── SUN-SEG_dataset/TrainDataset/
    ├── Frame/
    └── GT/                           # Frame과 같은 case 폴더 구조
```

## 저장소 구조

```
├── models/
│   ├── de_colonsegnet.py           # DE-ColonSegNet 모델
│   └── losses.py                   # 크기 적응형 Tversky Loss + Focal Loss
├── datasets/
│   ├── coco_dataset.py             # COCO 형식 Dataset과 데이터 증강
│   ├── weighted_sampler.py         # 작은 병변 오버샘플링
│   └── summarize_datasets.py       # 데이터셋 폴더 스캔 결과 요약
├── scripts/
│   ├── build_dataset.py            # 6개 데이터셋 통합과 Train/Val/Test 분할 (v2)
│   ├── convert_to_coco.py          # 초기 버전 변환 스크립트 (3개 데이터셋, Train/Val)
│   ├── verify_dataset.py           # 생성된 COCO JSON 검증과 샘플 시각화
│   └── verify_weighted_sampler.py  # 오버샘플링 비율 확인
├── train.py                        # 학습
├── evaluate.py                     # 전체/일반/작은 병변 분리 평가
├── export_onnx.py                  # ONNX 내보내기
├── benchmark_trt.py                # TensorRT 기본 정밀도(FP32) 엔진 빌드와 FPS 측정
├── benchmark_trt_int8.py           # INT8 PTQ와 TensorRT INT8 엔진 빌드 시도
├── benchmark_ort_int8.py           # INT8 PTQ와 ONNX Runtime FP32/INT8 속도 비교
├── fix_int8_onnx.py                # Conv Bias의 DequantizeLinear 노드를 FP32 상수로 복원
├── run_int8_fixed.py               # 수정된 INT8 모델로 TensorRT 엔진 빌드와 FPS 측정
├── inference.py                    # PyTorch 추론과 결과 영상 생성
└── inference_trt.py                # ONNX Runtime(TensorRT EP, FP16) 추론과 결과 영상 생성
```

## 환경 설정

```bash
conda env create -f environment.yml
conda activate de_colonsegnet_env
```

개발 환경: Python 3.10, PyTorch (CUDA 11.8), TensorRT 11.x, GeForce RTX 3060 12GB

## 실행 방법

모든 명령은 저장소 루트에서 실행합니다. 체크포인트와 엔진 파일은 `checkpoints/`, 결과 영상은 `outputs/`에 저장됩니다.

### 0. 경로 설정

스크립트 안의 데이터 경로가 `E:/Task_DE-ColonSegNet/...` 절대 경로로 되어 있습니다. 실행 전에 아래 파일의 경로를 본인 환경에 맞게 바꿔 주세요.

- `scripts/build_dataset.py`: `DATASETS_BASE`, `OUTPUT_DIR`
- `scripts/verify_dataset.py`, `scripts/verify_weighted_sampler.py`: 데이터셋 경로
- `train.py`, `evaluate.py`: `train.json`, `val.json`, `test.json` 경로
- `inference.py`, `inference_trt.py`, `benchmark_trt_int8.py`, `benchmark_ort_int8.py`: SUN-SEG 데이터셋 경로

### 1. 데이터셋 구축

```bash
python scripts/build_dataset.py     # datasets/coco_format_v2/{train,val,test}.json 생성
python scripts/verify_dataset.py    # (선택) JSON 검증과 샘플 시각화
```

### 2. 학습

```bash
python train.py                     # 배치 8, 학습률 1e-4, Early Stopping patience 3
python train.py --test_run          # 동작 확인용 짧은 실행 (배치 크기 2, 3배치만)
```

- AdamW(weight decay 1e-4)와 ReduceLROnPlateau(검증 Dice 기준, factor 0.5, patience 2)를 사용합니다.
- 에포크 수 제한 없이 Early Stopping으로 종료하며, 검증 Dice가 갱신될 때마다 `checkpoints/best_model.pth`를 저장합니다.
- 중단된 학습은 `--resume --resume_best_dice <이전 최고 Dice> --resume_patience_counter <카운터> --start_epoch <시작 에포크>`로 이어서 진행합니다.

### 3. 평가

```bash
python evaluate.py                  # test.json 기준 전체/일반/작은 병변 Dice, Recall, IoU
```

### 4. ONNX 변환과 TensorRT 벤치마크

```bash
python export_onnx.py               # checkpoints/best_model.onnx 생성
python benchmark_trt.py             # TensorRT 엔진 빌드 후 1,000프레임 FPS 측정
```

### 5. INT8 양자화

```bash
python benchmark_trt_int8.py        # INT8 PTQ → checkpoints/best_model_int8.onnx
python fix_int8_onnx.py             # Bias 복원 → checkpoints/best_model_int8_fixed.onnx
python run_int8_fixed.py            # INT8 엔진 빌드 후 1,000프레임 FPS 측정
```

`benchmark_trt_int8.py`는 양자화 후 엔진 빌드까지 시도하지만, TensorRT 11.x에서는 Bias 파싱 오류로 빌드에 실패합니다. 그래서 `fix_int8_onnx.py`로 모델을 고친 뒤 `run_int8_fixed.py`로 엔진을 빌드합니다. ONNX Runtime에서 FP32와 INT8 속도를 비교하려면 `python benchmark_ort_int8.py`를 실행합니다.

### 6. 결과 영상 생성

```bash
python inference.py                 # PyTorch → outputs/sun_seg_inference_result.mp4
python inference_trt.py             # ONNX Runtime TensorRT EP(FP16) → outputs/sun_seg_inference_trt_result.mp4
```

SUN-SEG의 모든 프레임에 대해 왼쪽은 원본, 오른쪽은 예측 마스크(파란색)를 겹친 영상을 만들고, 프레임당 평균 처리 시간과 FPS를 출력합니다.

## 향후 과제

- INT8 엔진의 분할 정확도(Dice, Recall) 검증
- Test셋 기준 성능과 작은 병변 지표 정리
- NVIDIA Jetson 등 내시경용 임베디드 엣지 보드로 포팅
- 실시간으로 얻은 병변 좌표(BBox/Mask)를 내시경 로봇 제어부(Kinematics)로 전달해, 카메라가 병변 쪽으로 자동 조향하는 술기 보조 기능 연동

## 참고

- DE-ColonSegNet 원 논문: https://ieeexplore.ieee.org/document/10590125

## 라이선스

MIT License
