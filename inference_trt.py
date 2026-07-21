import os
import glob
import time
import cv2
import numpy as np
from tqdm import tqdm
import onnxruntime as ort

from datasets.coco_dataset import get_transforms

def apply_blue_mask(image, mask, alpha=0.5):
    """
    원본 이미지 위에 파란색 분할 마스크를 반투명하게 덮어씌웁니다.
    """
    color_mask = np.zeros_like(image)
    color_mask[:, :, 0] = 255  # Blue channel max
    
    masked_part = cv2.bitwise_and(color_mask, color_mask, mask=mask)
    original_part = cv2.bitwise_and(image, image, mask=mask)
    
    blended = cv2.addWeighted(masked_part, alpha, original_part, 1 - alpha, 0)
    
    result = image.copy()
    np.copyto(result, blended, where=mask[:, :, None] > 0)
    return result

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def run_inference_trt():
    # 1. TensorRT 환경 준비 및 ONNX 로드
    onnx_path = "checkpoints/best_model.onnx"
    if not os.path.exists(onnx_path):
        print(f"ONNX 모델 파일 {onnx_path} 가 없습니다. export_onnx.py를 먼저 실행하세요.")
        return

    print("ONNX Runtime(TensorRT) 세션을 초기화합니다. (초기 엔진 빌드 시 시간이 소요될 수 있습니다.)")
    
    # TensorrtExecutionProvider 설정 (FP16 최적화 활성화)
    providers = [
        ('TensorrtExecutionProvider', {
            'device_id': 0,
            'trt_fp16_enable': True,
            'trt_engine_cache_enable': True,  # 엔진 빌드 후 캐시 저장
            'trt_engine_cache_path': './checkpoints/',
        }),
        'CUDAExecutionProvider',
        'CPUExecutionProvider',
    ]

    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    
    # 웜업(Warm-up) 실행
    print("엔진 웜업(Warm-up) 중...")
    dummy_input = np.random.randn(1, 3, 352, 352).astype(np.float32)
    for _ in range(10):
        session.run(None, {input_name: dummy_input})
    print("웜업 완료.")

    # 2. 데이터셋 로드
    transform = get_transforms("test")
    dataset_root = "E:/Task_DE-ColonSegNet/datasets/SUN-SEG_dataset"
    print("데이터셋 스캔 중...")
    
    search_pattern = os.path.join(dataset_root, "**", "Frame", "**", "*.jpg")
    all_image_paths = glob.glob(search_pattern, recursive=True)
    all_image_paths += glob.glob(os.path.join(dataset_root, "**", "Frame", "**", "*.png"), recursive=True)
    
    all_image_paths = sorted(all_image_paths)
    
    if not all_image_paths:
        print("SUN-SEG 데이터셋 이미지를 찾을 수 없습니다.")
        return
    
    print(f"총 {len(all_image_paths)}개의 이미지를 찾았습니다.")

    # 3. 비디오 저장 설정
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_video_path = os.path.join(out_dir, "sun_seg_inference_trt_result.mp4")

    INPUT_SIZE = 352
    frame_width = INPUT_SIZE * 2
    frame_height = INPUT_SIZE
    fps_video = 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(out_video_path, fourcc, fps_video, (frame_width, frame_height))

    total_forward_time = 0.0
    num_processed = 0

    print("추론 및 비디오 생성을 시작합니다...")
    
    # tqdm 진행바
    for img_path in tqdm(all_image_paths, desc="TRT Inference"):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Albumentations 적용
        augmented = transform(image=img_rgb)
        img_tensor = augmented["image"].numpy()  # ToTensorV2에 의해 shape은 (3, H, W)
        img_tensor = np.expand_dims(img_tensor, axis=0)  # (1, 3, 352, 352)
        img_tensor = img_tensor.astype(np.float32)

        # 시간 측정 (순수 TRT 추론)
        start_time = time.time()
        ort_outputs = session.run(None, {input_name: img_tensor})
        forward_time = time.time() - start_time
        
        total_forward_time += forward_time
        num_processed += 1

        # 첫 번째 출력이 main output (logits)
        logits = ort_outputs[0][0, 0, :, :] # (352, 352)
        
        # Sigmoid 및 이진화
        pred_mask = sigmoid(logits)
        binary_mask = (pred_mask > 0.5).astype(np.uint8)

        # 시각화 준비
        vis_img_bgr = cv2.resize(img_bgr, (INPUT_SIZE, INPUT_SIZE))
        overlay_img = apply_blue_mask(vis_img_bgr, binary_mask, alpha=0.5)

        # 좌우 결합 및 쓰기
        combined_frame = np.hstack((vis_img_bgr, overlay_img))
        video_writer.write(combined_frame)

    video_writer.release()
    print("\n" + "=" * 50)
    print(f"추론 완료! 비디오 저장 위치: {out_video_path}")
    if num_processed > 0:
        avg_forward_time = total_forward_time / num_processed
        hardware_fps = 1.0 / avg_forward_time
        print(f"TRT 실제 처리 평균 시간 (Forward pass): {avg_forward_time*1000:.2f} ms/frame")
        print(f"TRT 실제 하드웨어 FPS: {hardware_fps:.2f} FPS")
    print("=" * 50)

if __name__ == "__main__":
    run_inference_trt()
