import os
import glob
import time
import cv2
import numpy as np
import torch
from tqdm import tqdm

from models.de_colonsegnet import DE_ColonSegNet
from datasets.coco_dataset import get_transforms

def apply_blue_mask(image, mask, alpha=0.5):
    """
    원본 이미지 위에 파란색 분할 마스크를 반투명하게 덮어씌웁니다.
    """
    # 파란색 마스크 (B, G, R)
    color_mask = np.zeros_like(image)
    color_mask[:, :, 0] = 255  # Blue channel max
    
    # 마스크가 1인 영역만 color_mask 적용
    masked_part = cv2.bitwise_and(color_mask, color_mask, mask=mask)
    
    # 원본 이미지의 마스크 영역
    original_part = cv2.bitwise_and(image, image, mask=mask)
    
    # 반투명 오버레이
    blended = cv2.addWeighted(masked_part, alpha, original_part, 1 - alpha, 0)
    
    # 최종 이미지 결합
    result = image.copy()
    # 마스크 영역만 blended로 교체
    np.copyto(result, blended, where=mask[:, :, None] > 0)
    return result

def run_inference():
    """
    SUN-SEG 데이터셋 전체를 순회하며 1x2 비디오(원본|마스크)를 생성하고 
    실제 처리 FPS를 측정합니다.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장비: {device}")

    # 모델 준비
    model = DE_ColonSegNet(in_channels=3, out_channels=1).to(device)
    checkpoint_path = "checkpoints/best_model.pth"
    if not os.path.exists(checkpoint_path):
        print(f"체크포인트 {checkpoint_path} 가 없습니다.")
        return
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print("모델 가중치 로드 완료.")

    # 변환 (Val/Test용 사용)
    # get_transforms("test") returns A.Compose which expects image=RGB
    transform = get_transforms("test")

    # SUN-SEG 데이터셋 경로 스캔 (모든 Frame 폴더 내의 케이스들)
    dataset_root = "E:/Task_DE-ColonSegNet/datasets/SUN-SEG_dataset"
    print("데이터셋 스캔 중...")
    
    # glob으로 모든 이미지 경로 찾기 (Train, TestEasy, TestHard 내부의 Frame 폴더)
    # 패턴: TrainDataset/Frame/case_X/*.jpg, TestEasyDataset/Seen/Frame/case_X/*.jpg 등
    search_pattern = os.path.join(dataset_root, "**", "Frame", "**", "*.jpg")
    all_image_paths = glob.glob(search_pattern, recursive=True)
    
    # png 포맷이 섞여 있을 수도 있으니 추가
    all_image_paths += glob.glob(os.path.join(dataset_root, "**", "Frame", "**", "*.png"), recursive=True)
    
    # 정렬 (폴더/파일 이름순)
    all_image_paths = sorted(all_image_paths)
    
    if not all_image_paths:
        print("SUN-SEG 데이터셋 이미지를 찾을 수 없습니다.")
        return
    
    print(f"총 {len(all_image_paths)}개의 이미지를 찾았습니다.")

    # 출력 폴더 준비
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_video_path = os.path.join(out_dir, "sun_seg_inference_result.mp4")

    # VideoWriter 설정
    INPUT_SIZE = 352
    frame_width = INPUT_SIZE * 2  # 1x2 포맷이므로 너비는 2배
    frame_height = INPUT_SIZE
    fps_video = 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(out_video_path, fourcc, fps_video, (frame_width, frame_height))

    total_forward_time = 0.0
    num_processed = 0

    print("추론 및 비디오 생성을 시작합니다...")
    with torch.no_grad():
        for img_path in tqdm(all_image_paths, desc="Inference"):
            # 이미지 로드
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            
            # 전처리 (RGB 변환 및 albumentations 적용)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # transform 적용 (resize & normalize)
            # 마스크는 없으므로 image만 전달. ToTensorV2가 적용되어 나옴
            augmented = transform(image=img_rgb)
            img_tensor = augmented["image"].unsqueeze(0).to(device) # (1, 3, 352, 352)

            # 모델 추론 (시간 측정)
            start_time = time.time()
            outputs, _ = model(img_tensor)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            forward_time = time.time() - start_time
            
            total_forward_time += forward_time
            num_processed += 1

            # 마스크 이진화
            pred_mask = torch.sigmoid(outputs).squeeze().cpu().numpy()
            binary_mask = (pred_mask > 0.5).astype(np.uint8)

            # 시각화용 이미지 준비 (크기를 352x352로 맞춘 원본 이미지)
            # transform 에는 resize만 적용되고, 시각화는 원본 색상을 유지하기 위해 다시 resize 적용
            vis_img_bgr = cv2.resize(img_bgr, (INPUT_SIZE, INPUT_SIZE))
            
            # 오버레이 적용
            overlay_img = apply_blue_mask(vis_img_bgr, binary_mask, alpha=0.5)

            # 1x2 결합 (왼쪽: 원본, 오른쪽: 오버레이)
            combined_frame = np.hstack((vis_img_bgr, overlay_img))

            # 비디오에 프레임 쓰기
            video_writer.write(combined_frame)

    video_writer.release()
    print("\n" + "=" * 50)
    print(f"추론 완료! 비디오 저장 위치: {out_video_path}")
    if num_processed > 0:
        avg_forward_time = total_forward_time / num_processed
        hardware_fps = 1.0 / avg_forward_time
        print(f"모델 실제 처리 평균 시간 (Forward pass): {avg_forward_time*1000:.2f} ms/frame")
        print(f"모델 실제 하드웨어 FPS: {hardware_fps:.2f} FPS")
    print("=" * 50)

if __name__ == "__main__":
    run_inference()
