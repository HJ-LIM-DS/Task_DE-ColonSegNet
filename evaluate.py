import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.coco_dataset import PolypCOCODataset
from models.de_colonsegnet import DE_ColonSegNet
from train import calculate_metrics

def evaluate_model():
    """
    학습된 모델의 가중치를 불러와 테스트 데이터셋(test.json)에서
    전체, 일반 병변, 작은 병변에 대한 성능(Dice, Recall, IoU)을 각각 평가합니다.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장비: {device}")

    # 데이터셋 로드
    print("테스트 데이터셋 로드 중...")
    test_dataset = PolypCOCODataset(
        "E:/Task_DE-ColonSegNet/datasets/coco_format_v2/test.json", split="test"
    )
    # 평가 시에는 개별 이미지별 정확한 지표 할당을 위해 batch_size=1을 권장합니다.
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # 모델 정의 및 가중치 로드
    model = DE_ColonSegNet(in_channels=3, out_channels=1).to(device)
    checkpoint_path = "checkpoints/best_model.pth"
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"체크포인트 로드 완료: {checkpoint_path}")
    except Exception as e:
        print(f"체크포인트를 로드하는 데 실패했습니다. 파일이 존재하는지 확인하세요: {e}")
        return

    model.eval()

    # 지표 집계용 변수 (전체, 일반 병변, 작은 병변)
    metrics_all = {"dice": 0.0, "recall": 0.0, "iou": 0.0, "count": 0}
    metrics_general = {"dice": 0.0, "recall": 0.0, "iou": 0.0, "count": 0}
    metrics_small = {"dice": 0.0, "recall": 0.0, "iou": 0.0, "count": 0}

    print("-" * 50)
    print("평가를 시작합니다...")
    
    with torch.no_grad():
        for images, masks, meta in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)
            # batch_size=1 이므로 is_small은 tensor 내에 하나의 값으로 존재함
            is_small = meta["is_small"].item()

            outputs, _ = model(images)
            
            # calculate_metrics는 3개의 값을 반환함 (dice, recall, iou)
            d, r, iou = calculate_metrics(outputs, masks)

            # 1. 전체 지표에 누적
            metrics_all["dice"] += d
            metrics_all["recall"] += r
            metrics_all["iou"] += iou
            metrics_all["count"] += 1

            # 2. 병변 크기 그룹에 따른 분리 누적
            if is_small:
                metrics_small["dice"] += d
                metrics_small["recall"] += r
                metrics_small["iou"] += iou
                metrics_small["count"] += 1
            else:
                metrics_general["dice"] += d
                metrics_general["recall"] += r
                metrics_general["iou"] += iou
                metrics_general["count"] += 1

    # 평균 계산 함수
    def calc_mean(metric_dict):
        if metric_dict["count"] == 0:
            return 0.0, 0.0, 0.0
        c = metric_dict["count"]
        return metric_dict["dice"]/c, metric_dict["recall"]/c, metric_dict["iou"]/c

    avg_all_d, avg_all_r, avg_all_iou = calc_mean(metrics_all)
    avg_gen_d, avg_gen_r, avg_gen_iou = calc_mean(metrics_general)
    avg_sml_d, avg_sml_r, avg_sml_iou = calc_mean(metrics_small)

    print("\n" + "=" * 50)
    print("[ 최종 평가 결과 (Final Evaluation Results) ]")
    print(f"테스트 데이터 개수: 총 {metrics_all['count']}장")
    print(f"  - 일반 병변(General): {metrics_general['count']}장")
    print(f"  - 작은 병변(Small):   {metrics_small['count']}장")
    print("-" * 50)
    print(f"[ 전체 평균 (All) ]")
    print(f"  Dice Score : {avg_all_d:.4f}")
    print(f"  Recall     : {avg_all_r:.4f}")
    print(f"  IoU        : {avg_all_iou:.4f}")
    print("-" * 50)
    print(f"[ 일반 병변 평균 (General Lesion) ]")
    print(f"  Dice Score : {avg_gen_d:.4f}")
    print(f"  Recall     : {avg_gen_r:.4f}")
    print(f"  IoU        : {avg_gen_iou:.4f}")
    print("-" * 50)
    print(f"[ 작은 병변 평균 (Small Lesion) ]")
    print(f"  Dice Score : {avg_sml_d:.4f}")
    print(f"  Recall     : {avg_sml_r:.4f}")
    print(f"  IoU        : {avg_sml_iou:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    evaluate_model()
