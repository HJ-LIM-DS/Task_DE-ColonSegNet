import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.coco_dataset import PolypCOCODataset
from datasets.weighted_sampler import make_weighted_sampler
from models.de_colonsegnet import DE_ColonSegNet
from models.losses import CombinedLoss


def calculate_metrics(pred_logits, true_masks, threshold=0.5):
    """
    예측 logits와 실제 마스크(0 or 1)를 기반으로 Dice Score, Recall, IoU를 계산합니다.

    Args:
        pred_logits (Tensor): 모델의 출력 logits (B, 1, H, W)
        true_masks  (Tensor): 실제 이진 마스크 (B, 1, H, W)
        threshold   (float) : 이진화 기준 임계값

    Returns:
        dice   (float): Dice 계수
        recall (float): 재현율
        iou    (float): IoU 계수
    """
    pred_masks = (torch.sigmoid(pred_logits) > threshold).float()

    # 픽셀 단위 연산을 위해 (B, H*W)로 평탄화
    pred_masks = pred_masks.view(pred_masks.size(0), -1)
    true_masks = true_masks.view(true_masks.size(0), -1)

    tp = (pred_masks * true_masks).sum(dim=1)
    fp = (pred_masks * (1.0 - true_masks)).sum(dim=1)
    fn = ((1.0 - pred_masks) * true_masks).sum(dim=1)

    # 분모가 0이 되는 것을 방지하기 위해 epsilon(1e-5) 추가
    dice   = (2.0 * tp + 1e-5) / (2.0 * tp + fp + fn + 1e-5)
    recall = (tp + 1e-5) / (tp + fn + 1e-5)
    iou    = (tp + 1e-5) / (tp + fp + fn + 1e-5)

    return dice.mean().item(), recall.mean().item(), iou.mean().item()


def train_model(batch_size=8, lr=1e-4, patience=3, test_mode=False,
                resume=False, resume_best_dice=0.0, resume_patience_counter=0,
                start_epoch=1):
    """
    DE-ColonSegNet 전체 파인튜닝 학습 루프 함수입니다.
    에포크 수는 무한으로 설정하며, Early Stopping(patience=3)으로 종료합니다.
    Validation Dice Score가 갱신될 때마다 반드시 best_model.pth를 즉시 저장합니다.

    Args:
        batch_size             (int)  : 배치 크기
        lr                     (float): 초기 학습률
        patience               (int)  : Early Stopping 인내 에포크 수
        test_mode              (bool) : True인 경우 1 에포크, 배치 2 테스트 모드로 실행
        resume                 (bool) : checkpoints/best_model.pth에서 가중치를 불러와 이어서 학습
        resume_best_dice       (float): 재개 시 복원할 이전 최고 Val Dice 값
        resume_patience_counter(int)  : 재개 시 복원할 이전 Early Stopping 카운터
        start_epoch            (int)  : 재개 시작 에포크 번호 (로그 출력용)
    """
    if test_mode:
        batch_size = 2
        patience   = 1
        print("테스트 모드로 구동합니다. (Epoch: 무한/Early Stop 1, Batch: 2)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장비: {device}")

    # 1. COCO v2 데이터셋 로드
    print("데이터셋 로드 중...")
    train_dataset = PolypCOCODataset(
        "E:/Task_DE-ColonSegNet/datasets/coco_format_v2/train.json", split="train"
    )
    val_dataset = PolypCOCODataset(
        "E:/Task_DE-ColonSegNet/datasets/coco_format_v2/val.json", split="val"
    )

    # 2. Weighted Random Sampler 구축 (작은 병변 3.0배 오버샘플링)
    print("Weighted Sampler 생성 중...")
    train_sampler = make_weighted_sampler(train_dataset, small_weight=3.0)

    # 3. 데이터 로더 정의 (sampler 지정 시 shuffle=False 필수)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # 4. 모델, 옵티마이저, 스케줄러, 손실 함수 정의
    model     = DE_ColonSegNet(in_channels=3, out_channels=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )
    criterion = CombinedLoss()

    # 5. 이전 체크포인트에서 가중치 불러오기 (resume 모드)
    checkpoint_path = "checkpoints/best_model.pth"
    os.makedirs("checkpoints", exist_ok=True)

    if resume and os.path.exists(checkpoint_path):
        print(f"체크포인트 불러오는 중: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"체크포인트 로드 완료! 이전 Best Val Dice: {resume_best_dice:.4f}, "
              f"Early Stopping 카운터: {resume_patience_counter}/{patience}")

    best_val_dice    = resume_best_dice
    patience_counter = resume_patience_counter
    epoch            = start_epoch - 1  # 루프 진입 시 +1 되므로 1 빼둠

    print(f"학습을 시작합니다. Early Stopping patience={patience}")
    print("-" * 70)

    # 6. 무한 에포크 학습 루프 (Early Stopping으로만 종료)
    while True:
        epoch += 1
        epoch_start = time.time()

        # --- 훈련 ---
        model.train()
        train_loss = 0.0
        train_dice = train_recall = train_iou = 0.0
        num_train_batches = 0

        for i, (images, masks, meta) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            is_small = meta["is_small"]

            optimizer.zero_grad()

            outputs, aux_outputs = model(images)

            loss_main = criterion(outputs, masks, is_small)
            loss_aux  = criterion(aux_outputs, masks, is_small)
            loss      = loss_main + 0.4 * loss_aux

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            d, r, iou   = calculate_metrics(outputs, masks)
            train_dice   += d
            train_recall += r
            train_iou    += iou
            num_train_batches += 1

            if i % 10 == 0:
                print(
                    f"Epoch {epoch} - Batch {i}/{len(train_loader)} "
                    f"- Loss: {loss.item():.4f}",
                    flush=True
                )

            # 테스트 모드: 3 배치만 실행 후 종료
            if test_mode and i >= 2:
                break

        avg_train_loss   = train_loss   / num_train_batches
        avg_train_dice   = train_dice   / num_train_batches
        avg_train_recall = train_recall / num_train_batches
        avg_train_iou    = train_iou    / num_train_batches

        # --- 검증 ---
        model.eval()
        val_loss = 0.0
        val_dice = val_recall = val_iou = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for i, (images, masks, meta) in enumerate(val_loader):
                images, masks = images.to(device), masks.to(device)
                is_small = meta["is_small"]

                outputs, aux_outputs = model(images)

                loss_main = criterion(outputs, masks, is_small)
                loss_aux  = criterion(aux_outputs, masks, is_small)
                loss      = loss_main + 0.4 * loss_aux
                val_loss += loss.item()

                d, r, iou = calculate_metrics(outputs, masks)
                val_dice   += d
                val_recall += r
                val_iou    += iou
                num_val_batches += 1

                if test_mode and i >= 2:
                    break

        avg_val_loss   = val_loss   / num_val_batches
        avg_val_dice   = val_dice   / num_val_batches
        avg_val_recall = val_recall / num_val_batches
        avg_val_iou    = val_iou    / num_val_batches

        epoch_elapsed = time.time() - epoch_start
        current_lr    = optimizer.param_groups[0]["lr"]

        print("-" * 70)
        print(
            f"Epoch {epoch} 완료 ({epoch_elapsed:.1f}s) | LR: {current_lr:.2e}"
        )
        print(
            f"  Train | Loss: {avg_train_loss:.4f}, Dice: {avg_train_dice:.4f}, "
            f"Recall: {avg_train_recall:.4f}, IoU: {avg_train_iou:.4f}"
        )
        print(
            f"  Val   | Loss: {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}, "
            f"Recall: {avg_val_recall:.4f}, IoU: {avg_val_iou:.4f}"
        )

        # 7. ReduceLROnPlateau 스케줄러 갱신 (val Dice 기준)
        scheduler.step(avg_val_dice)

        # 8. 베스트 모델 즉시 저장 (Val Dice 갱신 시마다 반드시 저장)
        if avg_val_dice > best_val_dice:
            best_val_dice    = avg_val_dice
            patience_counter = 0

            # 가중치 파일 안전 저장 (tmp 파일로 먼저 저장 후 원자적 교체)
            tmp_path = checkpoint_path + ".tmp"
            torch.save(model.state_dict(), tmp_path)
            os.replace(tmp_path, checkpoint_path)

            print(
                f"  => Best model saved! "
                f"Val Dice: {best_val_dice:.4f}, Recall: {avg_val_recall:.4f}"
            )
        else:
            patience_counter += 1
            print(
                f"  Early Stopping 대기: {patience_counter}/{patience} "
                f"(Best Val Dice: {best_val_dice:.4f})"
            )

        print("-" * 70, flush=True)

        # 9. Early Stopping 종료 조건 체크
        if patience_counter >= patience:
            print(
                f"Early Stopping 조건 충족 ({patience} 에포크 연속 미갱신). "
                f"학습을 종료합니다."
            )
            print(f"최종 Best Val Dice: {best_val_dice:.4f}")
            break

        # 테스트 모드: 1 에포크 후 종료
        if test_mode:
            print("테스트 실행 검증 완료!")
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DE-ColonSegNet Full Training Script")
    parser.add_argument("--test_run",              action="store_true", help="테스트 실행 여부")
    parser.add_argument("--batch_size",            type=int,   default=8,     help="배치 크기")
    parser.add_argument("--lr",                    type=float, default=1e-4,  help="초기 학습률")
    parser.add_argument("--patience",              type=int,   default=3,     help="Early Stopping patience")
    parser.add_argument("--resume",                action="store_true",       help="체크포인트에서 이어서 학습")
    parser.add_argument("--resume_best_dice",      type=float, default=0.0,   help="재개 시 이전 최고 Val Dice")
    parser.add_argument("--resume_patience_counter", type=int, default=0,     help="재개 시 Early Stopping 카운터")
    parser.add_argument("--start_epoch",           type=int,   default=1,     help="재개 시작 에포크 번호")
    args = parser.parse_args()

    train_model(
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        test_mode=args.test_run,
        resume=args.resume,
        resume_best_dice=args.resume_best_dice,
        resume_patience_counter=args.resume_patience_counter,
        start_epoch=args.start_epoch,
    )
