import torch
import torch.nn as nn
import torch.nn.functional as F

class TverskyLoss(nn.Module):
    """
    Tversky Loss는 Precision과 Recall의 균형을 조절할 수 있는 손실 함수입니다.
    특히 대장 내시경 폴립 검출과 같이 미검출(FN, False Negative)을 최소화해야 하는
    상황에서 Recall에 더 높은 가중치를 주어 학습을 유도합니다.
    """
    def __init__(self, smooth=1e-5):
        super(TverskyLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true, is_small):
        """
        Args:
            y_pred (Tensor): 모델의 예측 logits 값 (B, 1, H, W)
            y_true (Tensor): 실제 마스크 값 (B, 1, H, W)
            is_small (list or Tensor): 배치 내 각 샘플이 작은 병변인지 여부 (B,)
        """
        # Logits 값을 0과 1 사이의 확률값으로 변환
        y_pred = torch.sigmoid(y_pred)

        # 텐서를 1차원으로 평탄화하여 계산 준비
        y_pred = y_pred.view(y_pred.size(0), -1)
        y_true = y_true.view(y_true.size(0), -1)

        loss = 0.0
        batch_size = y_pred.size(0)

        # 배치 내 각 샘플에 대해 개별 손실을 계산
        for i in range(batch_size):
            pred_i = y_pred[i]
            true_i = y_true[i]
            is_small_i = is_small[i]

            # 작은 병변(is_small_i가 True)인 경우 미검출 페널티(beta)를 더 높임
            if is_small_i:
                alpha = 0.15
                beta = 0.85
            else:
                alpha = 0.3
                beta = 0.7

            # True Positive, False Positive, False Negative 계산
            tp = (pred_i * true_i).sum()
            fp = (pred_i * (1.0 - true_i)).sum()
            fn = ((1.0 - pred_i) * true_i).sum()

            # Tversky Index 산출
            tversky_index = (tp + self.smooth) / (tp + alpha * fp + beta * fn + self.smooth)
            loss += (1.0 - tversky_index)

        # 배치 평균 손실 반환
        return loss / batch_size

class FocalLoss(nn.Module):
    """
    Focal Loss는 학습하기 어려운(hard) 샘플에 가중치를 더 부여하고,
    쉬운(easy) 샘플의 영향력을 줄여서 클래스 불균형을 해소하는 손실 함수입니다.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): 모델의 예측 logits 값 (B, 1, H, W)
            targets (Tensor): 실제 마스크 값 (B, 1, H, W)
        """
        # 수치적 안정성을 위해 F.binary_cross_entropy_with_logits 사용
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # pt는 모델이 올바르게 예측할 확률

        # Focal Loss 수식 적용
        focal_loss = self.alpha * ((1.0 - pt) ** self.gamma) * bce_loss
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    """
    Tversky Loss와 Focal Loss를 결합하여 미검출을 적극 방어하면서
    동시에 학습이 어려운 영역에 대한 세그멘테이션 성능을 강화합니다.
    """
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.tversky = TverskyLoss()
        self.focal = FocalLoss()

    def forward(self, inputs, targets, is_small):
        """
        Args:
            inputs (Tensor): 모델의 예측 logits 값 (B, 1, H, W)
            targets (Tensor): 실제 마스크 값 (B, 1, H, W)
            is_small (list or Tensor): 배치 내 각 샘플이 작은 병변인지 여부 (B,)
        """
        tversky_loss = self.tversky(inputs, targets, is_small)
        focal_loss = self.focal(inputs, targets)
        return tversky_loss + focal_loss
