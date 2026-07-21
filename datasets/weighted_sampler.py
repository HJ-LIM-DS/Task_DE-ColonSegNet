"""
weighted_sampler.py

작은 병변(small lesion) 이미지에 더 높은 샘플링 가중치를 부여하여
DataLoader에서 자동으로 오버샘플링하는 WeightedRandomSampler 생성 유틸리티.

사용 예:
    from datasets.weighted_sampler import make_weighted_sampler
    sampler = make_weighted_sampler(dataset, small_weight=3.0)
    loader = DataLoader(dataset, sampler=sampler, batch_size=8)
"""

import torch
from torch.utils.data import WeightedRandomSampler


def make_weighted_sampler(dataset, small_weight: float = 3.0, normal_weight: float = 1.0):
    """
    작은 병변 이미지에 가중치를 부여한 WeightedRandomSampler를 생성한다.

    Args:
        dataset: PolypCOCODataset 인스턴스 (is_small 플래그를 반환해야 함)
        small_weight (float): 작은 병변 이미지의 샘플링 가중치 (기본 3.0배)
        normal_weight (float): 일반 이미지의 샘플링 가중치 (기본 1.0)

    Returns:
        WeightedRandomSampler: 가중치가 설정된 PyTorch 샘플러
    """
    weights = []

    for i in range(len(dataset)):
        img_info = dataset.coco.loadImgs(dataset.ids[i])[0]
        is_small = img_info.get("small_lesion", False)
        weights.append(small_weight if is_small else normal_weight)

    weights_tensor = torch.tensor(weights, dtype=torch.float)
    # replacement=True: 동일 이미지가 한 에포크에 여러 번 나올 수 있음 (오버샘플링 핵심)
    sampler = WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=len(weights_tensor),
        replacement=True
    )
    small_count = sum(1 for w in weights if w == small_weight)
    print(f"  WeightedSampler: total={len(weights)}, "
          f"small_lesion={small_count} (weight={small_weight}x), "
          f"normal={len(weights)-small_count} (weight={normal_weight}x)")
    return sampler
