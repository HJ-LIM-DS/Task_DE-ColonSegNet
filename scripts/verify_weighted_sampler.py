"""
verify_weighted_sampler.py

WeightedSampler가 작은 병변 이미지를 올바르게 더 자주 샘플링하는지 검증한다.
"""

import os
import sys

sys.path.insert(0, "E:/Task_DE-ColonSegNet")

from datasets.coco_dataset import PolypCOCODataset
from datasets.weighted_sampler import make_weighted_sampler
from torch.utils.data import DataLoader

ANN_FILE = "E:/Task_DE-ColonSegNet/datasets/coco_format_v2/train.json"

print("Dataset 로드 중...")
dataset = PolypCOCODataset(ANN_FILE, split="train")
print(f"  총 이미지: {len(dataset)}")

print("\nWeighted Sampler 생성 중...")
sampler = make_weighted_sampler(dataset, small_weight=3.0)

print("\nDataLoader 테스트 (10 배치)...")
loader = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=0)

small_count = 0
total_count = 0
for i, (images, masks, metas) in enumerate(loader):
    if i >= 10:
        break
    batch_small = sum(1 for s in metas["is_small"] if s)
    small_count += batch_small
    total_count += len(metas["is_small"])

print(f"  10 배치 샘플링 결과:")
print(f"  총 샘플: {total_count}, 작은 병변: {small_count} ({100*small_count/max(total_count,1):.1f}%)")
print(f"  (기대치: 작은 병변 비율 38% x 3배 = 약 60~70%)")
print("\nWeighted Sampler 검증 완료!")
