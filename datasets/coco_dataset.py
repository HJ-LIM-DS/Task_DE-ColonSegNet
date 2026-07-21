"""
coco_dataset.py (v2)

MS COCO 형식 JSON을 읽어 이미지와 마스크를 반환하는 PyTorch Dataset.
작은 병변 메타데이터 및 데이터 증강 파이프라인이 포함되어 있다.

사용 예:
    from datasets.coco_dataset import PolypCOCODataset, get_transforms
    train_dataset = PolypCOCODataset(ann_file="train.json", split="train")
    val_dataset   = PolypCOCODataset(ann_file="val.json",   split="val")
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet 기준 정규화 값
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
INPUT_SIZE    = 352  # DE-ColonSegNet 기본 입력 크기


def get_transforms(split: str) -> A.Compose:
    """
    split에 따라 데이터 증강 파이프라인을 반환한다.

    Train:
        HorizontalFlip, VerticalFlip, RandomRotate90,
        RandomBrightnessContrast, ColorJitter,
        RandomResizedCrop (작은 병변 탐지 강화), GaussianBlur,
        Normalize, ToTensorV2
    Val / Test:
        Resize, Normalize, ToTensorV2 (증강 없이 정확한 평가)
    """
    if split == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4
            ),
            # RandomResizedCrop: 이미지 일부를 확대해 작은 병변을 더 크게 보여줌
            A.RandomResizedCrop(
                size=(INPUT_SIZE, INPUT_SIZE),
                scale=(0.5, 1.0),  # 원본의 50%~100% 크기를 잘라 352x352로 리사이즈
                ratio=(0.75, 1.33),
                p=0.5
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Resize(INPUT_SIZE, INPUT_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ], additional_targets={"mask": "mask"})
    else:
        # Val / Test: 정확한 평가를 위해 Resize만 적용
        return A.Compose([
            A.Resize(INPUT_SIZE, INPUT_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ], additional_targets={"mask": "mask"})


class PolypCOCODataset(Dataset):
    """
    COCO 형식의 annotation JSON 파일을 로드하는 PyTorch Dataset.

    반환값 (각 __getitem__ 호출 시):
        image  (Tensor[3, H, W]): 정규화된 RGB 이미지
        mask   (Tensor[1, H, W]): 이진 병변 마스크 (0 or 1)
        meta   (dict): {image_id, is_small, dataset, file_name}
    """

    def __init__(self, ann_file: str, split: str = "train"):
        """
        Args:
            ann_file (str): COCO JSON 파일 경로 (train.json, val.json, test.json)
            split (str): 'train', 'val', 'test' 중 하나
        """
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = get_transforms(split)
        self.split = split

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        # 이미지 로드
        img_path = img_info["file_name"]
        image = cv2.imread(img_path)
        if image is None:
            # 읽기 실패 시 검은 이미지로 대체 (학습 중 예외 방지)
            image = np.zeros((img_info["height"], img_info["width"], 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 통합 마스크 생성 (여러 annotation을 하나의 이진 마스크로 합침)
        mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann))

        # 변환 적용
        augmented = self.transforms(image=image, mask=mask)
        image = augmented["image"]            # Tensor[3, H, W]
        mask  = augmented["mask"]             # Tensor[H, W]
        mask  = mask.unsqueeze(0).float()    # Tensor[1, H, W]

        # 메타데이터 (학습 중 로깅, 가중치 계산 등에 활용)
        meta = {
            "image_id": img_id,
            "is_small": img_info.get("small_lesion", False),
            "dataset": img_info.get("dataset", "unknown"),
            "file_name": img_path,
        }

        return image, mask, meta


if __name__ == "__main__":
    # 빠른 동작 테스트
    import sys
    ann_path = "E:/Task_DE-ColonSegNet/datasets/coco_format_v2/train.json"
    if not os.path.exists(ann_path):
        print(f"train.json 없음. 먼저 build_dataset.py를 실행하세요.")
        sys.exit(0)

    ds = PolypCOCODataset(ann_path, split="train")
    img, mask, meta = ds[0]
    print(f"Image shape : {img.shape}")
    print(f"Mask shape  : {mask.shape}")
    print(f"Meta        : {meta}")
    print("Dataset 로드 성공!")
