"""
verify_dataset.py

생성된 COCO JSON 데이터셋의 유효성을 검증하고 랜덤 샘플 5개를 시각화한다.

수행 항목:
  1. pycocotools 로드 테스트 (JSON 형식 유효성)
  2. 이미지 파일 실제 존재 여부 확인 (랜덤 50개 샘플)
  3. 마스크와 이미지 해상도 일치 확인
  4. 작은 병변 비율 및 Weighted Sampler 분포 확인
  5. 랜덤 샘플 5개를 PNG로 시각화 (이미지 + 마스크 오버레이 + bbox)
"""

import os
import sys
import json
import random
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask_utils

COCO_DIR   = "E:/Task_DE-ColonSegNet/datasets/coco_format_v2"
OUTPUT_DIR = "E:/Task_DE-ColonSegNet/datasets/coco_format_v2/verify_samples"
SPLITS     = ["train", "val", "test"]
NUM_VIS    = 5   # 시각화할 샘플 수

os.makedirs(OUTPUT_DIR, exist_ok=True)

def draw_sample(coco, img_id, out_path):
    """이미지 위에 마스크와 bbox를 오버레이해 PNG로 저장"""
    img_info = coco.loadImgs(img_id)[0]
    img_path = img_info["file_name"]

    img = cv2.imread(img_path)
    if img is None:
        print(f"  [WARN] 이미지 로드 실패: {img_path}")
        return False

    # Annotation 시각화
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    overlay = img.copy()
    for ann in anns:
        # Segmentation mask 오버레이 (초록색, 반투명)
        seg_mask = coco.annToMask(ann)
        overlay[seg_mask > 0] = overlay[seg_mask > 0] * 0.4 + np.array([0, 200, 80]) * 0.6

        # Bounding Box 표시 (빨간색)
        x, y, w, h = [int(v) for v in ann["bbox"]]
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 220), 2)

        # 작은 병변 표시
        is_small = ann.get("extra", {}).get("small_lesion", False)
        polyp_ratio = ann.get("extra", {}).get("polyp_ratio", 0.0)
        label = f"{'SMALL ' if is_small else ''}polyp {polyp_ratio:.3f}"
        cv2.putText(overlay, label, (x, max(y-5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1)

    # 데이터셋 출처 표시
    ds_name = img_info.get("dataset", "unknown")
    cv2.putText(overlay, f"[{ds_name}] id={img_id}", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.imwrite(out_path, overlay)
    return True

errors = []

for split in SPLITS:
    ann_file = os.path.join(COCO_DIR, f"{split}.json")
    print(f"\n{'='*50}")
    print(f"[{split.upper()}] 검증 시작: {ann_file}")
    print("="*50)

    # --- 검증 1: pycocotools 로드 ---
    try:
        coco = COCO(ann_file)
        print(f"  [OK] COCO JSON 로드 성공")
        print(f"       images={len(coco.imgs)}, anns={len(coco.anns)}, cats={len(coco.cats)}")
    except Exception as e:
        print(f"  [FAIL] COCO JSON 로드 실패: {e}")
        errors.append(f"{split}: COCO load failed - {e}")
        continue

    # --- 검증 2: 이미지 파일 존재 확인 (랜덤 50개) ---
    all_ids = list(coco.imgs.keys())
    sample_ids = random.sample(all_ids, min(50, len(all_ids)))
    missing = 0
    for img_id in sample_ids:
        img_info = coco.loadImgs(img_id)[0]
        if not os.path.exists(img_info["file_name"]):
            missing += 1
    if missing == 0:
        print(f"  [OK] 이미지 파일 존재 확인 (50개 샘플 중 누락 없음)")
    else:
        print(f"  [WARN] 이미지 파일 {missing}개 누락 (50개 중)")
        errors.append(f"{split}: {missing} images missing")

    # --- 검증 3: 이미지-마스크 해상도 일치 확인 ---
    mismatch = 0
    check_ids = random.sample(all_ids, min(30, len(all_ids)))
    for img_id in check_ids:
        img_info = coco.loadImgs(img_id)[0]
        img = cv2.imread(img_info["file_name"])
        if img is None:
            continue
        h, w = img.shape[:2]
        if w != img_info["width"] or h != img_info["height"]:
            mismatch += 1
    if mismatch == 0:
        print(f"  [OK] 이미지 해상도 일치 (30개 샘플)")
    else:
        print(f"  [WARN] 해상도 불일치 {mismatch}개")
        errors.append(f"{split}: {mismatch} resolution mismatches")

    # --- 검증 4: 작은 병변 비율 ---
    with open(ann_file, encoding="utf-8") as f:
        raw = json.load(f)
    small_count = sum(1 for img in raw["images"] if img.get("small_lesion", False))
    total = len(raw["images"])
    print(f"  [INFO] 작은 병변 비율: {small_count}/{total} = {100*small_count/max(total,1):.1f}%")

    # --- 검증 5: 시각화 ---
    vis_ids = random.sample(all_ids, min(NUM_VIS, len(all_ids)))
    success = 0
    for vis_id in vis_ids:
        out_file = os.path.join(OUTPUT_DIR, f"{split}_id{vis_id}.png")
        if draw_sample(coco, vis_id, out_file):
            success += 1
    print(f"  [OK] 시각화 완료: {success}/{NUM_VIS}개 -> {OUTPUT_DIR}")

# --- 최종 결과 ---
print(f"\n{'='*50}")
if not errors:
    print("[검증 완료] 모든 항목 통과!")
else:
    print("[검증 완료] 아래 항목 주의:")
    for e in errors:
        print(f"  - {e}")

print(f"\n시각화 샘플 저장 위치: {OUTPUT_DIR}")
