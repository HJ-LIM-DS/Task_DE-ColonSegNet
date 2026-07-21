"""
build_dataset.py

6개 데이터셋을 MS COCO 형식으로 통합하고, Train/Val/Test로 분할하는 메인 파이프라인 스크립트.

포함 데이터셋:
  1. kvasir-seg         (1,000장, PNG mask)
  2. CVC-ClinicDB       (612장, PNG mask)
  3. ETIS-LaribPolypDB  (196장, PNG mask)
  4. BKAI-IGH NeoPolyp  (1,000장, PNG mask)
  5. PolypGen            (~1,800장, 6개 센터, PNG mask)
  6. SUN-SEG            (19,544 프레임에서 10장마다 1장 샘플링 -> ~1,954장, GT PNG mask)

작은 병변 기준: polyp 픽셀 비율 < 2% (전체 이미지 픽셀 대비)
분할 비율: Train 80% / Val 10% / Test 10%
"""

import os
import json
import cv2
import numpy as np
import hashlib
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==================== 설정값 ====================
DATASETS_BASE = "E:/Task_DE-ColonSegNet/datasets"
OUTPUT_DIR    = "E:/Task_DE-ColonSegNet/datasets/coco_format_v2"
SMALL_LESION_THRESHOLD = 0.02  # polyp 픽셀 비율이 2% 미만이면 작은 병변
SUNSEG_SAMPLE_INTERVAL = 10    # SUN-SEG: 10 프레임마다 1장 샘플링
VAL_RATIO  = 0.1
TEST_RATIO = 0.1
# ===============================================

def compute_md5(filepath):
    """이미지 파일의 MD5 해시를 계산하여 중복 제거에 활용"""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def create_coco_format():
    """빈 COCO 형식 딕셔너리 생성"""
    return {
        "info": {"description": "Polyp Detection Dataset (COCO format)", "version": "2.0"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "polyp", "supercategory": "lesion"}]
    }

def mask_to_annotations(mask_path, img_id, ann_id_start, img_w, img_h):
    """
    이진 마스크 이미지로부터 COCO annotation 목록을 생성한다.

    반환:
        anns (list): annotation 딕셔너리 목록
        next_ann_id (int): 다음 annotation에 사용할 ID
        is_small (bool): 작은 병변 여부 (polyp 픽셀 < 2%)
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return [], ann_id_start, False

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    total_pixels = img_w * img_h
    polyp_pixels = int(np.sum(binary > 0))
    polyp_ratio = polyp_pixels / total_pixels
    is_small = (polyp_ratio > 0) and (polyp_ratio < SMALL_LESION_THRESHOLD)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    anns = []
    ann_id = ann_id_start
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 10 or len(contour) < 3:
            # 너무 작은 artifact는 무시
            continue
        seg = contour.flatten().tolist()
        x, y, w, h = cv2.boundingRect(contour)
        anns.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "segmentation": [seg],
            "bbox": [x, y, w, h],
            "area": float(area),
            "iscrowd": 0,
            "extra": {"small_lesion": is_small, "polyp_ratio": round(polyp_ratio, 5)}
        })
        ann_id += 1

    return anns, ann_id, is_small

def collect_dataset_pairs(name, img_dir, mask_dir, img_exts, mask_ext, sample_interval=1, mask_suffix=""):
    """
    이미지-마스크 파일 쌍을 수집한다.

    Args:
        name (str): 데이터셋 이름
        img_dir (str): 이미지 폴더 절대 경로
        mask_dir (str): 마스크 폴더 절대 경로
        img_exts (list): 허용할 이미지 확장자 목록 (예: ['.jpg', '.jpeg', '.png'])
        mask_ext (str): 마스크 파일 확장자 (예: '.png')
        sample_interval (int): N 프레임마다 1장 샘플링 (SUN-SEG 등 연속 프레임용)
        mask_suffix (str): 마스크 파일명 접미사 (예: '_mask')

    반환:
        pairs (list of tuple): (img_path, mask_path, dataset_name) 쌍 목록
    """
    pairs = []

    if not os.path.isdir(img_dir):
        print(f"  [WARNING] img_dir not found: {img_dir}")
        return pairs

    # 하위 폴더가 있는 경우(SUN-SEG 등)와 없는 경우(kvasir 등) 모두 처리
    all_imgs = []
    for ext in img_exts:
        all_imgs += glob(os.path.join(img_dir, "**", f"*{ext}"), recursive=True)
        all_imgs += glob(os.path.join(img_dir, f"*{ext}"))
    # 중복 제거 및 정렬
    all_imgs = sorted(set(all_imgs))

    for i, img_path in enumerate(all_imgs):
        # 샘플 간격 적용 (연속 프레임 데이터)
        if i % sample_interval != 0:
            continue
        basename = os.path.splitext(os.path.basename(img_path))[0]
        # 마스크 경로 추측 (하위 폴더 구조 동일하게 유지)
        rel = os.path.relpath(img_path, img_dir)
        rel_no_ext = os.path.splitext(rel)[0]
        mask_path = os.path.join(mask_dir, rel_no_ext + mask_suffix + mask_ext)

        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path, name))

    print(f"  [{name}] Collected {len(pairs)} image-mask pairs (interval={sample_interval})")
    return pairs

def collect_all_pairs():
    """6개 데이터셋에서 이미지-마스크 쌍을 모두 수집"""
    all_pairs = []
    base = DATASETS_BASE

    # 1. kvasir-seg
    all_pairs += collect_dataset_pairs(
        "kvasir-seg",
        img_dir=f"{base}/kvasir-seg/images",
        mask_dir=f"{base}/kvasir-seg/masks",
        img_exts=[".jpg", ".jpeg"],
        mask_ext=".jpg"
    )

    # 2. CVC-ClinicDB
    all_pairs += collect_dataset_pairs(
        "CVC-ClinicDB",
        img_dir=f"{base}/CVC-ClinicDB/PNG/Original",
        mask_dir=f"{base}/CVC-ClinicDB/PNG/Ground Truth",
        img_exts=[".png"],
        mask_ext=".png"
    )

    # 3. ETIS-LaribPolypDB
    all_pairs += collect_dataset_pairs(
        "ETIS-LaribPolypDB",
        img_dir=f"{base}/ETIS-LaribPolypDB/images",
        mask_dir=f"{base}/ETIS-LaribPolypDB/masks",
        img_exts=[".png"],
        mask_ext=".png"
    )

    # 4. BKAI-IGH NeoPolyp (train 세트만 사용)
    all_pairs += collect_dataset_pairs(
        "BKAI-IGH",
        img_dir=f"{base}/BKAI-IGH NeoPolyp/train/train",
        mask_dir=f"{base}/BKAI-IGH NeoPolyp/train_gt/train_gt",
        img_exts=[".jpeg", ".jpg", ".png"],
        mask_ext=".jpeg"
    )

    # 5. PolypGen (C1~C6 센터 모두)
    for c_idx in range(1, 7):
        all_pairs += collect_dataset_pairs(
            f"PolypGen-C{c_idx}",
            img_dir=f"{base}/PolypGen/PolypGen2021_MultiCenterData_v3/data_C{c_idx}/images_C{c_idx}",
            mask_dir=f"{base}/PolypGen/PolypGen2021_MultiCenterData_v3/data_C{c_idx}/masks_C{c_idx}",
            img_exts=[".jpg", ".jpeg", ".png"],
            mask_ext=".jpg",
            mask_suffix="_mask"
        )

    # 6. SUN-SEG (10 프레임마다 1장 샘플링)
    all_pairs += collect_dataset_pairs(
        "SUN-SEG",
        img_dir=f"{base}/SUN-SEG_dataset/TrainDataset/Frame",
        mask_dir=f"{base}/SUN-SEG_dataset/TrainDataset/GT",
        img_exts=[".jpg", ".jpeg"],
        mask_ext=".png",
        sample_interval=SUNSEG_SAMPLE_INTERVAL
    )

    return all_pairs

def build_coco_from_pairs(pairs, start_img_id=1, start_ann_id=1):
    """
    이미지-마스크 쌍 목록을 받아 COCO 딕셔너리를 생성한다.
    """
    coco = create_coco_format()
    img_id = start_img_id
    ann_id = start_ann_id
    skipped = 0
    seen_hashes = set()

    for img_path, mask_path, ds_name in tqdm(pairs, desc="  Building COCO"):
        # 이미지 열기
        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue

        # MD5 해시로 중복 이미지 제거
        img_hash = compute_md5(img_path)
        if img_hash in seen_hashes:
            skipped += 1
            continue
        seen_hashes.add(img_hash)

        h, w = img.shape[:2]

        # Annotation 생성
        anns, ann_id, is_small = mask_to_annotations(mask_path, img_id, ann_id, w, h)

        # 마스크가 비어 있거나 annotation이 없으면 제외
        if len(anns) == 0:
            skipped += 1
            continue

        coco["images"].append({
            "id": img_id,
            "file_name": os.path.abspath(img_path).replace("\\", "/"),
            "width": w,
            "height": h,
            "dataset": ds_name,
            "small_lesion": is_small
        })
        coco["annotations"].extend(anns)
        img_id += 1

    print(f"  Skipped (duplicate/empty): {skipped}")
    return coco, img_id, ann_id

def split_and_save(all_pairs, output_dir):
    """
    전체 쌍을 Train/Val/Test로 분할하고 각각 COCO JSON으로 저장한다.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1차 분할: train+val / test
    train_val_pairs, test_pairs = train_test_split(
        all_pairs, test_size=TEST_RATIO, random_state=42
    )
    # 2차 분할: train / val
    val_ratio_adjusted = VAL_RATIO / (1 - TEST_RATIO)
    train_pairs, val_pairs = train_test_split(
        train_val_pairs, test_size=val_ratio_adjusted, random_state=42
    )

    print(f"\n총 이미지: {len(all_pairs)}")
    print(f"  Train: {len(train_pairs)}")
    print(f"  Val:   {len(val_pairs)}")
    print(f"  Test:  {len(test_pairs)}")

    ann_id = 1
    img_id = 1

    for split_name, pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        print(f"\n[{split_name.upper()}] 변환 중...")
        coco, img_id, ann_id = build_coco_from_pairs(pairs, img_id, ann_id)
        out_path = os.path.join(output_dir, f"{split_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False)

        # 통계 출력
        small_count = sum(1 for img in coco["images"] if img.get("small_lesion", False))
        print(f"  -> {out_path}")
        print(f"     Images: {len(coco['images'])}, Annotations: {len(coco['annotations'])}")
        print(f"     Small lesion images: {small_count} ({100*small_count/max(len(coco['images']),1):.1f}%)")

        # 데이터셋별 통계
        ds_count = {}
        for img in coco["images"]:
            ds = img.get("dataset", "unknown")
            ds_count[ds] = ds_count.get(ds, 0) + 1
        for ds, cnt in sorted(ds_count.items()):
            print(f"     - {ds}: {cnt}")

def main():
    print("=" * 60)
    print("DE-ColonSegNet Data Pipeline Builder v2.0")
    print("=" * 60)
    print(f"\n데이터셋 수집 중...")

    all_pairs = collect_all_pairs()
    print(f"\n총 수집된 쌍: {len(all_pairs)}")

    print("\nCOCO 형식으로 변환 및 분할 중...")
    split_and_save(all_pairs, OUTPUT_DIR)

    print("\n완료! 생성된 파일:")
    for f in ["train.json", "val.json", "test.json"]:
        fpath = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  {fpath}  ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
