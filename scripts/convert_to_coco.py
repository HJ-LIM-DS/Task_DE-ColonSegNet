import os
import json
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

def create_coco_format():
    return {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "polyp", "supercategory": "lesion"}]
    }

def get_contours_from_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None, None
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotations = []
    for contour in contours:
        if len(contour) < 3:
            continue
        # Polygon
        segmentation = contour.flatten().tolist()
        # BBox
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        if area > 10:  # Ignore too small artifacts
            annotations.append({
                "segmentation": [segmentation],
                "bbox": [x, y, w, h],
                "area": area,
                "iscrowd": 0,
                "category_id": 1
            })
    return annotations, (mask.shape[1], mask.shape[0])

def process_datasets(datasets_info, output_dir):
    coco_train = create_coco_format()
    coco_val = create_coco_format()
    
    img_id = 1
    ann_id = 1
    
    all_data = []
    
    # Collect all image and mask paths
    for ds_name, info in datasets_info.items():
        base_dir = info['base']
        img_dir = os.path.join(base_dir, info['img_dir'])
        mask_dir = os.path.join(base_dir, info['mask_dir'])
        img_ext = info['img_ext']
        mask_ext = info['mask_ext']
        
        img_paths = glob(os.path.join(img_dir, f"*{img_ext}"))
        for img_path in img_paths:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(mask_dir, f"{basename}{mask_ext}")
            if os.path.exists(mask_path):
                all_data.append((img_path, mask_path))

    # Split dataset
    train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)
    
    def build_coco(data_list, coco_dict, start_img_id, start_ann_id):
        curr_img_id = start_img_id
        curr_ann_id = start_ann_id
        for img_path, mask_path in data_list:
            anns, (width, height) = get_contours_from_mask(mask_path)
            if anns is None or len(anns) == 0:
                continue
                
            coco_dict["images"].append({
                "id": curr_img_id,
                "file_name": os.path.abspath(img_path),
                "width": width,
                "height": height
            })
            
            for ann in anns:
                ann["id"] = curr_ann_id
                ann["image_id"] = curr_img_id
                coco_dict["annotations"].append(ann)
                curr_ann_id += 1
            
            curr_img_id += 1
            
        return curr_img_id, curr_ann_id

    img_id, ann_id = build_coco(train_data, coco_train, img_id, ann_id)
    build_coco(val_data, coco_val, img_id, ann_id)
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.json"), "w") as f:
        json.dump(coco_train, f)
    with open(os.path.join(output_dir, "val.json"), "w") as f:
        json.dump(coco_val, f)

    print(f"Train images: {len(coco_train['images'])}, Train annotations: {len(coco_train['annotations'])}")
    print(f"Val images: {len(coco_val['images'])}, Val annotations: {len(coco_val['annotations'])}")

if __name__ == "__main__":
    datasets = {
        "kvasir-seg": {
            "base": "E:/Task_DE-ColonSegNet/datasets/kvasir-seg",
            "img_dir": "images", "mask_dir": "masks",
            "img_ext": ".jpg", "mask_ext": ".jpg"
        },
        "CVC-ClinicDB": {
            "base": "E:/Task_DE-ColonSegNet/datasets/CVC-ClinicDB",
            "img_dir": "PNG/Original", "mask_dir": "PNG/Ground Truth",
            "img_ext": ".png", "mask_ext": ".png"
        },
        "ETIS-LaribPolypDB": {
            "base": "E:/Task_DE-ColonSegNet/datasets/ETIS-LaribPolypDB",
            "img_dir": "images", "mask_dir": "masks",
            "img_ext": ".png", "mask_ext": ".png"
        }
    }
    
    out_dir = "E:/Task_DE-ColonSegNet/datasets/coco_format"
    print("Converting datasets to COCO format...")
    process_datasets(datasets, out_dir)
    print("Done!")
