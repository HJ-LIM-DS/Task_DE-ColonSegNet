import json
import os

json_path = "E:/Task_DE-ColonSegNet/datasets/scan_result.json"

if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for ds_name, ds_info in data.items():
        print(f"Dataset: {ds_name}")
        for struct in ds_info.get("structure", []):
            if struct.get("num_files", 0) > 0:
                print(f"  - {struct['rel_path']}: {struct['num_files']} files {struct.get('extensions', {})}")
        print("-" * 50)
else:
    print("File not found.")
