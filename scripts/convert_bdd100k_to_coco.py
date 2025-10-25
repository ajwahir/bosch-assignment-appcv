import json
import os
from tqdm import tqdm

# Adjust these paths according to your folder structure
data_root = "../data/"
labels_dir = os.path.join(data_root, "bdd100k_labels_release/bdd100k/labels")
images_dir = os.path.join(data_root, "bdd100k_images_100k/bdd100k/images/100k")

def bdd_to_coco(bdd_json_path, img_prefix, output_json_path):
    with open(bdd_json_path, "r") as f:
        bdd_data = json.load(f)

    coco = {"images": [], "annotations": [], "categories": []}
    categories = {}
    ann_id = 1

    for img_id, item in enumerate(tqdm(bdd_data, desc=f"Converting {os.path.basename(bdd_json_path)}")):
        # Images (now with attributes preserved)
        coco["images"].append({
            "id": img_id,
            "file_name": item["name"],
            "height": 720,
            "width": 1280,
            "attributes": item.get("attributes", {})  # <-- preserve weather/scene/timeofday
        })

        # Annotations
        for label in item.get("labels", []):
            if "box2d" in label:
                if label["category"] not in categories:
                    categories[label["category"]] = len(categories) + 1
                cat_id = categories[label["category"]]
                x1, y1, x2, y2 = label["box2d"].values()
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0
                })
                ann_id += 1

    # Categories
    coco["categories"] = [{"id": v, "name": k} for k, v in categories.items()]

    # Save COCO JSON
    with open(output_json_path, "w") as f:
        json.dump(coco, f, indent=4)
    print(f"Saved COCO-format JSON to: {output_json_path}")


if __name__ == "__main__":
    train_bdd_json = os.path.join(labels_dir, "bdd100k_labels_images_train.json")
    val_bdd_json   = os.path.join(labels_dir, "bdd100k_labels_images_val.json")

    train_output_json = os.path.join(labels_dir, "det_train_cocofmt.json")
    val_output_json   = os.path.join(labels_dir, "det_val_cocofmt.json")

    # Train
    bdd_to_coco(train_bdd_json, labels_dir, train_output_json)

    # Val
    bdd_to_coco(val_bdd_json, labels_dir, val_output_json)
