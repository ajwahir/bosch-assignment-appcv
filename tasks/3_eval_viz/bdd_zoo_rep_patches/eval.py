"""
Evaluation and Visualization for BDD100K Detection Outputs
"""

import os
import pickle
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.coco as fouc
import fiftyone.brain as fob
from fiftyone import ViewField as F

# # -----------------------------
# # CONFIG
# # -----------------------------
# VAL_ANN_FILE = "../../../../data/bdd100k_labels_release/bdd100k/labels/det_val_cocofmt.json"
# VAL_IMG_DIR = "../../../../data/bdd100k_images_100k/bdd100k/images/100k/val"
# OUTPUTS_PKL = "results/raw_predictions/outputs.pkl"
# RESULTS_DIR = "results/analysis"
# CONF_THRESHOLD = 0.3

# os.makedirs(RESULTS_DIR, exist_ok=True)

# # -----------------------------
# # LOAD DATA
# # -----------------------------
# with open(OUTPUTS_PKL, "rb") as f:
#     outputs = pickle.load(f)



# # Get COCO category IDs in the same order


# dt_file = os.path.join(RESULTS_DIR, "detections.json")
# with open(dt_file, "w") as f:
#     json.dump(coco_dt, f)

def plot_pr_curves(coco_eval, out_dir="results/analysis/pr_curves"):
    """
    Plot precision-recall curves per class from a COCOeval object.
    Saves one PNG per class, with AUC in the title.
    """
    os.makedirs(out_dir, exist_ok=True)

    precisions = coco_eval.eval['precision']  # [TxRxKxAxM]
    # dimensions: IoU thresholds x recall thresholds x classes x area ranges x maxDets
    cat_ids = coco_eval.params.catIds
    cat_names = [c["name"] for c in coco_eval.cocoGt.loadCats(cat_ids)]

    # Use area=all (index 0), maxDets=100 (index -1)
    area_ind = 0
    maxdet_ind = -1

    for cls_ind, cat_name in enumerate(cat_names):
        # precision: IoU thresholds x recall thresholds
        pr = precisions[:, :, cls_ind, area_ind, maxdet_ind]
        if pr.size == 0:
            continue
        # average over IoU thresholds
        pr = np.mean(pr, axis=0)
        recall = np.linspace(0.0, 1.0, pr.shape[0])

        # compute AUC
        auc = np.trapz(pr, recall)

        plt.figure()
        plt.plot(recall, pr, label=f"{cat_name} (AUC={auc:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve - {cat_name}\nAUC={auc:.3f}")
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.grid(True)
        plt.legend(loc="lower left")
        out_path = os.path.join(out_dir, f"pr_curve_{cat_name.replace(' ', '_')}.png")
        plt.savefig(out_path)
        plt.close()

    print(f"Saved per-class PR curves with AUC to {out_dir}")



def coco_classwise_eval(coco_gt, coco_dt, iou_type="bbox", out_csv="results/analysis/classwise_coco_eval.csv"):
    """
    Run COCO evaluation per class and save results to CSV.
    """
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    cat_ids = coco_gt.getCatIds()
    cat_names = [c["name"] for c in coco_gt.loadCats(cat_ids)]

    records = []

    for cat_id, cat_name in zip(cat_ids, cat_names):
        coco_eval.params.catIds = [cat_id]
        coco_eval.params.imgIds = sorted(coco_gt.getImgIds())
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics from coco_eval.stats
        # stats = [AP, AP50, AP75, AP_small, AP_medium, AP_large,
        #          AR1, AR10, AR100, AR_small, AR_medium, AR_large]
        stats = coco_eval.stats.tolist()

        record = {"category": cat_name}
        record.update({
            "AP": stats[0],
            "AP50": stats[1],
            "AP75": stats[2],
            "AP_small": stats[3],
            "AP_medium": stats[4],
            "AP_large": stats[5],
            "AR@1": stats[6],
            "AR@10": stats[7],
            "AR@100": stats[8],
            "AR_small": stats[9],
            "AR_medium": stats[10],
            "AR_large": stats[11],
        })
        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"Saved classwise COCO metrics to {out_csv}")
    return df

def get_label_from_class_id(class_id: int) -> str:
    """
    Map a class index to its corresponding BDD100K label.
    """
    classes = [
        "person", "rider", "car", "truck", "bus",
        "train", "motorcycle", "bicycle", "traffic light", "traffic sign",
    ]
    return classes[class_id]


def build_fiftyone_dataset(coco_gt, outputs, img_ids, val_img_dir, conf_threshold=0.3) -> fo.Dataset:
    """
    Build a FiftyOne dataset with ground truth and predictions.
    """
    if "bdd_val_eval" in fo.list_datasets():
        fo.delete_dataset("bdd_val_eval")

    dataset = fo.Dataset("bdd_val_eval", persistent=False)

    for idx, img_id in enumerate(img_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        filepath = os.path.join(val_img_dir, img_info["file_name"])
        sample = fo.Sample(filepath=filepath)

        # Ground truth
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        anns = coco_gt.loadAnns(ann_ids)
        gt_dets = [
            fo.Detection(
                label=coco_gt.loadCats([ann["category_id"]])[0]["name"],
                bounding_box=[
                    ann["bbox"][0] / img_info["width"],
                    ann["bbox"][1] / img_info["height"],
                    ann["bbox"][2] / img_info["width"],
                    ann["bbox"][3] / img_info["height"],
                ],
            )
            for ann in anns
        ]
        sample["ground_truth"] = fo.Detections(detections=gt_dets)

        # Metadata
        sample["daytime"] = img_info.get("attributes", {}).get("timeofday")
        sample["weather"] = img_info.get("attributes", {}).get("weather")
        sample["scene"] = img_info.get("attributes", {}).get("scene")
        sample["width"] = img_info["width"]
        sample["height"] = img_info["height"]
        sample["area"] = img_info["width"] * img_info["height"]

        # Predictions
        detections = []
        for class_id, dets in enumerate(outputs[idx]):
            for det in dets:
                x1, y1, x2, y2, score = det.tolist()
                if score < conf_threshold:
                    continue
                w, h = x2 - x1, y2 - y1
                detections.append(
                    fo.Detection(
                        label=get_label_from_class_id(class_id),
                        bounding_box=[
                            x1 / img_info["width"],
                            y1 / img_info["height"],
                            w / img_info["width"],
                            h / img_info["height"],
                        ],
                        confidence=float(score),
                    )
                )
        sample["predictions"] = fo.Detections(detections=detections)
        sample["num_predictions"] = len(detections)
        if detections:
            sample["max_confidence"] = max(det.confidence for det in detections)

        dataset.add_sample(sample)

    return dataset


def define_cluster_views(dataset: fo.Dataset) -> dict:
    """
    Define useful cluster views for failure analysis.
    """
    return {
        "1": dataset.filter_labels("predictions", F("eval") == "fp"),
        "2": dataset.filter_labels("ground_truth", F("eval") == "fn"),
        "3": dataset.filter_labels(
            "ground_truth",
            (F("eval") == "fn")
            & ((F("bounding_box")[2] * F("bounding_box")[3]) < 0.02),
        ),
        "4": dataset.match(F("daytime") == "night"),
        "5": dataset.filter_labels(
            "predictions", (F("eval") == "fp") & (F("confidence") > 0.9)
        ),
        "6": dataset.filter_labels("ground_truth", F("label") == "train"),
        "7": dataset.filter_labels(
            "ground_truth", (F("label") == "person") & (F("eval") == "fn")
        ),
    }


def main(coco_gt, outputs, img_ids, val_img_dir):
    """
    Main entry point for FiftyOne visualization and failure analysis.
    """
    # # -----------------------------
    # # COCO EVALUATION
    # # -----------------------------
    
    if "info" not in coco_gt.dataset:
        coco_gt.dataset["info"] = {}
    if "licenses" not in coco_gt.dataset:
        coco_gt.dataset["licenses"] = []
    coco_dt_obj = coco_gt.loadRes(dt_file)

    coco_eval = COCOeval(coco_gt, coco_dt_obj, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    plot_pr_curves(coco_eval)

    classwise_df = coco_classwise_eval(coco_gt, coco_dt_obj, out_csv=os.path.join(RESULTS_DIR, "classwise_coco_eval.csv"))
    print(classwise_df)

    # Build dataset
    fo_dataset = build_fiftyone_dataset(coco_gt, outputs, img_ids, val_img_dir)

    # Evaluate detections
    results = fo_dataset.evaluate_detections(
        "predictions",
        gt_field="ground_truth",
        eval_key="eval",
        iou=0.5,
    )
    print(results.metrics())

    # Define cluster views
    views = define_cluster_views(fo_dataset)

    # Compute embeddings for predictions
    fob.compute_visualization(
        fo_dataset,
        patches_field="predictions",
        brain_key="pred_viz",
    )

    # Launch the App
    session = fo.launch_app(fo_dataset)

    print("\nKeyboard controls:")
    print("1 = False Positives")
    print("2 = False Negatives")
    print("3 = Small missed objects")
    print("4 = Nighttime errors")
    print("5 = Overconfident false positives")
    print("6 = Rare class: train")
    print("7 = Missed pedestrians")
    print("q = quit\n")

    while True:
        key = input("Enter view number (1-7) or q to quit: ").strip()
        if key == "q":
            break
        if key in views:
            session.view = views[key]
            print(f"Switched to view {key}")
        else:
            print("Invalid key, try again")

    input("Press Enter to exit FiftyOne...")


if __name__ == "__main__":
    # -----------------------------
    # CONFIG
    # -----------------------------
    VAL_ANN_FILE = "../../../../data/bdd100k_labels_release/bdd100k/labels/det_val_cocofmt.json"
    VAL_IMG_DIR = "../../../../data/bdd100k_images_100k/bdd100k/images/100k/val"
    OUTPUTS_PKL = "results/raw_predictions/outputs.pkl"
    RESULTS_DIR = "results/analysis"
    CONF_THRESHOLD = 0.3
    # -----------------------------
    # LOAD DATA
    # -----------------------------
    with open(OUTPUTS_PKL, "rb") as f:
        outputs = pickle.load(f)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    dt_file = os.path.join(RESULTS_DIR, "detections.json")
    coco_gt = COCO(VAL_ANN_FILE)
    img_ids = coco_gt.getImgIds()
    # Build mapping from class index to COCO category_id
    bdd_classes = [
        "person", "rider", "car", "truck", "bus",
        "train", "motor", "bike", "traffic light", "traffic sign"]
    cat_ids = coco_gt.getCatIds(catNms=bdd_classes)
    class_to_catid = {i: cid for i, cid in enumerate(cat_ids)}

    class_to_catid[0]=6
    class_to_catid[1]=4
    class_to_catid[2]=3
    class_to_catid[3]=8
    class_to_catid[4]=7
    class_to_catid[5]=10
    class_to_catid[6]=5
    class_to_catid[7]=9
    class_to_catid[8]=2
    class_to_catid[9]=1

    print("Number of classes in model outputs:", len(outputs[0]))
    print("class_to_catid:", class_to_catid)

    # -----------------------------
    # PREPARE DETECTIONS FOR COCO EVAL
    # -----------------------------
    coco_dt = []

    for idx, img_id in enumerate(img_ids):
        pred_per_image = outputs[idx]
        coco_dt_img = []
        for class_id, dets in enumerate(pred_per_image):
            if dets.shape[0] == 0:
                continue
            coco_cat_id = class_to_catid[class_id]   # << use mapping here
            for det in dets:
                x1, y1, x2, y2, score = det.tolist()
                if score < CONF_THRESHOLD:
                    continue
                w, h = x2 - x1, y2 - y1
                coco_dt_img.append({
                    "image_id": img_id,
                    "category_id": coco_cat_id,
                    "bbox": [x1, y1, w, h],
                    "score": score
                })
        coco_dt.extend(coco_dt_img)
    with open(dt_file, "w") as f:
        json.dump(coco_dt, f)

    coco_gt = COCO(VAL_ANN_FILE)
    img_ids = coco_gt.getImgIds()
    main(coco_gt, outputs, img_ids, VAL_IMG_DIR)
