

# Model

## 1. Model Used: Faster R-CNN (ResNet-50 + FPN)

The selected model is **Faster R-CNN with a ResNet-50 backbone and Feature Pyramid Network (FPN)**, available in the **BDD100K model zoo**.  
This model was chosen because it provides a strong balance between detection accuracy, computational efficiency, and interpretability, which makes it suitable for autonomous driving datasets such as BDD100K.

---

### Why This Model Was Chosen

1. **Two-Stage Detection Architecture**  
   Faster R-CNN is a two-stage detector that first proposes regions using a Region Proposal Network (RPN) and then classifies and refines these proposals.  
   This approach provides higher localization accuracy and better handling of overlapping objects compared to single-stage detectors.

2. **ResNet-50 Backbone**  
   ResNet-50 is a widely used residual network that provides deep feature representations while maintaining computational efficiency.  
   It offers an excellent balance between accuracy and model size, making it ideal for quick experimentation and training on subsets.

3. **Feature Pyramid Network (FPN)**  
   The FPN module enhances the detector’s ability to recognize objects at multiple scales by combining low- and high-level feature maps.  
   This is particularly important for the BDD100K dataset, which contains many small and distant objects like traffic signs and lights.

4. **Dataset Suitability**  
   The BDD100K model zoo provides models already trained or fine-tuned on similar data, which ensures better domain alignment.  
   This reduces the need for extensive training while maintaining strong baseline performance.

5. **Extendability**  
   The Faster R-CNN architecture allows straightforward experimentation with modifications such as Cascade R-CNN, focal loss, or additional heads for other perception tasks.

---

### Alternative Models Considered

| Model | Pros | Cons |
|--------|------|------|
| YOLO / RetinaNet | High speed | Lower accuracy, poor small-object performance |
| ResNet-101 backbone | Higher accuracy | Slower and heavier |
| DETR | End-to-end transformer-based design | Requires large-scale training and long convergence time |

**Reason for final choice:** Faster R-CNN with ResNet-50 + FPN provides a balanced trade-off between speed, accuracy, and interpretability for BDD100K-scale object detection.
Also I am using my personal Macbook for all the tests it has not that powerful has M1 chip with an inbuilt gpu, selected the one that I could run locally as well thats why didn't select deeper models.

---

### Model Architecture Summary

| Component | Description |
|------------|-------------|
| Backbone | ResNet-50 pretrained on ImageNet |
| Neck | Feature Pyramid Network for multi-scale features |
| RPN | Region Proposal Network to generate candidate regions |
| ROI Heads | Classification and bounding box regression |
| Loss Functions | Cross-entropy and smooth L1 loss |

---

## 2. Training Pipeline

### Script
The training code is provided in `tasks/2_model/train_one_epoch.py`.  
This script demonstrates the end-to-end training pipeline using a small subset of BDD100K.

---

### How to Run

```bash
# From the repository root
cd tasks/2_model/
python3 train_one_epoch.py --subset 256 --batch 4 --epochs 1
```

#### Arguments

| Argument | Default | Description |
|-----------|----------|-------------|
| `--subset` | 256 | Number of samples to use for quick iteration |
| `--batch` | 4 | Batch size |
| `--epochs` | 1 | Number of epochs to train |
| `--repo_root` | auto-detected | Repository root path |

If `--repo_root` is not specified, it defaults to two levels above the script file.

---

### Checkpoint Output

After running the script, the model checkpoint is saved automatically at:

```
<repo_root>/checkpoints/fasterrcnn_one_epoch.pth
```

For example, if executed from the repository root:
```
./checkpoints/fasterrcnn_one_epoch.pth
```

---

### Notes

- The script automatically uses GPU (CUDA) if available.  
- To use pretrained weights from the BDD model zoo, modify the line:
  ```python
  model = build_model(num_classes=len(class_map) + 1, pretrained=True)
  ```
- The dataset loader excludes irrelevant labels:
  ```python
  exclude_classes=['lane', 'drivable area']
  ```
- A random seed (`random.seed(0)`) is used for reproducibility when selecting subsets.

---

## Summary

The Faster R-CNN (ResNet-50 + FPN) model was chosen for its proven performance in autonomous driving detection tasks, its robustness across scales, and its efficient training characteristics.  
The provided training pipeline successfully loads the BDD100K dataset, runs a one-epoch training iteration, and saves a reproducible model checkpoint, satisfying the requirements for model selection and minimal training demonstration.