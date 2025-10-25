"""BDD100K dataset analysis utilities.

Provides:
- Loader: loads train/val JSONs and basic file/folder info
- Analysis: computes multiple dataset QC/EDA metrics and stores results in a master JSON

Each analysis method updates a single key in the master structure. Methods are intentionally
simple and independent so they can be run individually or all together via `run_all()`.
"""
from collections import Counter, defaultdict
from pathlib import Path
import json
import math
from typing import Dict, List, Any, Tuple
from PIL import Image, ImageDraw


class Loader:
    """Loads annotation JSONs and exposes basic dataset paths and metadata.

    Attributes:
        train_json_path, val_json_path: input JSON files
        train, val: loaded lists of annotation dicts
        train_images_dir, val_images_dir: inferred image dirs (user can override)
    """

    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self.train_json_path = self.repo_root / "data" / "bdd100k_labels_release" / "bdd100k" / "labels" / "bdd100k_labels_images_train.json"
        self.val_json_path = self.repo_root / "data" / "bdd100k_labels_release" / "bdd100k" / "labels" / "bdd100k_labels_images_val.json"
        self.train_images_dir = self.repo_root / "data" / "bdd100k_images_100k" / "bdd100k" / "images" / "100k" / "train"
        self.val_images_dir = self.repo_root / "data" / "bdd100k_images_100k" / "bdd100k" / "images" / "100k" / "val"
        self.train: List[Dict[str, Any]] = []
        self.val: List[Dict[str, Any]] = []

    def load(self):
        """Load train and val JSONs into memory (streaming would be used for huge files).

        Returns: tuple (train_count, val_count)
        """
        if self.train_json_path.exists():
            with open(self.train_json_path, 'r') as f:
                self.train = json.load(f)
        else:
            self.train = []

        if self.val_json_path.exists():
            with open(self.val_json_path, 'r') as f:
                self.val = json.load(f)
        else:
            self.val = []

        return len(self.train), len(self.val)


class Analysis:
    """Compute dataset analysis metrics and update a master JSON structure.

    Use: create with a Loader instance, then call methods to fill analysis keys.
    Each method updates one key in self.results.

    Configurable parameters (passed to constructor) control thresholds and
    filtering applied across analyses:
        - exclude_classes: list of category names to ignore in any analysis.
        - aspect_ratio_thresh: ratio above which a bbox is considered extreme.
        - tiny_area_norm: normalized area threshold below which a box is "tiny"
            (normalized by image_area param or default 1280*720).
        - rare_pct: classes with proportion < rare_pct (0-1) are considered rare.
        - max_examples_list: maximum number of image names to include in any
            "examples" listing to avoid huge outputs.
        - small_box_pixel_thresh: pixel-area threshold used in image-level small-box
            heuristics (e.g., 50*50 by default).

    Note on "meta issues": we consider these problems as metadata issues in
    this analysis (and report them in `metadata_issues`):
        - unexpected/unknown values in attributes like 'weather', 'timeofday',
            'scene'.
        - inconsistent or malformed attribute types (e.g., non-string weather)
        - missing or malformed 'attributes' field (not a dict) or unparsable
            timestamps.  (We deliberately do NOT report a global missing_timestamps
            count here per your request.)
    """

    def __init__(self, loader: Loader,
                                exclude_classes: List[str] = None,
                                aspect_ratio_thresh: float = 20.0,
                                tiny_area_norm: float = 0.001,
                                rare_pct: float = 0.0,
                                max_examples_list: int = 100,
                                small_box_pixel_thresh: int = 50 * 50, 
                                save_extreme_dir: str = None,
                                save_annotator_noise_dir: str = None):
        self.loader = loader
        self.exclude_classes = set(exclude_classes or [])
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.tiny_area_norm = tiny_area_norm
        self.rare_pct = float(rare_pct)
        self.max_examples_list = int(max_examples_list)
        self.small_box_pixel_thresh = int(small_box_pixel_thresh)
        self.save_extreme_dir = save_extreme_dir
        self.save_annotator_noise_dir = save_annotator_noise_dir

        self.results: Dict[str, Any] = {
                'analysis_parameters': {},
                'dataset_info': {},
                'file_existence': {},
                'class_counts': {},
                'split_sizes': {},
                'bbox_stats': {},
                'cooccurrence': {},
                'temporal_metadata': {},
                'annotator_noise': {},
                'image_level_counts': {},
                'mislabeled_boxes': {},
                'class_imbalance': {},
                'split_leakage': {},
                'metadata_issues': {},
        }

    def _filter_labels(self, labels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Return labels list with any excluded classes removed."""
            if not labels:
                    return []
            return [lab for lab in labels if lab.get('category') not in self.exclude_classes]

    def _iter_items(self):
        for item in self.loader.train:
            yield 'train', item
        for item in self.loader.val:
            yield 'val', item

    def compute_dataset_info(self):
        """Populate basic counts: number of images in train/val and file paths."""
        train_count = len(self.loader.train)
        val_count = len(self.loader.val)
        self.results['dataset_info'] = {
            'train_annotations': train_count,
            'val_annotations': val_count,
            'train_images_dir': str(self.loader.train_images_dir),
            'val_images_dir': str(self.loader.val_images_dir),
        }

    def check_file_existence(self):
        """Check for images referenced in annotations and report missing counts per split."""
        missing = {'train': 0, 'val': 0}
        total = {'train': 0, 'val': 0}
        for split, item in self._iter_items():
            total[split] += 1
            img_name = item.get('name')
            if not img_name:
                missing[split] += 1
                continue
            img_path = (self.loader.train_images_dir if split == 'train' else self.loader.val_images_dir) / img_name
            if not img_path.exists():
                missing[split] += 1

        self.results['file_existence'] = {'missing_per_split': missing, 'total_per_split': total}

    def classwise_counts(self):
        """Compute classwise counts across train+val and update results."""
        c = Counter()
        for _, item in self._iter_items():
            for lab in self._filter_labels(item.get('labels', [])):
                cat = lab.get('category', 'unknown')
                c[cat] += 1

        self.results['class_counts'] = dict(c)

    def split_sizes(self):
        """Record number of images per split as a metric."""
        self.results['split_sizes'] = {
            'train_images': len(list(self.loader.train_images_dir.glob('*.jpg'))) if self.loader.train_images_dir.exists() else None,
            'val_images': len(list(self.loader.val_images_dir.glob('*.jpg'))) if self.loader.val_images_dir.exists() else None,
            'train_annotations': len(self.loader.train),
            'val_annotations': len(self.loader.val),
            'missing_image_annotations': len(self.loader.train) - len(list(self.loader.train_images_dir.glob('*.jpg'))) if self.loader.train_images_dir.exists() else None
        }

    def bbox_statistics(self, normalize_by_image_area=True):
        """Compute bbox area stats per class. If normalize_by_image_area is True,
        areas are divided by the image area inferred from the first image entry (approx).
        Updates 'bbox_stats' with per-class area distributions (min, median, mean, max).
        """
        per_class_areas = defaultdict(list)

        # naive image area guess: try to extract from attributes or default to 1280x720
        default_area = 1280 * 720

        # For speed, we won't open images; we'll normalize by default area unless the annotation contains size info.
        for _, item in self._iter_items():
            for lab in self._filter_labels(item.get('labels', [])):
                box = lab.get('box2d')
                if not box:
                    continue
                x1, y1, x2, y2 = box.get('x1'), box.get('y1'), box.get('x2'), box.get('y2')
                if None in (x1, y1, x2, y2):
                    continue
                area = max(0.0, (x2 - x1) * (y2 - y1))
                if normalize_by_image_area:
                    area = area / float(default_area)
                per_class_areas[lab.get('category', 'unknown')].append(area)

        out = {}
        for cls, vals in per_class_areas.items():
            if not vals:
                continue
            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            mean = sum(vals_sorted) / n
            median = vals_sorted[n//2]
            out[cls] = {'count': n, 'min': vals_sorted[0], 'median': median, 'mean': mean, 'max': vals_sorted[-1]}

        self.results['bbox_stats'] = out

    def cooccurrence(self):
        """Compute simple co-occurrence counts and normalized conditional frequencies.

        Sets 'cooccurrence': {pair: count, conditional: {A: {B: freq}}}
        """
        co = Counter()
        per_image = []
        for _, item in self._iter_items():
            cats = [lab.get('category', 'unknown') for lab in self._filter_labels(item.get('labels', []))]
            uniq = sorted(set(cats))
            per_image.append(uniq)
            for i in range(len(uniq)):
                for j in range(i+1, len(uniq)):
                    pair = (uniq[i], uniq[j])
                    co[pair] += 1

        # conditional frequencies P(B|A) approximated as co(A,B) / count(A)
        countA = Counter()
        for lst in per_image:
            for a in set(lst):
                countA[a] += 1

        conditional = {}
        for (a, b), v in co.items():
            conditional.setdefault(a, {})[b] = v / countA[a] if countA[a] else 0.0
            conditional.setdefault(b, {})[a] = v / countA[b] if countA[b] else 0.0

        # Convert tuple keys to string for JSON serialization
        pair_counts = {f"{a}|{b}": v for (a, b), v in co.items()}
        self.results['cooccurrence'] = {'pair_counts': pair_counts, 'conditional': conditional}

    def temporal_and_meta(self):
        """Compute metadata distributions: daytime, weather, scene vs class frequency."""
        # collect metadata counts
        meta_counts = {'timeofday': Counter(), 'weather': Counter(), 'scene': Counter()}
        class_by_time = defaultdict(Counter)
        class_by_weather = defaultdict(Counter)
        class_by_scene = defaultdict(Counter)

        for _, item in self._iter_items():
            attrs = item.get('attributes', {}) or {}
            tod = attrs.get('timeofday', 'unknown')
            weather = attrs.get('weather', 'unknown')
            scene = attrs.get('scene', 'unknown')
            meta_counts['timeofday'][tod] += 1
            meta_counts['weather'][weather] += 1
            meta_counts['scene'][scene] += 1
            for lab in self._filter_labels(item.get('labels', [])):
                cat = lab.get('category', 'unknown')
                class_by_time[tod][cat] += 1
                class_by_weather[weather][cat] += 1
                class_by_scene[scene][cat] += 1

        self.results['temporal_metadata'] = {
            'meta_counts': {k: dict(v) for k, v in meta_counts.items()},
            'class_by_time': {k: dict(v) for k, v in class_by_time.items()},
            'class_by_weather': {k: dict(v) for k, v in class_by_weather.items()},
            'class_by_scene': {k: dict(v) for k, v in class_by_scene.items()},
        }

    def annotator_noise(self, iou_threshold=0.95):
        """Detect overlapping boxes with same class and IoU > iou_threshold (possible duplicates).

        Adds 'annotator_noise' with counts and a few example image names.
        """
        def iou(a, b):
            xA = max(a[0], b[0]); yA = max(a[1], b[1]); xB = min(a[2], b[2]); yB = min(a[3], b[3])
            inter = max(0, xB - xA) * max(0, yB - yA)
            areaA = max(0, (a[2]-a[0])*(a[3]-a[1]))
            areaB = max(0, (b[2]-b[0])*(b[3]-b[1]))
            union = areaA + areaB - inter
            return inter / union if union > 0 else 0.0

            # record used parameter
        self.results['analysis_parameters']['annotator_noise_iou'] = float(iou_threshold)
        count = 0
        examples = []
        # collect duplicate boxes per image so we can save annotated examples
        dup_boxes_per_image = defaultdict(list)  # img_name -> list of (box, category, is_dup)
        dup_pairs_per_image = defaultdict(list)  # img_name -> list of {'a':box,'b':box,'category':cat}
        for _, item in self._iter_items():
            name = item.get('name')
            labs = [lab for lab in self._filter_labels(item.get('labels', [])) if lab.get('box2d')]
            # initialize all boxes as not-duplicate
            for lab in labs:
                b = lab['box2d']
                dup_boxes_per_image[name].append({'box': b, 'category': lab.get('category', 'unknown'), 'is_dup': False})

            # find duplicates among same-category boxes
            for i in range(len(labs)):
                for j in range(i+1, len(labs)):
                    if labs[i].get('category') != labs[j].get('category'):
                        continue
                    a = labs[i]['box2d']; b = labs[j]['box2d']
                    bb1 = (a['x1'], a['y1'], a['x2'], a['y2'])
                    bb2 = (b['x1'], b['y1'], b['x2'], b['y2'])
                    if iou(bb1, bb2) >= iou_threshold:
                        count += 1
                        examples.append(name)
                        # record the overlapping pair for later drawing
                        dup_pairs_per_image[name].append({'a': a, 'b': b, 'category': labs[i].get('category', 'unknown')})
                        # mark corresponding boxes as duplicates in our per-image store
                        # find and mark matching boxes (match by coords)
                        for entry in dup_boxes_per_image[name]:
                            bx = entry['box']
                            if bx.get('x1') == a.get('x1') and bx.get('y1') == a.get('y1') and bx.get('x2') == a.get('x2') and bx.get('y2') == a.get('y2'):
                                entry['is_dup'] = True
                            if bx.get('x1') == b.get('x1') and bx.get('y1') == b.get('y1') and bx.get('x2') == b.get('x2') and bx.get('y2') == b.get('y2'):
                                entry['is_dup'] = True

        # save annotated images for duplicates if requested
        saved = 0
        save_dir = Path(self.save_annotator_noise_dir) if self.save_annotator_noise_dir else None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            for img_name, entries in dup_boxes_per_image.items():
                # only save images that have at least one overlapping pair
                pairs = dup_pairs_per_image.get(img_name, [])
                if not pairs:
                    continue
                p_train = self.loader.train_images_dir / img_name
                p_val = self.loader.val_images_dir / img_name
                p = p_train if p_train.exists() else (p_val if p_val.exists() else None)
                if p is None:
                    continue
                try:
                    im = Image.open(p).convert('RGB')
                    draw = ImageDraw.Draw(im)
                    # draw only overlapping pairs: two boxes per pair in different colors
                    for pair in pairs:
                        a = pair.get('a')
                        b = pair.get('b')
                        cat = pair.get('category', '')
                        try:
                            # first box in red, second in orange
                            draw.rectangle([(a['x1'], a['y1']), (a['x2'], a['y2'])], outline='red', width=3)
                            draw.rectangle([(b['x1'], b['y1']), (b['x2'], b['y2'])], outline='orange', width=3)
                            # labels for each box
                            t1 = f"{cat} (dup A)"
                            t2 = f"{cat} (dup B)"
                            tx1, ty1 = int(a['x1']), int(max(0, a['y1'] - 14))
                            tx2, ty2 = int(b['x1']), int(max(0, b['y1'] - 14))
                            w1 = max(20, len(t1) * 6); h1 = 12
                            w2 = max(20, len(t2) * 6); h2 = 12
                            draw.rectangle([(tx1, ty1), (tx1 + w1, ty1 + h1)], fill='black')
                            draw.text((tx1 + 1, ty1), t1, fill='white')
                            draw.rectangle([(tx2, ty2), (tx2 + w2, ty2 + h2)], fill='black')
                            draw.text((tx2 + 1, ty2), t2, fill='white')
                        except Exception:
                            # ignore drawing failures for this pair
                            pass
                    outp = save_dir / img_name
                    im.save(outp)
                    saved += 1
                except Exception:
                    # ignore per-image save errors
                    continue

        self.results['annotator_noise'] = {'possible_duplicates': count, 'images': list(dict.fromkeys(examples)), 'annotator_examples_saved': saved}

    def image_level_counts(self):
        """Compute per-image object counts, images with zero objects, images with many small objects.

        Updates 'image_level_counts' with basic distributions.
        """
        obj_counts = []
        zero_images = 0
        many_small = 0
        for _, item in self._iter_items():
            labs = self._filter_labels(item.get('labels', []))
            n = len(labs)
            obj_counts.append(n)
            if n == 0:
                zero_images += 1
            # count many small objects (naive): more than 10 objects and average box area small
            small_boxes = 0
            for lab in labs:
                box = lab.get('box2d')
                if not box:
                    continue
                area = (box['x2'] - box['x1']) * (box['y2'] - box['y1'])
                if area < 50 * 50:  # < 50x50 px
                    small_boxes += 1
            if n > 10 and small_boxes > n * 0.5:
                many_small += 1

        self.results['image_level_counts'] = {
            'total_images_analyzed': len(obj_counts),
            'zero_object_images': zero_images,
            'many_small_objects_images': many_small,
            'obj_count_histogram': dict(Counter(obj_counts))
        }

    def detect_mislabeled(self, image_area=None):
        """Detect mislabeled boxes.

        Criteria (configurable via constructor):
          - aspect ratio > self.aspect_ratio_thresh => 'extreme_ar'
          - area/image_area < self.tiny_area_norm => 'tiny'
          - boxes outside image bounds (assumed 1280x720) => 'outside'

        The function returns counts for each category and a bounded list of example
        image names (capped by self.max_examples_list). Very large example lists
        are truncated to avoid huge JSON outputs.
        """
        # resolve thresholds: prefer explicit args, otherwise instance defaults
        aspect_ratio_thresh = float(self.aspect_ratio_thresh) if self.aspect_ratio_thresh is not None else float(self.aspect_ratio_thresh)
        tiny_area_norm = self.tiny_area_norm
        image_area = float(image_area) if image_area is not None else float(1280 * 720)

    # record used mislabeled detection parameters
        self.results['analysis_parameters']['mislabeled_aspect_ratio'] = float(aspect_ratio_thresh)
        self.results['analysis_parameters']['mislabeled_tiny_norm'] = float(tiny_area_norm)
        self.results['analysis_parameters']['mislabeled_image_area'] = float(image_area)

        extreme_ar = 0
        tiny = 0
        outside = 0
        examples = []
        extreme_images = set()
        save_dir = Path(self.save_extreme_dir) if self.save_extreme_dir else None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        for _, item in self._iter_items():
            for lab in self._filter_labels(item.get('labels', [])):
                box = lab.get('box2d')
                if not box:
                    continue
                w = box.get('x2', 0.0) - box.get('x1', 0.0)
                h = box.get('y2', 0.0) - box.get('y1', 0.0)
                # skip degenerate boxes
                if w <= 0 or h <= 0:
                    continue
                ar = max(w / h, h / w)
                area = w * h
                if ar > aspect_ratio_thresh:
                    extreme_ar += 1
                    extreme_images.add(item.get('name'))
                    if len(examples) < self.max_examples_list:
                        examples.append({'img': item.get('name'), 'type': 'extreme_ar', 'ar': ar})
                if (area / image_area) < tiny_area_norm:
                    tiny += 1
                    # also mark this image so we can save an annotated example (tiny boxes)
                    extreme_images.add(item.get('name'))
                    if len(examples) < self.max_examples_list:
                        examples.append({'img': item.get('name'), 'type': 'tiny', 'area_norm': area / image_area})
                # outside bounds: negative coords or coords greater than image assumed dims
                # outside bounds: negative coords or coords greater than assumed dims (1280x720)
                if box.get('x1', 0) < 0 or box.get('y1', 0) < 0 or box.get('x2', 0) > 1280 or box.get('y2', 0) > 720:
                    outside += 1
                    if len(examples) < self.max_examples_list:
                        examples.append({'img': item.get('name'), 'type': 'outside', 'box': box})

        self.results['mislabeled_boxes'] = {
            'extreme_aspect_ratio': extreme_ar,
            'tiny_boxes': tiny,
            'outside_boxes': outside,
            'examples_counted': min(len(examples), self.max_examples_list),
            'examples': examples
        }
        if save_dir and extreme_images:
            saved = 0
            for img_name in sorted(extreme_images):
                p_train = self.loader.train_images_dir / img_name
                p_val = self.loader.val_images_dir / img_name
                p = p_train if p_train.exists() else (p_val if p_val.exists() else None)
                if p is None:
                    continue
                try:
                    im = Image.open(p).convert('RGB')
                    draw = ImageDraw.Draw(im)
                    # draw all extreme boxes for this image
                    for _, item2 in self._iter_items():
                        if item2.get('name') != img_name:
                            continue
                        for lab2 in self._filter_labels(item2.get('labels', [])):
                            b2 = lab2.get('box2d')
                            if not b2:
                                continue
                            w2 = b2.get('x2', 0) - b2.get('x1', 0)
                            h2 = b2.get('y2', 0) - b2.get('y1', 0)
                            if w2 <= 0 or h2 <= 0:
                                continue
                            ar2 = max(w2 / h2, h2 / w2)
                            if ar2 > aspect_ratio_thresh:
                                # prepare to draw bbox and annotated label
                                # also check if this box is tiny so we can tag it
                                area2 = w2 * h2
                                tags = []
                                if ar2 > aspect_ratio_thresh:
                                    tags.append('aspect_ratio')
                                if (area2 / image_area) < tiny_area_norm:
                                    tags.append('tiny')
                                # draw bbox
                                draw.rectangle([(b2['x1'], b2['y1']), (b2['x2'], b2['y2'])], outline='red', width=3)
                                # draw label text (category) above the box for clarity
                                cat = lab2.get('category', '')
                                try:
                                    # text background for legibility
                                    tag_text = f" ({','.join(tags)})" if tags else ''
                                    text = f"{cat}{tag_text}"
                                    tx, ty = int(b2['x1']), int(max(0, b2['y1'] - 14))
                                    # estimate text size (font not specified; use default)
                                    text_w = max(20, len(text) * 6)
                                    text_h = 12
                                    draw.rectangle([(tx, ty), (tx + text_w, ty + text_h)], fill='black')
                                    draw.text((tx + 1, ty), text, fill='white')
                                except Exception:
                                    # ignore text-draw errors and continue
                                    pass
                            else:
                                # also check tiny-only boxes (may not have triggered ar2)
                                area2 = w2 * h2
                                if (area2 / image_area) < tiny_area_norm:
                                    # draw tiny bbox and label
                                    draw.rectangle([(b2['x1'], b2['y1']), (b2['x2'], b2['y2'])], outline='blue', width=2)
                                    cat = lab2.get('category', '')
                                    try:
                                        text = f"{cat} (tiny)"
                                        tx, ty = int(b2['x1']), int(max(0, b2['y1'] - 14))
                                        text_w = max(20, len(text) * 6)
                                        text_h = 12
                                        draw.rectangle([(tx, ty), (tx + text_w, ty + text_h)], fill='black')
                                        draw.text((tx + 1, ty), text, fill='white')
                                    except Exception:
                                        pass
                    outp = save_dir / img_name
                    im.save(outp)
                    saved += 1
                except Exception:
                    print("couldnt save extreme example for", outp)
                    # ignore individual image save failures
                    continue
            self.results['mislabeled_boxes']['extreme_examples_saved'] = saved

    def class_imbalance_and_examples(self, rare_thresh=100):
        """Identify class counts and rare classes.

        Behavior change: rare classes are determined by the
        fraction `self.rare_pct` if > 0.0. When `self.rare_pct` > 0 the code will
        compute the fraction of total examples and mark classes below that
        fraction as rare.

        The function stores full image lists for rare classes up to
        `self.max_examples_list`. If `self.max_examples_list` is 0 we store only
        counts (no examples).
        """
        c = Counter()
        examples = defaultdict(list)
        total_annotations = 0
        for _, item in self._iter_items():
            name = item.get('name')
            for lab in self._filter_labels(item.get('labels', [])):
                cat = lab.get('category', 'unknown')
                c[cat] += 1
                total_annotations += 1
                if self.max_examples_list > 0 and len(examples[cat]) < self.max_examples_list:
                    examples[cat].append(name)

        rare = {}
        if self.rare_pct and total_annotations > 0:
            # find classes with annotation proportion < rare_pct
            for k, v in c.items():
                if (v / total_annotations) < self.rare_pct:
                    rare[k] = {'count': v, 'examples': examples[k] if self.max_examples_list > 0 else []}

        self.results['class_imbalance'] = {'counts': dict(c), 'rare_classes': rare}

    def split_leakage_check(self):
        """Check for exact filename overlaps between train and val (simple leakage check)."""
        train_names = {item.get('name') for item in self.loader.train}
        val_names = {item.get('name') for item in self.loader.val}
        overlap = train_names.intersection(val_names)
        self.results['split_leakage'] = {'overlap_count': len(overlap), 'examples': list(overlap)[:20]}

    def metadata_consistency(self):
        """Check for unexpected or malformed metadata.

        We look for:
          - unexpected weather values (not in allowed set)
          - attributes field that is missing or not a dict

        """
        bad_weather = []
        bad_attributes = []
        allowed_weathers = {'clear', 'rainy', 'snowy', 'overcast', 'foggy', 'unknown', 'partly cloudy'}
        for _, item in self._iter_items():
            attrs = item.get('attributes')
            if attrs is None:
                bad_attributes.append({'img': item.get('name'), 'issue': 'missing_attributes'})
            elif not isinstance(attrs, dict):
                bad_attributes.append({'img': item.get('name'), 'issue': 'attributes_not_dict', 'value': str(attrs)})
            else:
                w = attrs.get('weather')
                if w and w not in allowed_weathers:
                    bad_weather.append({'img': item.get('name'), 'weather': w})


        self.results['metadata_issues'] = {
            'unexpected_weather_count': len(bad_weather),
            'bad_attributes_count': len(bad_attributes),
            # 'unexpected_weather_values': bad_weather,
        }

    def run_all(self):
        """Run all analysis functions and return results dict."""
        self.compute_dataset_info()
        self.check_file_existence()
        self.classwise_counts()
        self.split_sizes()
        self.bbox_statistics()
        self.cooccurrence()
        self.temporal_and_meta()
        self.annotator_noise()
        self.image_level_counts()
        self.detect_mislabeled()
        self.class_imbalance_and_examples()
        self.split_leakage_check()
        self.metadata_consistency()
        return self.results


def main():
    repo_root = Path(__file__).resolve().parents[2]
    loader = Loader(str(repo_root))
    loader.load()
    out_folder = repo_root / 'analysis_results/'
    out_folder.mkdir(parents=True, exist_ok=True)
    A = Analysis(loader, exclude_classes=['lane', 'drivable area'], max_examples_list=0, rare_pct=0.05, save_extreme_dir=str(out_folder / 'analysis_extreme_boxes'), tiny_area_norm=0.00001, save_annotator_noise_dir=str(out_folder / 'analysis_annotator_noise'))
    res = A.run_all()
    out = Path(out_folder / 'analysis_results.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=2)
    print('Wrote', out)


if __name__ == '__main__':
    main()
