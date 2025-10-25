"""Visualize analysis_results.json with interactive tabs.

Run with:
    streamlit run tasks/3_eval_viz/visualize_analysis.py

The app expects an `analysis_results.json` (default: analysis_results/analysis_results.json)
and optional image folders under analysis_results/analysis_annotator_noise and
analysis_results/analysis_extreme_boxes which are created by the analysis script.
"""
from pathlib import Path
import json
import streamlit as st
import pandas as pd
from PIL import Image


def load_results(path: Path):
    if not path.exists():
        st.error(f"Analysis file not found: {path}")
        return None
    with open(path, 'r') as f:
        return json.load(f)


def show_class_counts(tab, data):
    tab.subheader('Class counts')
    s = pd.Series(data).sort_values(ascending=False)
    tab.bar_chart(s)
    if len(s) > 30:
        tab.write('Top 30 classes')
        tab.bar_chart(s.head(30))


def show_bbox_stats(tab, data):
    tab.subheader('BBox stats (per-class)')
    df = pd.DataFrame.from_dict(data, orient='index')
    tab.dataframe(df)
    if not df.empty:
        tab.line_chart(df[['mean', 'median']])


def show_cooccurrence(tab, data):
    tab.subheader('Co-occurrence')
    pair_counts = data.get('pair_counts', {})
    if not pair_counts:
        tab.write('No cooccurrence data')
        return
    rows = []
    for k, v in pair_counts.items():
        a, b = k.split('|')
        rows.append({'A': a, 'B': b, 'count': v})
    df = pd.DataFrame(rows)
    tab.dataframe(df.sort_values('count', ascending=False).head(200))


def show_temporal(tab, data):
    tab.subheader('Temporal & metadata')
    meta = data.get('meta_counts', {})
    for k, v in meta.items():
        tab.write(f"**{k}**")
        tab.bar_chart(pd.Series(v))


def show_image_list_and_viewer(tab, images_list, images_dir: Path, title: str = 'Images'):
    tab.subheader(title)
    if not images_list:
        tab.write('No images listed')
        return
    sel = tab.selectbox('Select image', images_list)
    if sel:
        p = images_dir / sel
        if p.exists():
            try:
                im = Image.open(p)
                tab.image(im, caption=sel, use_column_width=True)
            except Exception as e:
                tab.error(f'Failed to open {p}: {e}')
        else:
            tab.warning(f'Image not found at expected path: {p}')


def main():
    st.set_page_config(layout='wide', page_title='Analysis Visualizer')
    st.title('Analysis results visualizer')

    repo_root = Path(__file__).resolve().parents[2]
    default_json = repo_root / 'analysis_results' / 'analysis_results.json'
    json_path = Path(st.text_input('Path to analysis_results.json', str(default_json)))

    data = load_results(json_path)
    if data is None:
        return

    tabs = st.tabs(list(data.keys()))

    for name, tab in zip(list(data.keys()), tabs):
        with tab:
            tab.write(f'### {name}')
            val = data.get(name)
            # show global analysis parameters when available
            if name == 'dataset_info' and data.get('analysis_parameters'):
                params = data.get('analysis_parameters', {})
                tab.markdown('**Analysis parameters**')
                tab.write(params)
            if name == 'class_counts' and isinstance(val, dict):
                show_class_counts(tab, val)
            elif name == 'file_existence' and isinstance(val, dict):
                tab.subheader('File existence')
                missing = val.get('missing_per_split', {})
                total = val.get('total_per_split', {})
                tab.bar_chart(pd.Series(missing))
                tab.write('Total per split')
                tab.bar_chart(pd.Series(total))
            elif name == 'bbox_stats' and isinstance(val, dict):
                show_bbox_stats(tab, val)
            elif name == 'cooccurrence' and isinstance(val, dict):
                show_cooccurrence(tab, val)
            elif name == 'split_sizes' and isinstance(val, dict):
                tab.subheader('Split sizes')
                tab.bar_chart(pd.Series({k: v for k, v in val.items() if isinstance(v, (int, float))}))
            elif name == 'temporal_metadata' and isinstance(val, dict):
                show_temporal(tab, val)
            elif name == 'annotator_noise' and isinstance(val, dict):
                tab.write(val)
                # display IoU threshold used (if available)
                iou = data.get('analysis_parameters', {}).get('annotator_noise_iou')
                if iou is not None:
                    tab.info(f'Overlap (IoU) threshold used: {iou}')
                # show saved images (if any)
                images_dir = json_path.parent / 'analysis_annotator_noise'
                images = []
                if images_dir.exists():
                    images = sorted([p.name for p in images_dir.iterdir() if p.suffix.lower() in ('.jpg', '.png', '.jpeg')])
                show_image_list_and_viewer(tab, images, images_dir, title='Annotator noise images')
            elif name == 'mislabeled_boxes' and isinstance(val, dict):
                tab.write(val)
                # show parameters used for mislabeled detection
                ar = data.get('analysis_parameters', {}).get('mislabeled_aspect_ratio')
                tiny = data.get('analysis_parameters', {}).get('mislabeled_tiny_norm')
                if ar is not None and tiny is not None:
                    tab.info(f'Mislabeled thresholds: aspect_ratio > {ar}, tiny area_norm < {tiny}')
                images_dir = json_path.parent / 'analysis_extreme_boxes'
                images = []
                if images_dir.exists():
                    images = sorted([p.name for p in images_dir.iterdir() if p.suffix.lower() in ('.jpg', '.png', '.jpeg')])
                show_image_list_and_viewer(tab, images, images_dir, title='Extreme / tiny box images')
            elif name == 'image_level_counts' and isinstance(val, dict):
                hist = val.get('obj_count_histogram', {})
                if hist:
                    tab.write('Histogram: distribution of object counts per image (x = #objects, y = #images)')
                    tab.bar_chart(pd.Series({int(k): v for k, v in hist.items()}))
                else:
                    tab.write(val)
            else:
                # generic dump for other keys
                try:
                    tab.json(val)
                except Exception:
                    tab.write(val)


if __name__ == '__main__':
    main()
