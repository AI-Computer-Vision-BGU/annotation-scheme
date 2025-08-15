import cv2
import os
import subprocess
import shutil
import glob
import numpy as np
from itertools import chain
import json
from contextlib import contextmanager
import sys
import random
import sys
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter
from copy import deepcopy
from tqdm import tqdm
import pickle
import re
import bitarray
import configs
import imageio_ffmpeg
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

#########################################################
#                  Visualization                        #
#########################################################
def draw_menu_panel(img, lines, *,
                    banner_bounds=None,   # (y_top, y_bottom) from your top menu banner
                    start_xy=None,        # (x,y) to override default placement
                    text_color=(255,240,191),  # readable cyan-white (BGR)
                    bg_color=(60,60,60),  # dark grey panel
                    alpha=0.35,
                    thickness=2,
                    font=None,
                    font_scale=None,
                    line_gap=1.35,
                    pad_x=12, pad_t=10, pad_y=10, margin_x=None):
    """
    Draw a small translucent panel with each string in `lines` on a new row.
    Returns: (panel_top, panel_bottom), metrics dict
    """
    H, W = img.shape[:2]
    if font is None:
        font = cv2.FONT_HERSHEY_SIMPLEX
    if font_scale is None:
        font_scale = max(0.7, min(1.4, 0.9 * (W / 1280.0)))
    if margin_x is None:
        margin_x = max(10, int(0.02 * W))

    # Default position: just below the top banner
    if start_xy is None:
        y0 = (banner_bounds[1] if banner_bounds else 0) + pad_y
        x0 = margin_x
    else:
        x0, y0 = start_xy

    # Measure longest line + line height
    max_w, line_h = 0, 0
    for ln in lines:
        (tw, th), _ = cv2.getTextSize(ln, font, font_scale, thickness)
        max_w = max(max_w, tw)
        line_h = max(line_h, int(th * 1.2))

    panel_w = min(W - 2*margin_x, max_w + 2*pad_x)
    panel_h = int(pad_t + len(lines) * line_h * line_gap + pad_t)

    # Background
    draw_translucent_panel(img, x0, y0, panel_w, panel_h, color=bg_color, alpha=alpha)

    # Draw text
    x_text = x0 + pad_x
    y = y0 + pad_t + line_h
    for ln in lines:
        cv2.putText(img, ln, (x_text, int(y)), font, font_scale, text_color, thickness, cv2.LINE_AA)
        y += int(line_h * line_gap)

    metrics = {
        "x_left": x_text,
        "line_h": line_h,
        "font_scale": font_scale,
        "thickness": thickness,
        "panel_right": x0 + panel_w,
        "panel_bottom": y0 + panel_h,
    }
    return (y0, y0 + panel_h), metrics


def draw_menu_banner(img, items, frame_idx, total_frames, thickness=2, top=True, alpha=0.35):
    """
    Draw a responsive banner:
      • menu items (wrapped, with exactly 3 spaces between items)
      • a counter line BELOW the menu (alone in its own line)
    """
    h, w = img.shape[:2]
    margin_x = max(10, int(0.02 * w))
    margin_y = max(8,  int(0.015 * h))
    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_base = 0.8
    font_scale = max(0.6, min(1.4, font_base * (w / 1280.0)))
    line_gap   = configs.LINE_GAP

    # --- wrap by items (keep exactly 3 spaces between items)
    sep = "   "
    lines, line = [], ""
    max_width = w - 2 * margin_x
    for item in items:
        trial = line + (sep if line else "") + item
        (tw, _), _ = cv2.getTextSize(trial, font, font_scale, thickness)
        if tw <= max_width or not line:
            line = trial
        else:
            lines.append(line)
            line = item
    if line:
        lines.append(line)

    # --- metrics
    (_, th), _ = cv2.getTextSize("Ag", font, font_scale, thickness)
    line_h   = int(th * 1.2)
    menu_h   = int(len(lines) * line_h * line_gap)
    counter_text = f"Frame Tracker: {frame_idx}/{max(1, total_frames-1)}"   # keep your old convention
    counter_h    = line_h                                    # same line height

    # total banner height = margins + menu lines + counter line
    banner_h = int(2 * margin_y + menu_h + counter_h)

    # --- translucent panel
    y0 = 0 if top else h - banner_h
    draw_translucent_panel(img, 0, y0, w, banner_h, color=configs.COLORS['panel_color'], alpha=alpha)

    # --- draw menu lines
    y = y0 + margin_y + line_h
    for ln in lines:
        cv2.putText(img, ln, (margin_x, int(y)),
                    font, font_scale, configs.COLORS['menu_class'], thickness, cv2.LINE_AA)
        y += int(line_h * line_gap)

    # --- draw counter on its own line (below menu)
    cv2.putText(img, counter_text, (margin_x, int(y)),
                font, font_scale, configs.COLORS['menu_class'], thickness, cv2.LINE_AA)

    return y0, y0 + banner_h


def draw_translucent_panel(img, x, y, w, h, color=(40,40,40), alpha=0.35):
    """Draw a semi-transparent rectangle over (x,y,w,h)."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, dst=img)


def build_classes_list(img, init_class, banner_bounds=None,
                       margin_x=None, pad_y=10):
    mapping = configs.OBJECT_CLASSES[init_class]
    color   = configs.OBJECT_COLORS[init_class]

    H, W = img.shape[:2]
    if margin_x is None:
        margin_x = max(10, int(0.02 * W))

    font = cv2.FONT_HERSHEY_SIMPLEX
    fscale   = max(0.7, min(1.4, 0.9 * (W / 1280.0)))
    thick    = 2
    line_gap = configs.LINE_GAP

    start_y = (banner_bounds[1] if banner_bounds else 0) + pad_y
    title   = f"Category (press [n] to change):   {init_class}"
    lines   = [title] + [f"[{cid}] {cname}" for cid, cname in mapping.items()]

    # Measure
    max_w, line_h = 0, 0
    for ln in lines:
        (tw, th), _ = cv2.getTextSize(ln, font, fscale, thick)
        max_w = max(max_w, tw)
        line_h = max(line_h, int(th * 1.2))

    pad_x, pad_t = 12, 10
    panel_w = min(W - 2*margin_x, max_w + 2*pad_x)
    panel_h = int(pad_t + len(lines) * line_h * line_gap + pad_t)

    # Panel
    draw_translucent_panel(img, margin_x, start_y, panel_w, panel_h, color=configs.COLORS['classes_background'], alpha=0.35)

    # Draw text
    y = start_y + pad_t + line_h
    x_left = margin_x + pad_x
    for i, ln in enumerate(lines):
        if i == 0:
            cv2.putText(img, ln, (margin_x, int(y)),
                        font, fscale, color, thick, cv2.LINE_AA)
        else:
            cv2.putText(img, ln, (x_left, int(y)),
                        font, fscale, color, thick, cv2.LINE_AA)
        y += int(line_h * line_gap)

    # Metrics for precise placement later
    metrics = {
        "x_left": margin_x,
        "y_title": y + pad_t + line_h,                   # baseline of title
        "y_first_item": start_y + pad_t + line_h + int(line_h * line_gap),
        "line_h": line_h, "font_scale": fscale, "thickness": thick
    }
    return (start_y, start_y + panel_h), img, mapping, metrics


def ask_class(img, max_id, prompt_win=None, panel_metrics=None,
              color=(0,255,255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    typed = ""

    # Place hint ABOVE the title line (inside panel top padding)
    if panel_metrics:
        x = panel_metrics["x_left"]
        lh = panel_metrics["line_h"]
        y = panel_metrics["y_title"] - int(0.35 * lh)   # safely above title
        fscale   = panel_metrics["font_scale"]
        thick    = panel_metrics["thickness"]
    else:
        # fallback
        h, w = img.shape[:2]
        x = max(10, int(0.02*w)); y = 120
        fscale = max(0.7, min(1.4, 0.9 * (w/1280.0)))
        thick  = 2

    while True:
        if prompt_win is not None:
            vis = img.copy()
            cv2.putText(vis, f"Class id: {typed or '_'}", (x, int(y)),
                        font, fscale, color, thick, cv2.LINE_AA)
            cv2.imshow(prompt_win, vis)

        key = cv2.waitKey(0) & 0xFF
        if key in (13, 32):                 # Enter/Space
            if typed:
                cid = int(typed)
                if 0 <= cid < max_id:
                    return cid
                else:
                    print('   [!] OUT OF BOUNDRIES - select different class')
                    typed = ""
            typed = ""
            continue
        if key in (27, ord('q')):           # Esc
            return -1
        if key in (ord('n'), ord('N')):     # change group
            return -10
        if key in (8, 127):                 # backspace/delete
            typed = typed[:-1]
        elif ord('0') <= key <= ord('9'):
            typed += chr(key)


def safe_draw_polygons(img, polygons, color, alpha=0.4):
    """
    polygons : list of polygons, where each polygon is a list/ndarray (N, 2)
    """
    if (
        polygons is None
        or (isinstance(polygons, (list, tuple)) and len(polygons) == 0)
        or (isinstance(polygons, np.ndarray) and polygons.size == 0)
        ):
        return

    # Ensure we always iterate over *individual* polygons
    if isinstance(polygons, np.ndarray) and polygons.dtype != object:
        poly_iter = polygons          # homogeneous 3-D array (k, n, 2)
    else:
        poly_iter = list(polygons)    # list or ragged ndarray → list copy

    # --- helper inside the function ---
    def _draw_single(p):
        p = np.asarray(p, dtype=np.int32).reshape(-1, 2)
        if p.shape[0] < 3:
            return                         # skip degenerate polygon
        p[:, 0] = p[:, 0].clip(0, w-1)
        p[:, 1] = p[:, 1].clip(0, h-1)
        cv2.fillPoly(overlay, [p.reshape(-1, 1, 2)], color)

    h, w = img.shape[:2]
    overlay = img.copy()

    for poly in poly_iter:
        # If we still have a “list of polygons”, drill one level deeper
        if isinstance(poly[0][0], (list, tuple, np.ndarray)):
            # poly is something like [poly1, poly2, …]
            for sub in poly:
                _draw_single(sub)
        else:
            _draw_single(poly)

    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, dst=img)
  

def visualize_bbs_from_db(tool_bbs, frame_paths, frame_names, diff):
    #### Visualize for testing::
    for bb_idx, yolo_bb in tool_bbs.items():

        orig_index = int(bb_idx) + diff
        frame = cv2.imread(os.path.join(frame_paths, frame_names[orig_index]))
        h, w = frame.shape[:2]
        if isinstance(yolo_bb[0], list):
            yolo_bb = yolo_bb[0]
        orig_bb = yolo_to_original_bb(yolo_bb, w, h)

        frame_with_bb = draw_bb_from_4p(frame, orig_bb, text=f'{bb_idx} <-> {orig_index}')

        cv2.imshow(f'old bbs', frame_with_bb)
        cv2.setWindowProperty(f'old bbs', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def show_and_destroy_yolo_res(frame_image, box, indices):
    yolo_res = draw_bb_from_4p(frame_image, box)
    cv2.imshow(f'yolo_frame-{indices[0]}', yolo_res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extract_annotations(success_indices,
                        combined_video_segments,
                        format='coco',  # 'coco' or 'yolo'
                          ):
    """
    EDIT -- No need to save each image with bb, only extract bbs as dictionary
    """
    success_indices = list(chain.from_iterable(success_indices))
    results = {obj: {'bb': {}, 'seg': {}} for obj in configs.OBJECT_TO_ANNOTATE.keys()}
    reversed_objects_to_annotate = {v: k for k, v in configs.OBJECT_TO_ANNOTATE.items()}
    for out_frame_idx in success_indices:
        try:
            for obj in results.keys():  # Initialize empty lists for each frame
                results[obj]['bb'][f'{out_frame_idx}'] = []
                results[obj]['seg'][f'{out_frame_idx}'] = []
                results[obj]['bb']['class_name'] = None

            for out_obj_id, out_mask in combined_video_segments[out_frame_idx].items():
                out_mask = np.array(out_mask[0]).astype(np.uint8)
                out_mask = np.stack((out_mask, out_mask, 255 * out_mask), axis=-1)
                out_obj_name = reversed_objects_to_annotate[out_obj_id]
                if out_obj_name in configs.OBJECT_WITH_BB:     # With BB
                    bb_coco, bb_yolo, segs = draw_bb(out_mask, include_bb=True, shape=None)
                    if format == 'coco':
                        results[out_obj_name]['bb'][f'{out_frame_idx}'].append(bb_coco)
                    elif format == 'yolo':
                        results[out_obj_name]['bb'][f'{out_frame_idx}'].append(bb_yolo)
                    results[out_obj_name]['seg'][f'{out_frame_idx}'] = segs

                else:                                          # With only segmentations 
                    _, _, segs = draw_bb(out_mask, include_bb=False, shape=None)
                    results[out_obj_name]['seg'][f'{out_frame_idx}'] = segs

        except Exception as e:
            print(f' > utils/visualzie_annotations() - frame: {out_frame_idx} Error: {e}')
            continue

    return results


def place_window(winname, winnsize=(1280, 720)):
    """
    Place the window at the top-left corner of the screen.
    """
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)       # Create once with normal mode
    cv2.resizeWindow(winname, *winnsize)               # Resize once (you can pick any size)
    cv2.moveWindow(winname, 0, 0)

#########################################################
#                    Arithmetics                        #
#########################################################

def yolo_to_original_bb(yolo_bbox, img_w, img_h):
    x_center, y_center, w, h = yolo_bbox
    x_center = x_center * img_w
    y_center = y_center * img_h
    w = w * img_w
    h = h * img_h

    # Calculate coordinates
    x_min = int(x_center - w / 2)
    y_min = int(y_center - h / 2)
    x_max = int(x_center + w / 2)
    y_max = int(y_center + h / 2)

    return [x_min, y_min, x_max, y_max]


def convert_bb_to_yolo_format(bb, w, h):
    x1, y1, x2, y2 = bb
    combined_w, combined_h = np.abs(x2 - x1), np.abs(y2 - y1)
    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h
    width = combined_w / w
    height = combined_h / h

    return [x_center, y_center, width, height]


def rescale_bbox_x1y1x2y2(bbox, orig_size, new_size):
    """
    Rescale COCO bbox [x_min, y_min, x_max, y_max] from original image size to new image size.
    
    Parameters:
        bbox (list or tuple): [x_min, y_min, x_max, y_max]
        orig_size (tuple): (orig_w, orig_h)
        new_size (tuple): (new_w, new_h)
        
    Returns:
        list: [new_x_min, new_y_min, new_width, new_height]
    """
    x1, y1, x2, y2 = bbox
    orig_w, orig_h = orig_size
    new_w, new_h = new_size

    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    new_x1 = x1 * scale_x
    new_y1 = y1 * scale_y
    new_x2 = x2 * scale_x
    new_y2 = y2 * scale_y

    width = new_x2 - new_x1
    height = new_y2 - new_y1

    return [
        new_x1,
        new_y1,
        width,
        height
    ]


def rescale_bbox_x1y1wh(bbox, orig_size, new_size):
    """
    Rescale a COCO-format bounding box [x, y, w, h] to a new image size.
    
    :param bbox: list or tuple [x1, y1, w, h]
    :param orig_size: (orig_width, orig_height)
    :param new_size: (new_width, new_height)
    :return: Rescaled bbox [x1', y1', w', h']
    """
    x1, y1, w, h = bbox
    orig_w, orig_h = orig_size
    new_w, new_h = new_size

    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    new_x1 = x1 * scale_x
    new_y1 = y1 * scale_y
    new_w = w * scale_x
    new_h = h * scale_y

    return [new_x1, new_y1, new_w, new_h]


def rescale_polygon(poly, orig_size, new_size):
    """
    Rescale a polygon from original image size to new image size.
    @param poly: List of points in the polygon, each point is a tuple (x, y).
    @param orig_size: Tuple (orig_width, orig_height) of the original image size.
    @param new_size: Tuple (new_width, new_height) of the new image size.
    @return: List of points in the rescaled polygon.
    """
    W0, H0 = orig_size
    W1, H1 = new_size
    sx, sy = W1 / W0, H1 / H0

    def _is_point_like(item):
        """True if item is a scalar or a 1-D array/seq of length 2."""
        try:
            return len(item) == 2 and np.isscalar(item[0])
        except TypeError:
            return False

    def _rescale(obj):
        # Obj is a polygon (list of points) when its first element is a point
        if _is_point_like(obj[0]):
            arr = np.asarray(obj, dtype=np.float32)
            arr *= np.array([sx, sy], np.float32)
            return arr.tolist()
        else:
            # Otherwise it is a list / ndarray of polygons – recurse
            return [_rescale(sub) for sub in obj]

    return _rescale(poly)


def draw_bounding_box(image, coord1, coord2, color=(0, 255, 0), thickness=2):
    # Ensure coordinates are integers
    x1, y1 = map(int, coord1)
    x2, y2 = map(int, coord2)

    # Draw the rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image


def draw_bb_from_4p(img, box, text=None, tool_label=None):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    # Draw circles at each corner and display coordinates
    for i, (x, y) in enumerate(corners):
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)  # Draw circle
        cv2.putText(img, f"({x}, {y})", (x - 20, y - 10) if i < 2 else (x - 20, y + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), 2)
    if text is not None:
        label_x = (x2 + x1) // 2  # middle of bb width
        label_y = y1 - 5 if y1 - 5 > 0 else 0 + 5
        cv2.putText(img, text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img


def draw_bb(out_mask, include_bb=True, shape=None):
    MIN_AREA = 100
    out_mask = out_mask[:, :, 0]
    orig_h, orig_w = out_mask.shape
    # Apply morphological closing to fill small gaps
    kernel = np.ones((5, 5), np.uint8)
    out_mask = cv2.morphologyEx(out_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the cleaned mask
    contours, _ = cv2.findContours(out_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segs = []
    if contours:
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = float('-inf'), float('-inf')

        for contour in contours:
            if cv2.contourArea(contour) < MIN_AREA:
                continue                   

            segs.append(contour.squeeze(1).tolist())  # Convert contour to list of points
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        
        if not include_bb:
            return _, _, segs
        combined_x, combined_y = x_min, y_min
        combined_w, combined_h = x_max - x_min, y_max - y_min
        
        bb = [combined_x, combined_y, combined_x+combined_w, combined_y+combined_h]  # [x1, y1, x2, y2] top-left, bottom-down
        # convert to yolov8 bb format:
        x_center = ((x_min + x_max) / 2) / orig_w
        y_center = ((y_min + y_max) / 2) / orig_h
        width = combined_w / orig_w
        height = combined_h / orig_h
        bb_yolo = [x_center, y_center, width, height]
        if shape is not None:
            bb_coco = rescale_bbox_x1y1x2y2(bb, (orig_w, orig_h), shape)
        else:
            bb_coco = [combined_x, combined_y, combined_w, combined_h]
            
        mask_with_combined_bbox = cv2.cvtColor(out_mask, cv2.COLOR_GRAY2BGR)
        draw_bb_from_4p(mask_with_combined_bbox,
                        [combined_x, combined_y, combined_x + combined_w, combined_y + combined_h])
        return bb_coco, bb_yolo, segs
    return [], [], []

#########################################################
#          Coco Annotations Functionallity              #
#########################################################
def save_open_video_names_as_pickles(set_, path="done_video_names.pkl", op='save'):
    """
    Save or load video names from a pickle file so that we can keep track of which videos have been processed.
    @param set_: Set of video names to save.
    @param path: Path to the pickle file.
    @param op: Operation to perform ('save' or 'open').
    """
    # Save to file
    if op == 'save':
        with open(path, "wb") as f:
            pickle.dump(set_, f, protocol=pickle.HIGHEST_PROTOCOL)
        return None
    
    if op == 'open':
        try:
            with open(path, "rb") as f:
                loaded_set = pickle.load(f)

            return loaded_set
        
        except FileNotFoundError as e:
            print(' File not found: {path} - {e}')
            return {}


def load_coco_annotations(json_path):
    with open(json_path, "r") as f:
        coco = json.load(f)
    return coco


def save_coco_annotations(coco, json_path):
    with open(json_path, "w") as f:
        json.dump(coco, f, indent=2)


def get_image_by_id(coco, image_id):
    for img in coco["images"]:
        if img["id"] == image_id:
            return img
    return None


def get_annotations_for_image(coco, image_id):
    return [ann for ann in coco["annotations"] if ann["image_id"] == image_id]


def draw_cocoBB_from_annotations(img_np, annotations, category_id_to_name, color=(0, 255, 0), orig_width=224, orig_height=224, target_size=(640, 480)):
    for ann in annotations:
        x, y, w, h = ann['bbox']
        scale_x = target_size[0] / orig_width
        scale_y = target_size[1] / orig_height
        x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        class_name = category_id_to_name.get(ann['category_id'], "Unknown")
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_np, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img_np


def draw_cocoBB_from_dict(img_np, annotations, class_name, color=(0, 255, 0), orig_width=224, orig_height=224, target_size=(640, 480)):
    for ann in annotations:
        if ann is not None and len(ann) == 4:  # Ensure ann is a valid bbox
            x, y, w, h = ann
            scale_x = target_size[0] / orig_width
            scale_y = target_size[1] / orig_height
            x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            # class_name = category_id_to_name.get(ann['category_id'], "Unknown")
            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_np, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            # print(f"Invalid annotation found: {ann}. Skipping drawing.")
            continue
    return img_np


def delete_annotation(coco_data, id_to_remove):
    """
    Delete an annotation by its ID from the COCO dataset.

    @param coco_data: dict loaded from COCO annotation JSON
    @param id_to_remove: int, ID of the annotation to delete
    """
    image_id = None
    new_images = []
    for img in coco_data["images"]:
        if img["id"] == id_to_remove:
            image_id = img["id"]
        else:
            new_images.append(img)
    coco_data["images"] = new_images

    # Remove all annotations with that image_id
    if image_id is not None:
        coco_data["annotations"] = [ann for ann in coco_data["annotations"] if ann["image_id"] != image_id]

    return coco_data


def update_annotation_bbox(coco_data, image_id, new_bboxes):
    """
    Replace all bounding boxes for a given image_id with new ones.

    @param coco_data: dict loaded from COCO annotation JSON
    @param image_id: int, ID of the image to update
    @param new_bboxes: list of dicts, each containing 'category_id' and 'bbox

    """
    # Remove old bboxes for this image
    coco_data["annotations"] = [ann for ann in coco_data["annotations"] if ann["image_id"] != image_id]

    # Add new bboxes
    next_ann_id = max([ann["id"] for ann in coco_data["annotations"]], default=0) + 1
    for new_ann in new_bboxes:
        coco_data["annotations"].append({
            "id": next_ann_id,
            "image_id": image_id,
            "category_id": new_ann["category_id"],
            "bbox": new_ann["bbox"],
            "area": new_ann["bbox"][2] * new_ann["bbox"][3],
            "iscrowd": 0
        })
        next_ann_id += 1


def merge_images_to_coco(args, new_images, new_annotations, save_as=None):
    """
    Merge new_images, new_annotations into an in-memory COCO dict
    
    Pay attention, the file_name as follows -- vidname_numericNumber.jpg

    @param args: An object containing the COCO dataset path and annotations.
    @param new_images: List of new images to add.
    @param new_annotations: List of new annotations to add.
    """
    # --------------------------------------- current maxima
    img_ids  = [img["id"] for img in args.annotations["images"]]
    ann_ids  = [ann["id"] for ann in args.annotations["annotations"]]
    max_img_id  = max(img_ids,  default=0)
    max_ann_id  = max(ann_ids,  default=0)

    # last numeric file name assuming file name as follows -- vidname_numericNumber.jpg
    def numeric_part(fname):
        return int(fname.split('_')[-1].split('.')[0]) if fname.split('_')[-1].split('.')[0].isdigit() else None

    numeric_names = [numeric_part(img["file_name"]) for img in args.annotations["images"]]
    numeric_names = [n for n in numeric_names if n is not None]
    max_name_num  = max(numeric_names, default=0)
    pad = 5#len(os.path.splitext(args.annotations["images"][-1]["file_name"])[0]) if args.annotations.get("images") else 0

    # --------------------------------------- id / name remapping
    id_map = {}
    fixed_images = []
    for offset, img in enumerate(new_images, start=1):
        new_id = max_img_id + offset
        id_map[img["id"]] = new_id
        
        old_fname = img["file_name"]
        video_name = old_fname.split('_')[0]
        seq_num  = max_name_num + offset
        ext      = os.path.splitext(img["file_name"])[1] or ".jpg"
        new_name = f"{video_name}_{seq_num:0{pad}d}{ext}"
        # --- rename file on disk ----------------------------------------
        src = os.path.join(args.coco_data, 'images', old_fname)
        dst = os.path.join(args.coco_data, 'images', new_name)
        if os.path.exists(src):
            shutil.move(src, dst)          # rename / move
        else:
            print(f" File not found: {src} (skipped renaming)")

        img_fix = deepcopy(img)
        img_fix["id"] = new_id
        img_fix["file_name"] = new_name
        fixed_images.append(img_fix)

    fixed_annos = []
    for offset, ann in enumerate(new_annotations, start=1):
        ann_fix = deepcopy(ann)
        ann_fix["id"] = max_ann_id + offset
        ann_fix["image_id"] = id_map[ann["image_id"]]
        fixed_annos.append(ann_fix)

    # --------------------------------------- extend
    args.annotations.setdefault("images", []).extend(fixed_images)
    args.annotations.setdefault("annotations", []).extend(fixed_annos)

    with open(os.path.join(args.coco_data, 'annotations.json'), "w", encoding="utf-8") as f:
        json.dump(args.annotations, f, indent=2, ensure_ascii=False)


def save_as_coco_format(image, frame_idx, bb_coco, args):
    """
    Save a single image and its corresponding bounding box in COCO format.
    @param image: The image to save.
    @param frame_idx: The index of the frame (used for naming).
    @param bb_coco: The bounding box in COCO format [x_min, y_min, width, height].
    @param args: An object containing the COCO dataset path and annotations.
    """
    img_id = f"{args.vid_name}_{frame_idx:05d}"
    cv2.imwrite(os.path.join(args.coco_data, 'images', f'{img_id}.jpg'), cv2.resize(image, dsize=args.new_shape, interpolation=cv2.INTER_LINEAR))
    id_ = len(args.annotations['images'])
    args.annotations['images'].append({
                        'file_name': f'{img_id}.jpg',
                        'height': args.new_shape[1],
                        'width': args.new_shape[0],
                        'id': id_
                    })

    args.annotations['annotations'].append({
        'image_id': id_,
        'category_id': args.curr_tool_id,
        'bbox': bb_coco,
        'area': bb_coco[2] * bb_coco[3],
        'iscrowd': 0,
        'id': id_
    })

    # update on real time:
    with open(os.path.join(args.coco_data, "annotations.json"), "w") as f:
        json.dump(args.annotations, f, indent=4)


#########################################################
#                   Scheme Helpers                      #
#########################################################
def check_existence(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    return True


def video_to_frames(video_path, output_path):
    """
    Extract all frames from the given video and save them as .jpg images
    in the specified output directory, with filenames like 00001.jpg, 00002.jpg, etc.

    :param video_path: Full path to the video file.
    :param output_path: Directory path where extracted frames will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Build the ffmpeg command
    command = [
        "ffmpeg",
        "-loglevel", "error",  # Suppress all non-error messages
        "-i", video_path,  # Input video file
        "-q:v", "2",  # Set output quality; lower is better (range is roughly 2–31)
        "-start_number", "0",  # Start naming frames from 00000, 00001, ...
        f"{output_path}/%05d.jpg"  # Output filename pattern
    ]

    # Run the command
    subprocess.run(command, check=True)


def video_to_frames_slow(video_path, output_path):
    """
    Extract all frames from the given video and save them as .jpg images
    in the specified output directory, with filenames like 00000.jpg, 00001.jpg, etc.
    
    Uses OpenCV for fast, native decoding with tqdm progress bar.

    :param video_path: Full path to the video file.
    :param output_path: Directory path where extracted frames will be saved.
    """
    os.makedirs(output_path, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Extracting frames")

    frame_idx = 0
    success, frame = cap.read()
    while success:
        filename = os.path.join(output_path, f"{frame_idx:05d}.jpg")
        # if frame_idx % 3 == 0:
        cv2.imwrite(filename, frame)
        frame_idx += 1
        pbar.update(1)
        success, frame = cap.read()

    pbar.close()
    cap.release()


def reset_working_dir(directory, delete_=False):
    if delete_:
        shutil.rmtree(directory)
        return
    jpg_files = glob.glob(os.path.join(directory, "*.jpg"))
    # Loop through and delete each .jpg file
    for file_path in jpg_files:
        os.remove(file_path)


def initial_working_dir(output_frames_path, frame_names, current_working_file, indcies):
    reset_working_dir(current_working_file)

    for idx in indcies:
        shutil.copy(os.path.join(output_frames_path, frame_names[idx]), current_working_file)

    return np.array(frame_names)[indcies]


def find_range(curr_i, n, bit_state=None, range_step=50, total_indices=20):
    """
    every range space should contain $range_step$ images per time except last range that contains > 50 < 100
    @param bit_state: bit trace of the done frame. i.e., 00011111 (the first three frames are not annotated) 
    
    """
    # print(f' ------ {bit_state}')
    m = re.search(r'0+', bit_state.to01())
    if m.start() == curr_i and m.end() - m.start() < range_step:
        return range(m.start(), m.end())
    
    if m.end() - m.start() > 0 and curr_i <= m.end() and m.end() <= range_step:
        return range(curr_i, m.end())

    else:
        if curr_i + range_step < n:
            return range(curr_i,
                        curr_i + range_step)  # select_random_indices(range(curr_i, curr_i + range_step), total_indices)
        return range(curr_i, n)  # select_random_indices(range(curr_i, n), total_indices)


def ensure_bitset(state, video_name, n_frames):
    """Return bitarray for video, creating or expanding it if necessary."""
    if video_name not in state:
        state[video_name] = bitarray.bitarray(n_frames)
        state[video_name].setall(False)
    elif len(state[video_name]) < n_frames:          # video re-encoded?
        extra = n_frames - len(state[video_name])
        state[video_name].extend([False]*extra)
    return state[video_name]


def select_random_indices(indices, total=20):
    """
    Selects 20 random indices from the given list, ensuring the first and last indices are included.
    :param indices: A list of indices (all >= 50).
    :param total: total indices to select
    :return list: A list of $total selected indices.
    """
    if len(indices) < total:
        return indices

    first, last = indices[0], indices[-1]  # Ensure first and last indices are included
    middle_indices = indices[1:-1]  # Exclude first and last for random selection
    selected_middle = random.sample(middle_indices, total - 2)
    selected_indices = [first] + selected_middle + [last]

    return sorted(selected_indices)


def get_k_frames(frame_names, k=100, dist=None):
    # choose only k frames to annotate (no need to annotate all frames)
    if len(frame_names) > k:
        if dist:
            indices = np.arange(len(frame_names))
            mu = (len(frame_names) - 1) / 2.0  # Center of the array
            sigma = len(frame_names) / 4.0  # Adjust spread based on array length
            probs = np.exp(-(indices - mu) ** 2 / (2 * sigma ** 2))
            probs /= probs.sum()  # Normalize to sum to 1
            # Randomly select indices without replacement
            selected_indices = np.random.choice(indices, size=k, replace=False, p=probs).tolist()
            selected_indices.sort()

            return np.array(frame_names)[selected_indices].tolist()
        else:
            diff = len(frame_names) - k
            frame_names = frame_names[diff // 2: k + diff // 2]

    return frame_names


def prompt_question(ask_user, query='Next Video? [y/N]: '):
    if ask_user:
        query = ' Q) ' + query
        next_vid = input(query).lower()
        if 'y' in next_vid:  # exit this function
            return True
    return False


@contextmanager
def suppress_output():
    # Open null files for stdout and stderr
    with open(os.devnull, 'w') as devnull:
        # Save the original stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            # Redirect stdout and stderr to devnull
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            # Restore stdout and stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def no_point_selected_by_user(point_prompts=None):

    if point_prompts == None:
        return True

    for obj_id, streams in point_prompts.items():
        for stream in streams.keys():
            if 'inc' in stream or 'exc' in stream:
                if streams[stream] != []:
                    return False
            else:
                if streams[stream] is not None:
                    return False
    
    return True


def _yes_no(prompt, default=False):
    """
    Ask a yes/no question on stdin and return True/False.
    Empty input returns the default.
    """
    while True:
        ans = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
        if not ans:
            return default
        if 'y' in ans.lower():
            return True
        if 'n' in ans.lower():
            return False
        print("  ➜ Please answer with y / n")
    
#########################################################
#               FILES Functionality                  #
#########################################################
def write_to_json(json_path, data):
    with open(json_path, 'w') as jf:
        json.dump(data, jf, indent=4)


def read_json(json_path):
    if os.path.exists(json_path):
        with open(json_path, "r") as file:
            data = json.load(file)  # Load JSON as dictionary
        return data
    return {}


def get_tacker_paths(tracker):
    return set(tracker.keys())


def input_with_path_completion(message="Enter path: "):
    completer = PathCompleter(only_directories=False, expanduser=True)
    return prompt(message, completer=completer)


