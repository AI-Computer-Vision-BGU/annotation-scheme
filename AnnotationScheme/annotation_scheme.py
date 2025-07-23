"""
This scheme is responsible for the annotation scheme mentioned in our Maintenance Datasets paper
it works as follows:
 - get video input
 - convert the input to frames
     - First, use YOLOv8 to predict the bb, is detected, goto SAM2
     - IF not, the first frame will pop up to choose the initial bb manually
 - SAM2 is then applied to continue to detect the bb based on the initial bb we choose
 - repeat every n frames

 
 UPDATES:
  13/07/2025 -- added annotations for hand segmentations by asking user in real-time to choose points for hands or tool (or both)

# TODO:

SEARCH WORDS IN THIS FILE FOR EDITTING/FIXING:
1. MOVE
2. EDIT
3. REMOVE
"""
from PIL import Image
from pathlib import Path
import os
import sys
sys.path.append(os.getcwd())
from AnnotationScheme.utils.sam2_loader import SAM2_ROOT  # noqa: F401
from segmentanything.sam2.build_sam import build_sam2
from segmentanything.sam2.build_sam import (build_sam2_video_predictor)
from segmentanything.sam2.sam2_image_predictor import SAM2ImagePredictor
from utils import utils
import configs
import json
import  torch
from types import SimpleNamespace
import cv2
import numpy as np
import argparse
import warnings
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device: {device}')

# ---------------------------------------------------------- Admin
parser = argparse.ArgumentParser(
        description="Semi-automatic video annotation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--weights", type=str,
                        default='t', help="choose sam2 weights [tiny (t), small (s), base_plus (b), large (l)] default- t")
parser.add_argument("--new_shape", nargs=2, type=int, metavar=("W", "H"),
                        default=(640, 360), help="Resize frames before export")

parser.add_argument("--repeat", type=int,
                        default=20, help="Resize frames before export")

args_parser = parser.parse_args()

# INITIALIZE MODELS AND GLOBAL VARIABLES
sam_video_predictor = build_sam2_video_predictor(configs.config_weights_mapping[args_parser.weights]['configs'],
                                                 configs.config_weights_mapping[args_parser.weights]['weights'], device=device)
sam_predictor = SAM2ImagePredictor(build_sam2(configs.config_weights_mapping[args_parser.weights]['configs'],
                                              configs.config_weights_mapping[args_parser.weights]['weights'], device=device))
yolo_failed = False
# right click for points to segment, left point for excluded segmentation
segment_by_points = True
# this is for the tool hand segmentation. 1- tool, 2- hand
active_prompt = '1'
tool_hand_segmentation = {
    '1': {'include_points':[], 'exclude_points':[], 'start_point': None, 'end_point': None},
    '2': {'include_points':[], 'exclude_points':[], 'start_point': None, 'end_point': None}
}
# these variables for manually annotator GUI
next_video = False
is_drawing = False
image_copy = None   # To reset the image during dynamic drawing
clean_state = None  # the clean image state for the drawing
winname = f'{-1}'   # before starting...
tracker_path = 'tracker.json' # track the database paths (add for each path if manually or yolo)
bb_editting_color = (0, 0, 255)
cursor_pos = None            # (x, y) or None



#########################################################
#                     Helpers                           #
#########################################################
def mouse_callback(event, x, y, flags, param):
    global start_point, end_point, is_drawing, image_copy
    global clean_state, segment_by_points, active_prompt, cursor_pos

    # ---------- live lines ------------------------------------
    temp_image = image_copy.copy()
    cv2.line(temp_image, (x, 0), (x, temp_image.shape[0]), configs.COLORS['cross_line'], 1)
    cv2.line(temp_image, (0, y), (temp_image.shape[1], y), configs.COLORS['cross_line'], 1)

    if event in (cv2.EVENT_MOUSEMOVE,
                 cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
        cursor_pos = (x, y)

    # ---------- LEFT  button down  -----------------------------------
    if event == cv2.EVENT_LBUTTONDOWN:
        if segment_by_points:
            tool_hand_segmentation[active_prompt]['include_points'].append([x, y])
            cv2.circle(image_copy, (x, y), 5, param[active_prompt], -1)
        else:
            image_copy = clean_state.copy()
            tool_hand_segmentation[active_prompt]['start_point'] = [x, y]
            is_drawing  = True

    # ---------- RIGHT button down  -----------------------------------
    elif event == cv2.EVENT_RBUTTONDOWN:
        if segment_by_points:
            tool_hand_segmentation[active_prompt]['exclude_points'].append([x, y])
            cv2.circle(image_copy, (x, y), 5, configs.COLORS['exclude'], -1)
        else:
            image_copy = clean_state.copy()
            tool_hand_segmentation[active_prompt]['start_point'] = [x, y]
            is_drawing  = True

    # ---------- mouse move (rectangle preview) ------------------------
    elif event == cv2.EVENT_MOUSEMOVE and is_drawing and not segment_by_points:
        tool_hand_segmentation[active_prompt]['end_point'] = [x, y]
        cv2.rectangle(temp_image, tool_hand_segmentation[active_prompt]['start_point'],
                       tool_hand_segmentation[active_prompt]['end_point'], (0, 255, 0), 2)

    # ---------- button up  -------------------------------------------
    elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
        is_drawing = False
        if not segment_by_points:
            tool_hand_segmentation[active_prompt]['end_point'] = [x, y]
            cv2.rectangle(image_copy, tool_hand_segmentation[active_prompt]['start_point'],
                           tool_hand_segmentation[active_prompt]['end_point'],
                          bb_editting_color, 2)

    # ---------- refresh window ---------------------------------------
    cv2.imshow(winname, temp_image)


def add_manually_bb(frame_image, mode='regular'):
    global winname, image_copy, clean_state, start_point, end_point, active_prompt, cursor_pos, segment_by_points

    hand_tool_color = {
        '1': (255, 255, 0),  # tool - blue
        '2': (255, 0, 255),  # hand - pink
    }
    # Ask user to choose hand or tool
    active_prompt = '1'  # tool by default -- BB and segmentation 
    image_copy = frame_image.copy()  # Create a copy for resetting during drawing
    clean_state = frame_image.copy()
    utils.place_window(winname)
    cv2.imshow(winname, image_copy)
    cv2.setMouseCallback(winname, mouse_callback, hand_tool_color)

    while True:
        frame_vis = image_copy.copy()

        # ---------- live cross-hair and BB ----------
        if cursor_pos is not None:
            cx, cy = cursor_pos
            cv2.line(frame_vis, (cx, 0),  (cx, frame_vis.shape[0]), configs.COLORS['cross_line'], 1)
            cv2.line(frame_vis, (0,  cy), (frame_vis.shape[1], cy), configs.COLORS['cross_line'], 1)
        
        if tool_hand_segmentation[active_prompt]['start_point'] is not None:
            if tool_hand_segmentation[active_prompt]['end_point'] is not None:
                cv2.rectangle(frame_vis, tuple(tool_hand_segmentation[active_prompt]['start_point']), tuple(tool_hand_segmentation[active_prompt]['end_point']), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame_vis, tuple(tool_hand_segmentation[active_prompt]['start_point']), tuple(cursor_pos), (0, 255, 0), 2)

        # --- overlay current mode on a copy so user sees it live
        mode_text = "TOOL" if active_prompt == '1' else "HAND"
        edit_mode_text = "BB" if not segment_by_points else "Points"
        cv2.putText(frame_vis, f"Mode (1- TOOL, 2- HAND):   {mode_text}", (configs.x_offset, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_vis, f"Drawing (press b to change):  {edit_mode_text}", (configs.x_offset, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(winname, frame_vis)

        key = cv2.waitKey(10) & 0xFF   # 20 ms tick → smooth UI
        if key in (13, 32):            # Enter or Space → finish
            break
        elif key == ord('1'):
            active_prompt = '1'
        elif key == ord('2'):
            active_prompt = '2'
        elif key == ord('b'):
            segment_by_points = not segment_by_points
        


def reset_globals():
    global next_video, tool_hand_segmentation

    next_video = False
    for obj_id in tool_hand_segmentation.keys():
        for stream in tool_hand_segmentation[obj_id].keys():
            if 'inc' in stream or 'exc' in stream:
                tool_hand_segmentation[obj_id][stream] = []
            else:
                tool_hand_segmentation[obj_id][stream] = None


def run_preview(preview_path, tool_bbs, tool_segs=None, hand_segs=None, new_shape=(640, 360), milli=0, save_vis=False):
    """
    Run the result of the bounding boxes with a choice to edit
    :param preview_path: path to MP4 bb r esult
    :param tool_bbs: mapping of frame index to bounding box coordinates
    :return:
    """
    global segment_by_points, winname
    if save_vis:
        if not os.path.exists(os.path.join(preview_path, 'vis')):
            os.makedirs(os.path.join(preview_path, 'vis'))
    images = []
    annotations = []
    img_id = len(os.listdir(os.path.join(args.coco_data, 'images')))
    ann_id = len(annotations)
    frame_idx = 0
    selected_class = -1
    total_frames = len(os.listdir(os.path.join(preview_path, 'frames')))
    winname = f'Preview - {os.path.split(preview_path)[-1]}'#  winname here so the window does not close and open after each frame..

    while frame_idx < total_frames:
        orig_frame_path = os.path.join(preview_path, 'frames', f'{frame_idx:05d}.jpg')
        orig_frame_img = cv2.imread(orig_frame_path)
        clean_frame_img = orig_frame_img.copy()
        to_save_image = orig_frame_img.copy()
        orig_h, orig_w, _ = orig_frame_img.shape
        frame_name = f"{os.path.split(preview_path)[-1]}_{frame_idx:05d}.jpg"
        img_saved = False
        img_id_indicator = False

        bbox = tool_bbs.get(str(frame_idx))  
        # skip if missing OR explicitly marked as "empty"
        if bbox is None or len(bbox) == 0 or (isinstance(bbox, list) and bbox == [None]):
            frame_idx += 1
            continue
        
        orig_frame_img = utils.draw_cocoBB_from_dict(orig_frame_img.copy(), tool_bbs[f'{frame_idx}'], configs.category_id_to_name[selected_class] if selected_class > -1 else "Unknown",
                                                        orig_width=orig_w, orig_height=orig_h,
                                                        target_size=(orig_w, orig_h))

        if f'{frame_idx}' in tool_segs:
            utils.safe_draw_polygons(orig_frame_img, tool_segs[f'{frame_idx}'], color=(0, 255, 0), alpha=0.4)

        if f'{frame_idx}' in hand_segs and hand_segs[f'{frame_idx}']:
            utils.safe_draw_polygons(orig_frame_img, hand_segs[f'{frame_idx}'], color=(0, 0, 255), alpha=0.4)

        vis_image = orig_frame_img.copy()
        # --------------------------------------------------  instructions
        instr = "[a/Enter] Accept   [e] Edit BB   [n] Change Class   [q] Quit BB   [x] Quit Program"
        cv2.putText(orig_frame_img, instr, (configs.x_offset, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(orig_frame_img, f'{frame_idx}/{total_frames-1}', (configs.x_offset, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 255, 0), 3, cv2.LINE_AA)
        
        utils.place_window(winname, winnsize=configs.winnsize)  # Place the window at the top left corner
        cv2.imshow(winname, orig_frame_img)

        key = cv2.waitKey(0) & 0xFF  
        if key in (ord('q'), 27): 
            # TODO -- also, save here the annotation before existing.
            cv2.destroyAllWindows()
            utils.save_coco_annotations(args.annotations, os.path.join(args.coco_data, 'annotations.json'))
            utils.save_open_video_names_as_pickles(args.done_video_names, path=os.path.join(args.coco_data, 'done_video_names.pkl'), op='save')
            return
        
        if key in (ord('x'),):  # Quit the program
            cv2.destroyAllWindows()
            sys.exit()

        if key in (ord('e'),):  # EDIT BB
            reset_globals() 
            # segment_by_points = False
            cv2.putText(clean_frame_img, f'{frame_idx}/{total_frames-1}', (configs.x_offset, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 255, 0), 3, cv2.LINE_AA)
            add_manually_bb(clean_frame_img, mode='review')
            results = sam_prompt_to_polygons(clean_frame_img, eps=1.5, min_area=10)
            if results:
                if "tool" in results:
                    tool_bbs[f'{frame_idx}'] = results["tool"]["bbox"]
                    tool_segs[f'{frame_idx}'] = results["tool"]["segs"]
                    print(f" [*] Updated BB for frame {frame_idx}: (Scaled to the new shape)")
                
                if "hand" in results:
                    hand_segs[f'{frame_idx}'] = results["hand"]["segs"]
                
            else:
                tool_bbs[f"{frame_idx}"] = [[]]  # mark skipped
                tool_segs[f"{frame_idx}"] = [[]]  # mark skipped
                print(f" [*] Skipped frame {frame_idx}")

            cv2.destroyAllWindows()
            continue

            
        elif key == ord('n'):  # Change category in-GUI
            image_with_classes = orig_frame_img.copy()
            y_offset = 165
            class_instr_lines = []
            class_instr = ""
            for cid, cname in configs.category_id_to_name.items():
                class_instr += f"[{cid}] {cname}   "
                # if cid % 3 == 2:
                class_instr_lines.append(class_instr.strip())
                class_instr = ""
            if class_instr:
                class_instr_lines.append(class_instr.strip())
            for line in class_instr_lines:
                cv2.putText(image_with_classes, line, (configs.x_offset, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 12), 2, cv2.LINE_AA)
                y_offset += 50
            cv2.imshow(winname, image_with_classes)
            while True:
                user_choice = utils.ask_class(image_with_classes, max_id=12, prompt_win=winname, x_offset=configs.x_offset, selected_class=selected_class)
                if user_choice < 0 or user_choice >= len(configs.category_id_to_name):
                    print(" [!] Invalid class. Please select a valid category.")
                    continue
                else:
                    if user_choice != selected_class:
                        print(f" [*] Updated category to {configs.category_id_to_name[user_choice]}")
                        selected_class = user_choice
                    frame_idx -= 1 # Go back to the same frame
                    break


        elif key in (ord('a'), 13, 32):  # Accept the current BB
            if selected_class == -1:
                print(" [!] No category selected. Please select a category.")
                # frame_idx -= 1  # Go back to the same frame
                continue
            else:
                if tool_bbs[f"{frame_idx}"]:  # add tool if exist
                    cv2.imwrite(os.path.join(args.coco_data, 'images', frame_name), cv2.resize(to_save_image, dsize=new_shape, interpolation=cv2.INTER_LINEAR))
                    if save_vis:
                        cv2.imwrite(os.path.join(preview_path, 'vis', frame_name), vis_image)
                    img_saved = True
                    bb = utils.rescale_bbox_x1y1wh(tool_bbs[f"{frame_idx}"][0], (orig_w, orig_h), new_shape)  # assuming always one bb in each frame..
                    # print(f" Accepted BB for frame {frame_idx}: img_id: {img_id}- annID: {ann_id} --> {frame_name}")
                    images.append({
                        "id": img_id,
                        "file_name": frame_name,
                        "width": new_shape[0],
                        "height": new_shape[1]
                    })
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": selected_class,  
                        "bbox":bb,
                        "area": bb[-1] * bb[-2],
                        "iscrowd": 0,
                        "tool_segmentation": utils.rescale_polygon(tool_segs[f'{frame_idx}'], (orig_w, orig_h), new_shape),
                        "hand_segmentation": [],
                    })
                    ann_id += 1
                if f'{frame_idx}' in hand_segs and hand_segs[f'{frame_idx}']:  # add hands
                    # print(f" Accepted hand segmentations for frame {frame_idx}")
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 10,  
                        "bbox":[],
                        "area": 0,
                        "iscrowd": 0,
                        "tool_segmentation": [],
                        "hand_segmentation": utils.rescale_polygon(hand_segs[f'{frame_idx}'], (orig_w, orig_h), new_shape)
                    })
                    if not img_saved:
                        cv2.imwrite(os.path.join(args.coco_data, 'images', frame_name), cv2.resize(to_save_image, dsize=new_shape, interpolation=cv2.INTER_LINEAR))
                        img_saved = True
                    ann_id += 1
                    img_id += 1
                    img_id_indicator = True
                    
                else:
                    if not img_saved:
                        # allow the user to see that this bb is indeed -1
                        print(f" [!] Discard BB for frame {frame_idx}: {tool_bbs[f'{frame_idx}']}")
                
                if not img_id_indicator and img_saved:
                    img_id += 1

        # cv2.destroyAllWindows()
        frame_idx += 1

    
    # First, ask user if he want to discard annotations
    question_img = orig_frame_img.copy()
    cv2.putText(question_img, "Remove bbs? (y/n)", (1280//2, 720//6), cv2.FONT_HERSHEY_SIMPLEX,
                1.9, (0, 0, 255), 4, cv2.LINE_AA)

    utils.place_window(winname, winnsize=configs.winnsize)  # Place the window at the top left corner
    cv2.imshow(winname, question_img)

    while True:
        key = cv2.waitKey(0)
        if key in (ord('y'), ord('Y')):
            cv2.destroyAllWindows()
            break
        elif key in (ord('n'), ord('N')):
            # add them to annotations.json
            # utils.add_to_coco(args, images, annotations)
            args.annotations['images'].extend(images)
            args.annotations['annotations'].extend(annotations)
            utils.save_coco_annotations(args.annotations, os.path.join(args.coco_data, 'annotations.json'))
            cv2.destroyAllWindows()
            break

    

#########################################################
#                YOLO + SAM2 Auxiliary                  #
#########################################################
def run_propagation(inference_state):
  video_segments = {}  # video_segments contains the per-frame segmentation results
  for out_frame_idx, out_obj_ids, out_mask_logits in sam_video_predictor.propagate_in_video(inference_state):
      video_segments[out_frame_idx] = {
          out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
          for i, out_obj_id in enumerate(out_obj_ids)
      }
  return video_segments


def second_stage(sam_video_predictor, output_frames_path, detected_frame_idx):
    """
    Initialise SAM once, add obj-10 (tool) + obj-20 (hand) prompts,
    then propagate.
    """
    global tool_hand_segmentation
    # 1) init state on the temp frame folder
    inference_state = sam_video_predictor.init_state(video_path=output_frames_path)
    TOOL_ID, HAND_ID = 10, 20

    if tool_hand_segmentation['1']['start_point'] and tool_hand_segmentation['1']['end_point']:
        x1, y1 = start_point
        x2, y2 = end_point
        sam_video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=detected_frame_idx,
            obj_id=TOOL_ID,
            box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        )

    # ----------------------------------------------------  TOOL
    if tool_hand_segmentation['1']['include_points']:      
        sam_video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=detected_frame_idx,
            obj_id=TOOL_ID,
            points=np.array(tool_hand_segmentation['1']['include_points'] +
                             tool_hand_segmentation['1']['exclude_points'],
                             dtype=np.float32),
            labels=np.array([1]*len(tool_hand_segmentation['1']['include_points']) +
                            [0]*len(tool_hand_segmentation['1']['exclude_points']),
                            dtype=np.int32)
        )

    # ----------------------------------------------------  HAND
    if tool_hand_segmentation['2']['include_points']:      
        sam_video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=detected_frame_idx,
            obj_id=HAND_ID,
            points=np.array(tool_hand_segmentation['2']['include_points'] +
                             tool_hand_segmentation['2']['exclude_points'],
                             dtype=np.float32),
            labels=np.array([1]*len(tool_hand_segmentation['2']['include_points']) +
                            [0]*len(tool_hand_segmentation['2']['exclude_points']),
                            dtype=np.int32)
        )

    video_segments = run_propagation(inference_state)  
    return video_segments


def sam_prompt_to_polygons(img_bgr, eps=1.5, min_area=10):
    """
    Run SAM on Image -- if the user want to edit the bounding box, hand segmentation
    @param img_bgr: The input image in BGR format
    @param eps: The epsilon value for polygon approximation
    @param min_area: The minimum area for a contour to be considered valid
    """
    global sam_predictor, tool_hand_segmentation

    sam_predictor.set_image(img_bgr[..., ::-1].copy())
    H, W = img_bgr.shape[:2]
    results = {}

    # ------------------------------------------------------- helpers
    def _mask_to_polys(mask):
        polys = []
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue
            polys.append(c.squeeze(1).tolist())
        return polys

    def _run_sam(box=None, points=None, labels=None):
        masks, _, _ = sam_predictor.predict(
            box=np.asarray([box]) if box is not None else None,
            point_coords=np.asarray(points) if points is not None else None,
            point_labels=np.asarray(labels) if labels is not None else None,
            multimask_output=False)
        return masks[0].astype(np.uint8)

    # -------------------------------------------------------  1. RECTANGLE?e
    if tool_hand_segmentation['1']['start_point'] and tool_hand_segmentation['1']['end_point']:
        x1, y1 = tool_hand_segmentation['1']['start_point']
        x2, y2 = tool_hand_segmentation['1']['end_point']
        box = [min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)]
        mask = _run_sam(box=[min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
        results["tool"] = {"bbox": [box], "segs": _mask_to_polys(mask)}
        print(f" [*] Detected tool with bounding box: {len(results['tool']['segs'])}")

    # -------------------------------------------------------  2. POINT PROMPTS?
    for prompt_key, label in (('1', "tool"), ('2', "hand")):
        inc = tool_hand_segmentation[prompt_key]['include_points']
        exc = tool_hand_segmentation[prompt_key]['exclude_points']
        if not inc and not exc:                 # no clicks -- skip this class
            continue

        pts = np.array(inc + exc, np.float32)
        lbl = np.array([1] * len(inc) + [0] * len(exc), np.int32)

        mask = _run_sam(points=pts, labels=lbl)
        polys = _mask_to_polys(mask)

        entry = {"mask": mask, "segs": polys}

        # ─── only the TOOL class needs a bounding-box ──────────────────
        if label == "tool":
            # fastest: bounding box of the binary mask
            ys, xs = np.where(mask)             # pixel coords where mask == 1
            if xs.size:                         # guard against empty mask
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                entry["bbox"] = [[int(x_min), int(y_min),
                                int(x_max - x_min), int(y_max - y_min)]]

        results[label] = entry

    return results


#########################################################
#             Annotation Functionality                  #
#########################################################
def locate_tool_from_frame(frame_image,
                           last_detected_frame,
                           ask_user=False,):
    """
    Locate manually by points the tool in the current frame

    @param frame_image:
    @param last_detected_frame:
    @param ask_user:
    @return:
    """
    global winname, next_video, segment_by_points

    if segment_by_points:           # Choose points?

        # TODO -- ask user if he want to skip the video withen the window of opencv
        if ask_user:
            # Add text to the frame
            question_img = frame_image.copy()
            quere = f'[N] Next Video   [X] Exit Program'
            cv2.putText(question_img, quere, (60, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        1.9, (0, 255, 255), 3, cv2.LINE_AA)

            cv2.imshow(winname, question_img)

            while True:
                key = cv2.waitKey(0)
                if key in (ord('n'), ord('N')):
                    next_video = True
                    return False, -1
                elif key in (ord('x'), ord('X')):
                    cv2.destroyAllWindows()
                    exit(0)  # Exit the program:
                else:
                    break

            ask_user = False
        
        add_manually_bb(frame_image)
    return ask_user, last_detected_frame


def does_yolo_failed(last_detected_frame, i, boarder=9):
    """
    The yolo should run on the first 10 frames, if failed to detect,
    make the user detected manually.

    :param last_detected_frame: last frames bb detected
    :param i                  : the current frame index
    :param boarder            : number of frames for yolo to check
    :return:
    """
    if np.abs(last_detected_frame - i) > boarder:
        return True
    return False


def annotate_all_video_manually(v_path, curr_tool_output_path=None,
                                frame_names=None,
                                frame_paths=None,
                                preview_before_save=False, n=50):
    """
    Manually annotate $n samples from the input video in a roboflow scheme annotations.
    :param v_path: path to the video
    :param curr_tool_output_path: path to the result directory
    :param n: number of samples to choose
    :return:
    """
    global winname, image_copy, clean_state
    vid_name = os.path.split(v_path)[-1].split('.')[0]
    if curr_tool_output_path is None:
        sub_root_saved_path = os.path.join(curr_tool_output_path, f'{vid_name}')
    else:
        sub_root_saved_path = curr_tool_output_path

    frame_paths = os.path.join(sub_root_saved_path, 'frames')
    os.makedirs(sub_root_saved_path, exist_ok=True)
    os.makedirs(frame_paths, exist_ok=True)

    if frame_names is None:
        utils.video_to_frames(v_path, frame_paths)
        frame_names = [
            p for p in os.listdir(frame_paths)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        # work only with 100 frames
        frame_names = utils.get_k_frames(frame_names)

    saved_path = os.path.join(sub_root_saved_path, f'BBvis')
    os.makedirs(saved_path, exist_ok=True)
    first_frame = cv2.imread(os.path.join(frame_paths, frame_names[0]))
    frame_length = len(frame_names)
    samples = range(len(frame_names))  # no rndom choice
    height, width, _ = first_frame.shape
    fps = 30
    if v_path is not None:
        cap = cv2.VideoCapture(v_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
    out = cv2.VideoWriter(os.path.join(sub_root_saved_path, f'bbVIS_{vid_name}.mp4'),
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (width, height))
    tool_bbs = {}
    for counter, sample_idx in enumerate(samples):
        frame_number = int(frame_names[sample_idx].split('.')[0])
        frame_image = cv2.imread(os.path.join(frame_paths, frame_names[sample_idx]))
        winname = f'{frame_number}/{frame_length}'
        print(f'{counter + 1}/{len(samples)}')
        image_copy = frame_image.copy()  # Create a copy for resetting during drawing
        clean_state = frame_image.copy()
        cv2.imshow(winname, frame_image)
        cv2.setMouseCallback(winname, mouse_callback)
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imwrite(os.path.join(saved_path, f'{frame_number:05d}.jpg'), image_copy)
        out.write(image_copy)
        if start_point is not None and end_point is not None:
            tool_bbs[f'{sample_idx}'] = utils.convert_bb_to_yolo_format([*start_point, *end_point], width, height)
        else:
            # skipping the frame
            if not utils.prompt_question(True, query="Are you sure?"):
                print(f'DELETING THIS BB {sample_idx}')
                tool_bbs[f'{sample_idx}'] = [-1, -1, -1, -1]

    out.release()

    if preview_before_save:
        run_preview(os.path.join(sub_root_saved_path, f'bbVIS_{vid_name}.mp4'))

    return tool_bbs, {}


def annotate_video_using_sam(args,
                             curr_tool_output_path=None,
                             extract_tool=True,
                             frame_names=None,
                             preview_before_save=False, debug=False):
    """
    This function is responsible for annotating the given video by:
        - point prompts
        - Hand segmentations as a matrix representation for the mask

    @param args: namespace contains all the user choices
    @param curr_tool_output_path: output path to save the results
    @param extract_tool: a flag whether to extract BB (if a tool is used in the video)
    @param frame_names: a flag whether to extract the hand segmentations
    @param opt_for_manual: a flag whether to ask the user for manually add bb
    @param preview_before_save: play the results as a video (bb results)
    @param debug: whether to debug the algorithm steps
    @return tool_bbs, hands_segmentations: a dict for each frame the bbs and hand seg.
    """
    global winname, image_copy, clean_state, segment_by_points, next_video, tool_hand_segmentation
    v_path = args.video_path
    segment_by_points = True
    vid_name = os.path.split(v_path)[-1].split('.')[0]
    if curr_tool_output_path is None: # wrong here, fix
        sub_root_saved_path = os.path.join(curr_tool_output_path, f'{vid_name}')
    else:
        sub_root_saved_path = curr_tool_output_path

    frame_paths = os.path.join(sub_root_saved_path, 'frames')
    os.makedirs(sub_root_saved_path, exist_ok=True)
    os.makedirs(frame_paths, exist_ok=True)

    if frame_names is None:
        with utils.suppress_output():  # to suppress the output to stdout
            utils.video_to_frames_slow(v_path, frame_paths)   # change this to video_to_frames
        frame_names = [
            p for p in os.listdir(frame_paths)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    current_working_file = os.path.join(sub_root_saved_path, 'working_file')  # file that holds n images each time
    os.makedirs(current_working_file, exist_ok=True)

    hands_segmentations = {}
    combined_video_segments = {}
    success_indices = []
    ask_user = True   # ask if skip current video (SAM2 purpose)n
    skipped_frames = {}  # none annotated frames
    i = 0
    last_detected_frame = i
    total_frames = len(frame_names) - 1
    winname = f'{vid_name}'

    utils.place_window(winname, winnsize=configs.winnsize)  # Place the window at the top-left corner

    while i < total_frames:
        reset_globals()
        # Prepare the current window of frames
        indices = utils.find_range(i, frame_names, range_step=args.repeat)
        if debug:
            print(f' > Starting with {indices[0]} - {indices[-1]}/{len(indices)} frames from {len(frame_names)} frames')

        if extract_tool:
            # Stage 1: point prompts or manual annotation
            frame_number = int(frame_names[i].split('.')[0])
            frame_image = cv2.imread(os.path.join(frame_paths, frame_names[i]))
            cv2.putText(frame_image, f'{i}/{total_frames}', (configs.x_offset, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2, cv2.LINE_AA)
            
            ask_user, last_detected_frame = locate_tool_from_frame(frame_image,
                                                                   last_detected_frame,
                                                                   ask_user=ask_user,)

            if next_video:
                cv2.destroyAllWindows()
                return

            if (len(tool_hand_segmentation['1']['include_points']) == 0 and len(tool_hand_segmentation['2']['include_points']) == 0) and (start_point is None or end_point is None):
                skipped_frames[f'{i}'] = [[-1, -1, -1, -1]]
                i += 1
                continue


            # Stage 2: apply SAM2 on the manually bb
            if debug:
                for obj_id in tool_hand_segmentation.keys():
                    print(f'{obj_id}: ')
                    for stream in tool_hand_segmentation[obj_id]:
                        print(f' > {stream}: {tool_hand_segmentation[obj_id][stream]}')

            print(f' > SAM FOR : {indices} - {indices[0]} - {indices[-1]}')
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            # cv2.waitKey(1)
            curr_frame_names = utils.initial_working_dir(frame_paths, frame_names, current_working_file, indcies=indices)

            video_segments = second_stage(sam_video_predictor, current_working_file, 0)
            video_segments = {k + i: v for k, v in video_segments.items()}
            combined_video_segments = {**combined_video_segments, **video_segments}
            success_indices.append(indices)

        # Algorithm step
        i = indices[-1] + 1
        last_detected_frame = indices[-1]

    cv2.destroyAllWindows()
    # SAVING SAM RESULTS
    tool_bbs, tool_segs, hand_segs = utils.extract_annotations(success_indices,
                                                               combined_video_segments,
                                                               format='coco')

    # to ensure the correct annotation, run a preview and edit the one that needed to editted.
    if preview_before_save:
        run_preview(sub_root_saved_path, tool_bbs, tool_segs, hand_segs, save_vis=args.save_visualization)

    if args.save_visualization:
        utils.reset_working_dir(frame_paths, delete_=True)
        utils.reset_working_dir(current_working_file, delete_=True)
    else:    
        utils.reset_working_dir(sub_root_saved_path, delete_=True)
    # utils.reset_working_dir(frame_paths, delete_=True)


def get_different_from_original(length_, k=100):
    """
    When creating the annotations, we save the bbs of the 100 choose frame to annotate with starting index as 0
    although 0 is not the original starting frame index since we choose a window of size 100 from the whole video timeline

    to get the original index, we need to shift back the window exactly the same way we shifted it to choose 100 frames

    see function utils/get_k_frames()
    """

    if length_ > k:
        return (length_ - k) // 2

    return 0


################################################################################################################ START Fix COCO

def fix_annotations(args, target_size=(640, 460)):
    """
    Fix the annotations of the given directory path
    :param args: user options
    :return:
    """
    global winname, start_point, end_point
    if 'coco_annotation_tracker_idx' not in args.tracker:
        args.tracker['coco_annotation_tracker_idx'] = -1

    start_id = args.tracker['coco_annotation_tracker_idx']

    """
    current mapping:

    adjs -> wrench
    tape -> plier
    drill -> ratchet
    plier -> tapemeasure
    ratchet -> hammer
    """

    new_coco = utils.load_coco_annotations(args.output_path) if os.path.exists(args.output_path) else {
            "images": [],
            "annotations": [],
            "categories":[],
        }
    coco = utils.load_coco_annotations(args.annotations_path)
    coco["images"].sort(key=lambda x: x["id"])  # or x["file_name"]
    coco["annotations"].sort(key=lambda x: x["id"])

    for img_idx, img_info in enumerate(coco["images"]):
        if img_info["id"] <= start_id:  # already done.
            continue
        
        reset_globals()
        image_path = Path(args.images_path) / img_info["file_name"]
        if not image_path.exists():
            print(f"[Missing] {image_path}")
            continue

        try:
            img = Image.open(image_path)
            orig_width, orig_height = img.size
            # img = img.resize(target_size)
            img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            scale_x = orig_width / target_size[0] 
            scale_y = orig_height / target_size[1]
        except Exception as e:
            print(f"[INVALID IMAGE] {image_path} -> {e}")
            continue
        print(f'img_dx: {img_idx} - {img_info["file_name"]}')
        anns = coco["annotations"][img_idx]

        # -----------------------------------  Edditing bb
        exit_flag = False
        while True:
            image_display = utils.draw_cocoBB_from_annotations(img_np.copy(), [anns], configs.category_id_to_name,
                                                       orig_width=orig_width, orig_height=orig_height,
                                                       target_size=(orig_width, orig_height))
            instr = "[a/Enter] Accept   [e] Edit BB   [n] Change Category   [q] Quit"
            cv2.putText(image_display, instr, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 2, cv2.LINE_AA)

            winname = f"Fix Annotations - class: ({orig_width},{orig_height}){configs.category_id_to_name.get(anns['category_id'], 'Unknown')} ({img_idx + 1}/{len(coco['images'])})"
            cv2.imshow(winname, image_display)
            cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            key = cv2.waitKey(0) & 0xFF
            if key in (ord('q'), 27):  # Quit
                exit_flag = True
                break

            elif key == ord('e'):  # Edit bounding box
                # cv2.destroyAllWindows()
                reset_globals()
                add_manually_bb(img_np)
                if start_point and end_point:
                    x1, y1 = start_point
                    x2, y2 = end_point
                    new_w, new_h = abs(x2 - x1), abs(y2 - y1)
                    anns['bbox'] = [min(x1, x2), min(y1, y2), new_w, new_h]
                    anns['area'] = new_w * new_h
                    print(f" Updated BB: {[x1, y1, new_w, new_h]}")
                else:
                    print(" No BB drawn. Annotation will be skipped.")
                    take_it = False
                    break  # skip to next image

            elif key == ord('n'):  # Change category in-GUI
                image_with_classes = image_display.copy()
                y_offset = 60
                class_instr_lines = []
                class_instr = ""
                for cid, cname in configs.category_id_to_name.items():
                    class_instr += f"[{cid}] {cname}   "
                    if cid % 3 == 2:
                        class_instr_lines.append(class_instr.strip())
                        class_instr = ""
                if class_instr:
                    class_instr_lines.append(class_instr.strip())
                for line in class_instr_lines:
                    cv2.putText(image_with_classes, line, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                    y_offset += 30
                cv2.imshow(winname, image_with_classes)
                key2 = cv2.waitKey(0) & 0xFF
                selected_class = key2 - ord('0')
                if selected_class in configs.category_id_to_name:
                    anns["category_id"] = selected_class
                    print(f" Updated category to {configs.category_id_to_name[selected_class]}")
                else:
                    print(" Invalid class. No change.")

            elif key in (ord('a'), 13):  # Accept (Enter or 'a')
                take_it = True
                break

        if exit_flag:
            break

        if take_it:
            print(f"✔ Accepted: {anns['bbox']} / class: {configs.category_id_to_name[anns['category_id']]}")
            new_coco['images'].append(img_info)
            new_coco['annotations'].append(anns)

        cv2.destroyAllWindows()
        args.tracker['coco_annotation_tracker_idx'] = img_info["id"]

        # --------------------------------------------------  finnished
        # save the tracker
        with open('tracker.json', 'w') as f:
            json.dump(args.tracker, f, indent=4)

        # save the new annotations
        new_coco["categories"] = coco["categories"]
        print(f"Fixed annotations saved to {args.output_path}")
        print(f"Last processed image ID: {args.tracker['coco_annotation_tracker_idx']}")
        utils.save_coco_annotations(new_coco, args.output_path)

################################################################################################################ End Fix COCO

def annotator(args, fixer=False):
    """
    The introduction of annotating a video. Here, the annotation scheme is done based on user options
    :param args: user options
    :return:
    """
    global segment_by_points
    tracker = utils.read_json(tracker_path)
    args.tracker = tracker

    if args.video_path is not None:     # Single video
        vid_name = os.path.split(args.video_path)[-1].split('.')[0]
        curr_tool_output_path = os.path.join(args.output_dir, vid_name)
        os.makedirs(curr_tool_output_path, exist_ok=True)
        if args.manually:
            tool_bbs, hands_segmentations = annotate_all_video_manually(args.video_path, curr_tool_output_path)
        else:
            tool_bbs, hands_segmentations = annotate_video_using_sam(args, curr_tool_output_path, preview_before_save=True)

        # print(f'tool_bbs: {tool_bbs}\n\nhand_segmentations: {hands_segmentations}')


    elif args.directory_path:
        tool_categories = [tool_cat for tool_cat in os.listdir(args.directory_path) if '.DS_' not in tool_cat ]#and tool_cat in configs.CATEGORIES]
        for tool_category in tool_categories:
            if tool_category in configs.NONE_TOOL_CATEGORIES:
                continue
            print(f'Annotating {tool_category} videos')
            args.curr_tool_id = args.category_id_mapping[tool_category] if tool_category in args.category_id_mapping else -1
            videos = [vid for vid in os.listdir(os.path.join(args.directory_path, tool_category)) if vid.endswith(('.mp4', '.MP4', '.mov'))]
            for video in videos:
                video_path = os.path.join(args.directory_path, tool_category, video)
                args.video_path = video_path
                vid_name = os.path.split(video_path)[-1].split('.')[0]
                if vid_name in args.done_video_names:
                    continue
                args.vid_name = vid_name
                curr_tool_output_path = os.path.join(args.output_dir, vid_name)
                os.makedirs(curr_tool_output_path, exist_ok=True)

                print(f' > Annotating {vid_name}')
                if args.manually:
                    tool_bbs, hands_segmentations = annotate_all_video_manually(args.video_path, curr_tool_output_path)
                else:
                    annotate_video_using_sam(args, curr_tool_output_path, preview_before_save=True)

    elif args.fixer:  # Fix annotations
        # Fix the annotations
        print(f'Fixing annotations in {args.directory_path}')
        fix_annotations(args)


def _yes_no(prompt, default=False):
    """
    Ask a yes/no question on stdin and return True/False.
    Empty input returns the default.
    """
    while True:
        ans = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
        if not ans:
            return default
        if ans in {"y", "yes", "Y", "Yes", "YES"}:
            return True
        if ans in {"n", "no", "N", "No", "NO"}:
            return False
        print("  ➜ Please answer with y / n")
    

def ask_user_for_run_config():
    """
    Query the user for the desired annotation mode and store the
    answers in a global variable called `args`.
    """
    global args, args_parser                    # the rest of the script expects this
    print("\n=== Annotation Scheme - interactive setup ===")

    # 1) choose high-level mode ------------------------------------------------
    print("Choose what to annotate:")
    print("  1) A single video file")
    print("  2) Directory of videos")
    print("  3) Fix Annotations")


    mode = None
    while mode not in {"1", "2", "3"}:
        mode = input("Mode [1/2/3]: ").strip()
    
    weights_name_mapping = {
        't': 'tiny',
        's': 'small',
        'b': 'base_plus',
        'l': 'large',
    }

    # initialise defaults --------------------------
    args = SimpleNamespace(
        video_path=None,
        directory_path=None,
        output_dir="annotation_results",
        weights=weights_name_mapping[args_parser.weights],
        manually=False,
        repeat=args_parser.repeat,
        fixer=False,
        tracker=None,
        coco_data=None,
        done_video_names=None,
        save_visualization=False,
    )

    # ---------------------------------------------- single video
    if mode == "1":                               
        args.video_path = utils.input_with_path_completion("Full path to video: ")
        args.manually = _yes_no("Annotate manually (skip SAM2)?", default=False)
        od = utils.input_with_path_completion(f'Output directory (default: {args.output_dir}): ')
        if od:
            args.output_dir = od
        args.save_visualization = _yes_no("Save visualization results?", default=False)

    # ---------------------------------------------- Directory
    elif mode == "2":                            
        args.directory_path = utils.input_with_path_completion(f"Directory path (current dir is - {os.getcwd()}): ")
        while not os.path.exists(args.directory_path):
            print(f'[!] Invalid Path')
            args.directory_path = utils.input_with_path_completion(f"Directory path (current dir is - {os.getcwd()}): ")
        args.manually = _yes_no("Annotate manually (skip SAM2)?", default=False)
        args.save_visualization = _yes_no("Save visualization results?", default=False)
        args.new_shape = args_parser.new_shape # default shape for the video frames
        args.coco_data = utils.input_with_path_completion(f"Saved directory path (will be saved under - {os.getcwd()}): ")
        if args.coco_data == '':
            args.coco_data = 'results_coco_format'
        if not os.path.exists(args.coco_data):
            Path(f'{args.coco_data}/images').mkdir(parents=True, exist_ok=True)
            args.annotations = {'images': [], 'annotations': [],
                "categories": [
                {"id": 0, "name": "SL", "supercategory": "tool"},
                {"id": 1, "name": "adjustable spanner", "supercategory": "tool"},
                {"id": 2, "name": "allen", "supercategory": "tool"},
                {"id": 3, "name": "drill", "supercategory": "tool"},
                {"id": 4, "name": "hammer", "supercategory": "tool"},
                {"id": 5, "name": "plier", "supercategory": "tool"},
                {"id": 6, "name": "ratchet", "supercategory": "tool"},
                {"id": 7, "name": "screwdriver", "supercategory": "tool"},
                {"id": 8, "name": "tapemeasure", "supercategory": "tool"},
                {"id": 9, "name": "wrench", "supercategory": "tool"},
                {"id": 10, "name": "hands", "supercategory": "hands"},
                {"id": 11, "name": "saw", "supercategory": "tool"},
            ],}
            with open(os.path.join(args.coco_data, 'annotations.json'), 'w') as f:
                json.dump(args.annotations, f, indent=4)
            args.done_video_names = set()
        else:
            with open(os.path.join(args.coco_data, 'annotations.json'), 'r') as f:
                args.annotations = json.load(f)
            # extract all names of video
            if os.path.exists(os.path.join(args.coco_data, "done_video_names.pkl")):
                args.done_video_names = utils.save_open_video_names_as_pickles(None, path=os.path.join(args.coco_data, "done_video_names.pkl"), op='open')
            else:
                args.done_video_names = set([fname['file_name'].split('_')[0] for fname in args.annotations['images']])

        args.category_id_mapping = {cat['name'].lower(): cat['id'] for cat in args.annotations['categories']}
        args.id_category_mapping = {cat['id']: cat['name'].lower() for cat in args.annotations['categories']}

    # ---------------------------------------------- Fixer
    elif mode == "3":
        args.fixer = True
        coco_folder_path = utils.input_with_path_completion(f"The path to coco based format folder [ccontains images and annotations.json] getcwd: {os.getcwd()}: ")
        args.images_path = os.path.join(coco_folder_path, 'images')
        args.annotations_path = os.path.join(coco_folder_path, 'annotations.json')
        args.output_path = os.path.join(coco_folder_path, 'fix_annotations.json')
    
    
    os.makedirs(args.output_dir, exist_ok=True)
    print("\n--------------------------------------------------------------------")
    print("Arguments:")
    for k, v in vars(args).items():
        if k not in ['annotations']:
            print(f"  {k:20s}: {v}")
    print("--------------------------------------------------------------------\n")
    return args


if __name__ == "__main__":

    args = ask_user_for_run_config()
    annotator(args)
    if args.coco_data is not None:
        utils.save_open_video_names_as_pickles(args.done_video_names, path=os.path.join(args.coco_data, 'done_video_names.pkl'), op='save')
