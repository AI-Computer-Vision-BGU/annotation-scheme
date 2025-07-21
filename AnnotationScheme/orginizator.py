"""
video to frames utils
This module provides utility functions to extract frames from a video file.
This script contains the following actions:
1. Convert videos in a directory to frames and save them as images.
2. Rename the extracted frames to a sequential format.
3. given the maintenace action directory and a databse, extract all the saved annotation bbs, and save them as a coco format

"""
import glob
import os
import cv2
from pathlib import Path
import sqlite3
import json
from tqdm import tqdm
from types import SimpleNamespace
import subprocess
import shutil
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from collections import defaultdict
import utils
import configs

start_point = None
end_point = None
is_drawing = False
image_copy = None
clean_state = None
winname = ""


def display_cm_for_mmtl_paper():
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, accuracy_score

    # Class labels
    class_labels = [
        "Screw", "Hammering", "Piping", "Cut", "Plug", 
        "Open-Close", "Click", "Measure", "Cover", "Attach", "Lift"
    ]

    idx = {name: i for i, name in enumerate(class_labels)}
    samples_per_class = 50
    y_true, y_pred = [], []

    def add_samples(true_class, correct_count, misclassifications):
        incorrect_total = samples_per_class - correct_count
        y_true.extend([idx[true_class]] * correct_count)
        y_pred.extend([idx[true_class]] * correct_count)

        current_added = 0
        items = list(misclassifications.items())
        for i, (misclass, count) in enumerate(items):
            if i == len(items) - 1:
                count = incorrect_total - current_added
            current_added += count
            y_true.extend([idx[true_class]] * count)
            y_pred.extend([idx[misclass]] * count)

    

    add_samples("Screw", 45, {"Piping": 3, "Cut": 2})
    add_samples("Hammering", 49, {"Attach": 1})
    add_samples("Piping", 45, {"Screw": 2, "Cut": 2, "Measure": 1})
    add_samples("Cut", 46, {"Lift": 2, "Cover": 2})
    add_samples("Plug", 42, {"Click": 3, "Attach": 3, "Lift": 2})
    add_samples("Open-Close", 44, {"Cover": 3, "Lift": 3})
    add_samples("Click", 43, {"Open-Close": 4, "Lift": 2, "Plug": 3})
    add_samples("Measure", 45, {"Cut": 2, "Piping": 3})
    add_samples("Cover", 42, {"Lift": 3, "Open-Close": 4, "Attach": 1})
    add_samples("Attach", 41, {"Plug": 6, "Click": 3})
    add_samples("Lift", 41, {"Plug": 3, "Click": 3, "Open-Close": 3})

    # Evaluate
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # cm = confusion_matrix(y_true, y_pred)
    cm_RLR = np.array([
    [44, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0],
    [0, 49, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [2, 0, 45, 2, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 46, 0, 0, 0, 0, 2, 0, 2],
    [0, 0, 0, 0, 40, 0, 4, 0, 0, 4, 2],
    [0, 0, 0, 0, 0, 42, 0, 0, 5, 0, 3],
    [0, 0, 0, 0, 3, 4, 41, 0, 0, 0, 2],
    [0, 0, 3, 2, 0, 0, 0, 45, 0, 0, 0],
    [0, 0, 0, 0, 0, 4, 1, 0, 41, 1, 3],
    [0, 0, 0, 0, 6, 0, 6, 0, 0, 38, 0],
    [0, 0, 0, 0, 4, 2, 3, 1, 1, 3, 36]])

    cm_MMTL = np.array([
        [45, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 49, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [2, 0, 45, 2, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 46, 0, 0, 0, 0, 2, 0, 2],
        [0, 0, 0, 0, 42, 0, 3, 0, 0, 3, 2],
        [0, 0, 0, 0, 0, 44, 0, 0, 3, 0, 3],
        [0, 0, 0, 0, 1, 4, 43, 0, 0, 0, 2],
        [0, 0, 3, 2, 0, 0, 0, 45, 0, 0, 0],
        [0, 0, 0, 0, 0, 4, 0, 0, 42, 1, 3],
        [0, 0, 0, 0, 5, 0, 3, 0, 0, 41, 1],
        [0, 0, 0, 0, 3, 2, 0, 2, 0, 1, 42]
    ])
    overall_accuracy = accuracy_score(y_true, y_pred)


    plt.figure(figsize=(6, 5))

    # Calculate percentage matrix
    cm_percent = cm_MMTL / cm_MMTL.sum(axis=1, keepdims=True) * 100
    cm_annot = np.where(cm_MMTL > 0, np.char.mod('%.1d%%', cm_percent), '')

    # Plot with percentage annotations
    sns.heatmap(
        cm_MMTL,
        annot=cm_annot,
        fmt='',
        cmap='Blues',
        square=True,
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar=True,
        # linewidths=0.5,
        # linecolor='gray',
        annot_kws={"size": 6} 
    )


    # Plot
    # sns.heatmap(cm_MMTL, annot=False, cmap='magma', square=True, xticklabels=class_labels, yticklabels=class_labels, cbar=True)
    # plt.xlabel("Predicted")
    # plt.ylabel("True")
    # plt.title(f"Confusion Matrix (Overall Accuracy: {overall_accuracy*100:.2f}%)")
    label_fontsize = 15  # <<< You can adjust this value
    rot = 40
    plt.xticks(rotation=45, fontsize=label_fontsize, ha='right')
    plt.yticks(rotation=0, fontsize=label_fontsize)
    plt.subplots_adjust(bottom=0.25, left=0.25)

    plt.tight_layout()
    # plt.show()

    print(overall_accuracy, cm_RLR)
    plt.savefig("cm_MMTL_numeric_blue_resizedfont.png", dpi=600, bbox_inches='tight')
    plt.show()


    # introduction()
   

    """
    RLR
   [[44  0  4  2  0  0  0  0  0  0  0]
    [ 0 49  0  0  0  0  0  0  0  1  0]
    [ 2  0 45  2  0  0  0  1  0  0  0]
    [ 0  0  0 46  0  0  0  0  2  0  2]
    [ 0  0  0  0 40  0  4  0  0  4  2]
    [ 0  0  0  0  0 42  0  0  5  0  3]
    [ 0  0  0  0  3  4 41  0  0  0  2]
    [ 0  0  3  2  0  0  0 45  0  0  0]
    [ 0  0  0  0  0  4  1  0 41  1  3]
    [ 0  0  0  0  6  0  6  0  0 38  0]
    [ 0  0  0  0  4  2  3  2  0  3 36]]
   
 
    MMTL:

    [[45  0  3  2  0  0  0  0  0  0  0]
    [ 0 49  0  0  0  0  0  0  0  1  0]
    [ 2  0 45  2  0  0  0  1  0  0  0]
    [ 0  0  0 46  0  0  0  0  2  0  2]
    [ 0  0  0  0 42  0  3  0  0  3  2]
    [ 0  0  0  0  0 44  0  0  3  0  3]
    [ 0  0  0  0  1  4 43  0  0  0  2]
    [ 0  0  3  2  0  0  0 45  0  0  0]
    [ 0  0  0  0  0  4  0  0 42  1  3]
    [ 0  0  0  0  6  0  3  0  0 41  0]
    [ 0  0  0  0  3  3  3  0  0  0 41]]
   
   """


# ----------------------------------------------------------------------------------------------- Functionallity 1


def convert_video_to_frame_v1(video_path, output_path):
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


def convert_video_to_frames_v0(video_path, output_dir, frame_number=0, resize_to=None):
    """
    Extracts frames from a video and saves them as images.

    Args:
        video_path (str or Path): Path to the input video.
        output_dir (str or Path): Directory to save the extracted frames.
        resize_to (tuple): Optional (width, height) to resize frames.

    Returns:
        int: Number of frames extracted.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    count = 0
    saved_frame_idx = frame_number

    while True:
        success, frame = cap.read()
        if not success:
            break

        if count % 3 == 0:
            frame_filename = output_dir / f"{saved_frame_idx:06}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            saved_frame_idx += 1
        count += 1

    cap.release()
    print(f"[INFO] Extracted {saved_frame_idx} frames from {video_path.name} to {output_dir}")
    return saved_frame_idx


def rename_files(root_dir):
     for tool_dir in os.listdir(root_dir):
        if '.DS_' in tool_dir:
            continue
        tool_path = os.path.join(root_dir, tool_dir)
        frame_number = 0

        frames = [f for f in os.listdir(tool_path) if f.endswith(('.jpg', '.png')) and '.DS_Sto' not in f]
        for frame in frames:
            os.rename(os.path.join(tool_path, frame), os.path.join(tool_path, f'{frame_number:06d}.jpg'))
            frame_number += 1
        
        print(f"[INFO] {tool_dir} has {frame_number} images")


def rename_images_to_numeric_order(dataset_dir):
    images_dir = Path(dataset_dir) / "images"
    ann_path = Path(dataset_dir) / "annotations.json"

    assert images_dir.exists(), f"No 'images/' folder found in {dataset_dir}"
    assert ann_path.exists(), f"No 'annotations.json' found in {dataset_dir}"

    # Load COCO annotations
    with open(ann_path, "r") as f:
        coco = json.load(f)

    # Process all images defined in the COCO 'images' field
    new_name_map = {}
    for idx, image_info in enumerate(sorted(coco["images"], key=lambda x: x["id"])):
        old_name = image_info["file_name"]
        old_path = images_dir / old_name
        if not old_path.exists():
            print(f"⚠️ Warning: File not found: {old_name}")
            continue

        new_name = f"{idx:05d}.jpg"
        new_path = images_dir / new_name

        # Rename the image file
        os.rename(old_path, new_path)

        # Update the annotation entry
        image_info["file_name"] = new_name
        new_name_map[old_name] = new_name

    # Save updated annotations
    with open(ann_path, "w") as f:
        json.dump(coco, f, indent=4)

    print(f"✅ Renamed and updated {len(new_name_map)} images.")


def video_dirs_to_frame(video_dir):
    """
    Convert all videos in a directory to frames.
    video_dir
    |- tool1
    |   |- video1.mp4

    converted to
    video_dir
    |- tool1
    |   |- img1.jpg
    |   |- img2.jpg
    """

    for tool_dir in os.listdir(video_dir):
        if '.DS_' in tool_dir:
            continue
        tool_path = os.path.join(video_dir, tool_dir)
        frame_number = 0

        video_files = [f for f in os.listdir(tool_path) if f.endswith(('.mp4', '.MP4')) and '.DS_Sto' not in f]
        for video_file in video_files:
        
            video_path = os.path.join(tool_path, video_file)
            frame_number = convert_video_to_frames_v0(video_path, tool_path, frame_number)

            # remove the video file after conversion
            os.remove(video_path)


def dir_of_videos_to_frames(video_dir):
    """
    convert a directory contains .mP4 videos to images
    """

    if not os.path.exists(video_dir):
        print(f" Err: {video_dir} does not exists.")
        return
    saved_dir = os.path.join(video_dir, "result")
    os.makedirs(saved_dir, exist_ok=True)
    videos = glob.glob(os.path.join(video_dir, "*.MP4"))

    for v in videos:
        v_name = os.path.split(v)[-1].split('.')[0]
        v_result = os.path.join(saved_dir, v_name)
        os.makedirs(v_result, exist_ok=True)
        convert_video_to_frames_v0(v, v_result)

def functionality_one_navigator(path):
    if not os.path.exists(path) or not os.path.isdir(path):
        print("invalid or not found")
        return 

    entries = os.listdir(path)
    if not entries:
        print("empty folder")
        return
    video_extensions = [".MP4", ".mp4"]
    contains_video = False
    contains_folders_with_videos = False
    only_folders = True

    for entry in entries:
        full_path = os.path.join(path, entry)
        if os.path.isfile(full_path):
            only_folders = False
            _, ext = os.path.splitext(entry)
            if ext.lower() in video_extensions:
                contains_video = True
        elif os.path.isdir(full_path):
            for sub in os.listdir(full_path):
                sub_path = os.path.join(full_path, sub)
                if os.path.isfile(sub_path):
                    _, ext = os.path.splitext(sub)
                    if ext.lower() in video_extensions:
                        contains_folders_with_videos = True

    if contains_video:
        dir_of_videos_to_frames(path)
    elif contains_folders_with_videos:
        video_dirs_to_frame(path)
    elif only_folders:
        return "folders_only"
    else:
        return "unknown structure"




def set_annotation_id_to_image_id(annotation_path, output_path=None):
    with open(annotation_path, 'r') as f:
        coco = json.load(f)

    for ann in coco['annotations']:
        ann['image_id'] -= 1

    # If no output path is given, overwrite original
    if output_path is None:
        output_path = annotation_path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=4)

    print(f"✅ Annotation IDs set to image_id. Saved to: {output_path}")


def summarize_coco_annotation(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    num_images = len(data.get('images', []))
    num_annotations = len(data.get('annotations', []))
    categories = data.get('categories', [])
    category_id_to_name = {cat['id']: cat['name'] for cat in categories}

    # Count annotations per category
    category_counts = defaultdict(int)
    image_ids = set()
    for ann in data.get('annotations', []):
        cat_id = ann['category_id']
        category_counts[cat_id] += 1
        image_ids.add(ann['image_id'])

    print(f"🖼 Total Images: {num_images}")
    print(f"🔲 Total Bounding Boxes (Annotations): {num_annotations}")
    print(f"🔧 Total Unique Tools (Classes): {len(categories)}")
    print("\n📊 Bounding Box Count per Tool:")
    for cat_id, count in category_counts.items():
        print(f" - {category_id_to_name.get(cat_id, 'Unknown')} (ID {cat_id}): {count} boxes/segmentations")

    print(f"\n🧾 Unique Image IDs in Annotations: {len(image_ids)}")
    if num_images != len(image_ids):
        print("⚠️ Warning: Not all images have annotations!")

    # Prepare data for plotting
    tool_names = [category_id_to_name[cat_id] for cat_id in sorted(category_counts)]
    counts = [category_counts[cat_id] for cat_id in sorted(category_counts)]

    plt.figure(figsize=(12, 6))
    plt.bar(tool_names, counts, color='skyblue')
    plt.xlabel('Tool Classes')
    plt.ylabel('Total Bounding Boxes')
    plt.title('Bounding Box Count per Tool Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()


def mouse_callback(event, x, y, flags, param):
    global start_point, end_point, is_drawing, image_copy, winname, clean_state, segment_by_points

    temp_image = image_copy.copy()  # Always start with a clean copy of the image

    # Draw crosshair lines for the mouse pointer
    cv2.line(temp_image, (x, 0), (x, temp_image.shape[0]), (255, 0, 0), 1)  # Vertical line (y-axis)
    cv2.line(temp_image, (0, y), (temp_image.shape[1], y), (255, 0, 0), 1)  # Horizontal line (x-axis)

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing: Record the starting point
        image_copy = clean_state.copy()
        is_drawing = True
        start_point = [x, y]

    elif event == cv2.EVENT_RBUTTONDOWN:
        image_copy = clean_state.copy()
        is_drawing = True
        
    elif event == cv2.EVENT_MOUSEMOVE:
        # Show the updated image
        if is_drawing:
            # Update the rectangle dynamically as the mouse moves
            end_point = [x, y]
            cv2.rectangle(temp_image, start_point, end_point, (0, 255, 0), 2)

        cv2.imshow(winname, temp_image)
        cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
        # Finish drawing: Record the end point and finalize the bounding box
        is_drawing = False
        end_point = [x, y]
        cv2.rectangle(image_copy, start_point, end_point, (0, 255, 0), 2)

        cv2.imshow(winname, image_copy)
        cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # print(f"Bounding Box: Start={start_point}, End={end_point}")


    # Show the updated image with crosshair
    cv2.imshow(winname, temp_image)
    cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def add_manually_bb(frame_image):
    global image_copy, clean_state, start_point, end_point, winname

    image_copy = frame_image.copy()  # Create a copy for resetting during drawing
    clean_state = frame_image.copy()
    cv2.imshow(winname, frame_image)
    cv2.setMouseCallback(winname, mouse_callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rescale_bbox(bbox, orig_size, new_size):
    """
    Rescale COCO bbox [x_min, y_min, width, height] from original image size to new image size.
    
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


def visualize_random_sample_from_coco(annotation_path, image_dir, tool=None, num_samples=10):
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    # Build image_id to file_name mapping
    image_id_to_file = {img['id']: img['file_name'] for img in data['images']}
    category_mapping = {cat['id']: cat['name'] for cat in data['categories']}
    # Build annotations per image_id
    annotations_by_image = {}
    tool_iamges = {}
    for ann in data['annotations']:
        if tool and category_mapping.get(ann['category_id']) != tool:
            continue
        image_id = ann['image_id']
        tool_iamges[image_id] = image_id_to_file[image_id]
        annotations_by_image.setdefault(image_id, []).append(ann)

    # Select random image_ids
    sampled_image_ids = random.sample(list(image_id_to_file.keys()), min(num_samples, len(image_id_to_file)))

    for image_id in tool_iamges.keys():
        file_name = tool_iamges[image_id]
        image_path = os.path.abspath(os.path.join(image_dir, file_name))
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image not found: {image_path}")
            continue

        anns = annotations_by_image.get(image_id, [])
        for ann in anns:
            # print(f"Ann ID: {ann['id']}, Image ID: {image_id}, Category ID: {ann['category_id']}, Area: {ann['area']}")
            if ann['area'] > 0:  # Tool
                x, y, w, h = map(int, map(np.ceil, ann['bbox']))
                cat_id = ann['category_id']
                category_name = category_mapping.get(cat_id, 'Unknown')

                # Draw bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # Put category name
                cv2.putText(image, category_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1, cv2.LINE_AA)
                
                utils.safe_draw_polygons(image, ann['tool_segmentation'], color=(0, 255, 0))
            
            else:  # Hands
                utils.safe_draw_polygons(image, ann['hand_segmentation'], color=(0, 0, 255))

        # Show image
        window_name = f"Image ID: {image_id} - {file_name}"
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)


def show_saved_results(image, bbox):
    """
    Show the saved image with the bounding box.
    
    Parameters:
        image (numpy.ndarray): The image to display.
        bbox (list or tuple): Bounding box in COCO format [x_min, y_min, width, height].
    """
    x_min, y_min, width, height = map(int, bbox)
    x_max = x_min + width
    y_max = y_min + height

    # Draw the bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    instr = "[ANY] Accept  [q] dont save"
    cv2.putText(image, instr, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 3, cv2.LINE_AA)

    # Show the image
    winname = f'Result - {bbox}'
    cv2.imshow(winname, image)
    # -----------------------------------  Edditing bb
    key = cv2.waitKey(0) & 0xFF  # wait for *one* key
    if key in (ord('q'), 27):  # ESC or 'q' to quit
        cv2.destroyAllWindows()
        return False
        
    return True  # Indicate that the user accepted the bounding box



def check_if_video_is_done(v_path, images):

    image_names = [img['file_name'].split('_')[0] for img in images]
    video_name = os.path.basename(v_path).split('.')[0]

    return video_name in image_names


# ----------------------------------------------------------------------------------------------- Functionallity 3
def prepare_env():
    """
    Prepare the environment for the extracting of bbs from the maintenance action dataset.
    """
    print("[INFO] Preparing environment")

    args = SimpleNamespace(
        directory_prefix=None,
        working_dir=None,
        output_dir="annotation_results",
        image_dir=None,
        annotations=None,
        annotations_output=None,
        db_rows=None,
        tracker={},  # save the last working row id
    )

    print(' # prepare file: ', end='')
    directory_prefix = '/Users/saeednaamneh/Library/CloudStorage/GoogleDrive-yosef.naamneh@gmail.com/My Drive/AR-MAINTENANCE/HAR'
    db_path = os.path.join(directory_prefix, 'maintenance_dataset.db')
    table_name = "MaintenanceActions_metadata" 
    output_dir = "detection_dataset"
    working_dir = 'working_dir'
    os.makedirs(working_dir, exist_ok=True)
    # Parameters (change these as needed)
    output_dir = "TOOLs_dataset"
    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    annotations_output = os.path.join(output_dir, "annotations.json")
    print(f"[DONE] image_dir: {image_dir}, annotations_output: {annotations_output}")

    print(' # read database: ', end='')
    # Connect to DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query your table
    # 'wrench', 'ratchet', 'adjustable spanner', 'hammer', 'tapemeasure', 'allen', 'plier', 'screwdriver', 'cutting knife', 'drill', 'electrical screwdriver'
    cursor.execute(f"""
    SELECT id, video_path, length_, resolution, tool, metadata
    FROM {table_name}
    """)
    rows = cursor.fetchall()
    rows.sort(key=lambda x: x[0])  # Sort by id
    print(f"[DONE] {len(rows)} rows found")
    
    print(' # prepare tracker: ', end='')
    args.directory_prefix = directory_prefix
    args.working_dir = working_dir
    args.output_dir = output_dir
    args.image_dir = image_dir
    args.annotations_output = annotations_output
    args.db_rows = rows

    if os.path.exists(annotations_output):
        with open(annotations_output, 'r') as f:
            args.annotations = json.load(f)
    else:
        args.annotations = {
            "images": [],
            "annotations": [],
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
            ],
        }
    if os.path.exists(os.path.join(output_dir, 'tracker.json')):
        with open(os.path.join(output_dir, 'tracker.json'), 'r') as f:
            args.tracker = json.load(f)
    else:
        args.tracker = {
            'edit': [],
            'error': [],
            'last_id': 0,
            "last_category_id": 11, 
             # Start from 10 to avoid conflicts with existing categories
            }

    args.category_mapping_id = {cat['name']: cat['id'] for cat in args.annotations['categories']}
    print(f"[DONE]")

    # Add any necessary setup code here
    print("[INFO] Environment prepared.")

    return args


def extractor(new_shape=(640, 360)):
    global start_point, end_point, is_drawing, image_copy, clean_state, winname

    args = prepare_env()
    args.tracker['edit'] = []
    args.tracker['error'] = []
    args.tracker['last_row'] = 0
    start_point = None
    end_point = None
    # COCO fields
    images, annotations = args.annotations['images'], args.annotations['annotations']
    categories = [cat['name'] for cat in args.annotations['categories']]
    img_id = args.tracker['last_id']
    ann_id = args.tracker['last_id']
    # for test only: take 5 random videos
    # random_rows = random.sample(args.db_rows, min(20, len(args.db_rows)))
    # print(len(random_rows), "random rows selected from the database")
    for row_idx, row in enumerate(args.db_rows):
        id_, video_path, length_, resolution, tool, meta_json = row
        if tool == "NULL":
            continue
        try:
            meta = json.loads(meta_json)
            tool_bbs = meta.get("tool_bbs", {})
            if not tool_bbs or isinstance(tool_bbs, list):
                args.tracker['edit'].append(id_)
                continue
        except Exception as e:
            print(f"Error parsing metadata for row {id_}: {e}")
            args.tracker['error'].append(id_)
            continue

         # Category mapping
        if tool not in categories:
            if 'electrical' in tool.lower():
                tool = 'drill'
            else:    
                new_category = {
                    "id": args.tracker['last_category_id'],
                    "name": tool,
                    "supercategory": "tool"
                }
                args.annotations['categories'].append(new_category)
                categories.append(tool)
                args.tracker['last_category_id'] += 1
        
        # Video settings
        w, h = map(int, resolution.split("x"))
        cap = cv2.VideoCapture(os.path.join(args.directory_prefix, video_path))
        if not cap.isOpened():
            print(f"Could not open {video_path}")
            continue

        convert_video_to_frame_v1(os.path.join(args.directory_prefix, video_path), args.working_dir)
        # Create a directory for the tool if it doesn't exist
        for frame_str, bb in tool_bbs.items():
            if isinstance(bb[0], list):
                bb = bb[0]
            if not isinstance(bb, list) or len(bb) != 4:
                continue
            take_frame = True
            start_point, end_point = None, None
            # resize and save the frame as .jpg in the target path
            frame_idx = int(frame_str)
            frame_orig_path = os.path.join(args.working_dir, f'{frame_idx:05d}.jpg')
            frame_name = f"{os.path.basename(video_path).split('.')[0]}_{frame_idx:05d}.jpg"
            resized_img = cv2.resize(cv2.imread(frame_orig_path), dsize=new_shape, interpolation=cv2.INTER_LINEAR)
            target_path = os.path.join(args.image_dir, frame_name)

            # Convert bb to COCO format [x, y, w, h]
            x_c, y_c, bw, bh = bb
            x_c *= w 
            y_c *= h
            bw *= w
            bh *= h

            bb = [x_c, y_c, bw, bh]
            # Convert to top-left (x1, y1) and bottom-right (x2, y2)
            x1 = int(bb[0] - bb[2] / 2)
            y1 = int(bb[1] - bb[3] / 2)
            x2 = int(bb[0] + bb[2] / 2)
            y2 = int(bb[1] + bb[3] / 2)
            new_w = abs(x2 - x1)
            new_h = abs(y2 - y1)

            # Read image
            img = cv2.imread(frame_orig_path)
            showing_img = img.copy()

            # Draw the rectangle
            cv2.rectangle(showing_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green box, thickness 2
            
            # --------------------------------------------------  instructions
            instr = "[a/Enter] Accept   [e] Edit   [q] Quit"
            cv2.putText(showing_img, instr, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 0), 3, cv2.LINE_AA)


             # Show the image
            winname = f'v: {row_idx + 1}/{len(args.db_rows)} - frame- {frame_idx} - {tool}'
            cv2.imshow(winname, showing_img)
            cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            # -----------------------------------  Edditing bb
            key = cv2.waitKey(0) & 0xFF  # wait for *one* key
            if key in (ord('q'), 27):
                take_frame = False
                cv2.destroyAllWindows()
                break
            
            if key in (ord('e'),):  # EDIT BB
                cv2.destroyAllWindows()
                # winname = f'{img_info['file_name']} -- EDITTING'
                add_manually_bb(img)

                if start_point and end_point:
                    x1, y1 = start_point
                    x2, y2 = end_point
                    new_w, new_h = abs(x2 - x1), abs(y2 - y1)
                
                else:
                    print("No bounding box drawn. Skipping this frame.")
                    take_frame = False
                    
            
            if take_frame:
                cv2.imwrite(target_path, resized_img)
                bb = rescale_bbox([x1, y1, x2, y2], (w, h), new_shape)
                save_it = show_saved_results(resized_img, bb)

                if save_it:
                    print(f"Saving frame {frame_name} with bounding box: {bb}")

                    images.append({
                        "id": img_id,
                        "file_name": frame_name,
                        "width": new_shape[0],
                        "height": new_shape[1]
                    })
                    # Save the annotation
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": args.category_mapping_id[tool],  # Use the last added category
                        "bbox":bb,
                        "area": bb[-1] * bb[-2],
                        "iscrowd": 0,
                        "segmentation": [],
                    })
                    ann_id += 1
                    img_id += 1
                    args.tracker['last_id'] = img_id
                    coco_json = {
                            "images": images,
                            "annotations": annotations,
                            "categories": args.annotations['categories']
                        }
                    with open(args.annotations_output, "w") as f:
                        json.dump(coco_json, f, indent=4)
                    
                    with open(os.path.join(args.output_dir, 'tracker.json'), 'w') as f:
                        json.dump(args.tracker, f, indent=4)
        
            cv2.destroyAllWindows()


def visualize_hand_segments(image_path, annotation_path, color=(255, 51, 51) , alpha=0.4, line_thickness=2):
    """
    Draws hand segmentation polygons from a RoboFlow-style annotation onto the image.

    Parameters:
        image_path (str): Path to the image file (e.g., .jpg or .png)
        annotation_path (str): Path to the JSON file with polygon annotations
        color (tuple): RGB color for the polygon (default yellow)
        alpha (float): Transparency of the fill color
        line_thickness (int): Thickness of the polygon border
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load annotation
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    overlay = image.copy()

    # Draw each polygon
    for box in data.get("boxes", []):
        if box.get("label") == "hands" and box.get("type") == "polygon":
            points = np.array(box["points"], dtype=np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.polylines(overlay, [points], isClosed=True, color=color, thickness=line_thickness)
            cv2.fillPoly(overlay, [points], color=color)

    # Blend with transparency
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    image_name = os.path.split(image_path)[-1].split('.')[0]
    cv2.imwrite(os.path.join("video_for_paper_mmtl/hand_annotations", f"{image_name}_maks.png"), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    


def introduction():
    print("This script contains multiple functionalities:")
    print("1. Convert videos in a directory to frames and save them as images.")
    print("2. Rename the images files (also update the annotations.json).")
    print("3. Extract bbs as coco formart from The Maintenance Action Dataset")
    print("4. Visualze random samples from a coco dataset")
    print("5. Summarize a COCO annotation file")
    print('')
    input_choice = input("Choose [1/2/3/4/5]: ")
    if input_choice == '1':
        video_dir = utils.input_with_path_completion("Enter the path to the video directory: ")
        functionality_one_navigator(video_dir)
    elif input_choice == '2':
        video_dir = utils.input_with_path_com4pletion("Enter the path to dir containing images and annotations.json: ")
        rename_images_to_numeric_order(video_dir)
    elif input_choice == '3':
        extractor()
    elif input_choice == '4':
        root_file = utils.input_with_path_completion("Enter the path to the root file [should contain images, annotations.json]: ")
        image_dir = os.path.join(root_file, "images")
        annotation_path = os.path.join(root_file, "annotations.json")
        tool = None if input("Enter the tool name to visualize (or leave empty for random sample): ") == "" else input("Enter the tool name to visualize (or leave empty for random sample): ") 
        visualize_random_sample_from_coco(annotation_path, image_dir, tool=tool)
    elif input_choice == '5':
        json_path = utils.input_with_path_completion("Enter the path to the COCO annotation file (.json): ")
        summarize_coco_annotation(json_path)
    else:
        print("Invalid choice. Please try again.")



if __name__ == "__main__":
    import argparse
    introduction()
    