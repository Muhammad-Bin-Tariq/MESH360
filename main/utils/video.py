import cv2
import os
import subprocess
import math
from rembg import remove, new_session
import numpy as np

def get_video_length(video_path):
    """
    Get the length of a video in seconds.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        float: Video length in seconds, or None if error occurs
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return None
    
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file '{video_path}'.")
            return None
        
        # Get the frames per second (fps)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Get the total number of frames
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        # Calculate duration in seconds
        duration = frame_count / fps if fps > 0 else 0
        
        # Release the video capture object
        cap.release()
        
        return duration
    
    except Exception as e:
        print(f"Error processing video: {e}")
        return None


def extract_frames_based_on_time(video_path, target_frames):
    """
    Run colmap2nerf on a video, choosing fps to hit a target image count.

    Args:
        video_path (str): Path to the input video.
        target_frames (int): Desired number of output images.

    Returns:
        float: The fps value used for extraction.
    """
    duration = get_video_length(video_path)
    if not duration or duration <= 0:
        raise ValueError("Video duration could not be determined.")

    # Calculate fps: for 17 second video with 150 target frames = 150/17 ≈ 8.82 fps
    fps = target_frames / duration
    
    # Round up to nearest whole number
    
    fps = math.ceil(fps)
    
    print(f"Video duration: {duration:.2f} seconds")
    print(f"Target frames: {target_frames}")
    print(f"Calculated fps: {fps}")

    cmd = [
        "python",
        "./scripts/colmap2nerf.py",
        "--video_in",
        video_path,
        "--video_fps",
        str(fps),
        "--run_colmap",
        "--aabb_scale",
        "32",
        "--overwrite",
    ]

    print(f"\nRunning colmap2nerf with fps={fps} to extract ~{target_frames} images...")
    print(f"Command: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

    return fps


def yolo_segment_and_remove_background(image_path):
    """Use YOLOv11 detections plus rembg to isolate objects into PNGs."""

    if not os.path.isdir(image_path):
        print(f"Error: Directory '{image_path}' not found.")
        return

    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    image_files = [f for f in os.listdir(image_path) if f.lower().endswith(image_extensions)]

    if not image_files:
        print(f"No images found in '{image_path}'.")
        return

    output_dir = os.path.join("main", "data", "videos", "yolo_images")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {len(image_files)} images with YOLOv11 + rembg...")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing ultralytics for YOLO detection...")
        subprocess.run(["pip", "install", "ultralytics"], check=True)
        from ultralytics import YOLO

    yolo_model = YOLO("yolo11x.pt")

    try:
        session = new_session("u2net", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    except Exception as exc:
        print(f"Warning: CUDA session failed ({exc}). Falling back to default provider.")
        session = new_session("u2net")

    for idx, filename in enumerate(image_files, 1):
        src_path = os.path.join(image_path, filename)

        original = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        if original is None:
            print(f"Warning: Unable to read '{filename}', skipping.")
            continue

        if original.ndim == 2:
            det_input = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        elif original.shape[2] == 4:
            det_input = cv2.cvtColor(original, cv2.COLOR_BGRA2BGR)
        else:
            det_input = original.copy()

        detections = yolo_model(det_input, verbose=False)[0]
        boxes = detections.boxes

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            scores = conf * areas
            best_idx = int(np.argmax(scores))
            x1, y1, x2, y2 = xyxy[best_idx]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            margin_x = int((x2 - x1) * 0.1)
            margin_y = int((y2 - y1) * 0.1)
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(det_input.shape[1], x2 + margin_x)
            y2 = min(det_input.shape[0], y2 + margin_y)

            if x2 <= x1 or y2 <= y1:
                x1, y1, x2, y2 = 0, 0, det_input.shape[1], det_input.shape[0]
        else:
            x1, y1, x2, y2 = 0, 0, det_input.shape[1], det_input.shape[0]

        roi = det_input[y1:y2, x1:x2]
        if roi.size == 0:
            print(f"Warning: Empty ROI for '{filename}', skipping.")
            continue

        success, buffer = cv2.imencode(".png", roi)
        if not success:
            print(f"Warning: Could not encode ROI for '{filename}', skipping.")
            continue

        processed_bytes = remove(
            buffer.tobytes(),
            session=session,
            only_mask=False,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
        )

        processed_array = np.frombuffer(processed_bytes, dtype=np.uint8)
        roi_result = cv2.imdecode(processed_array, cv2.IMREAD_UNCHANGED)
        if roi_result is None:
            print(f"Warning: Could not decode processed ROI for '{filename}', skipping.")
            continue

        if roi_result.ndim == 2:
            roi_result = cv2.cvtColor(roi_result, cv2.COLOR_GRAY2BGRA)
        elif roi_result.shape[2] == 3:
            roi_result = cv2.cvtColor(roi_result, cv2.COLOR_BGR2BGRA)

        if roi_result.shape[0] != (y2 - y1) or roi_result.shape[1] != (x2 - x1):
            roi_result = cv2.resize(roi_result, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)

        rgb_full = np.zeros((det_input.shape[0], det_input.shape[1], 3), dtype=np.float32)
        alpha_full = np.zeros((det_input.shape[0], det_input.shape[1]), dtype=np.float32)

        roi_rgb = roi_result[:, :, :3].astype(np.float32)
        roi_alpha = roi_result[:, :, 3].astype(np.float32) / 255.0

        rgb_full[y1:y2, x1:x2] = roi_rgb
        alpha_full[y1:y2, x1:x2] = roi_alpha

        output_rgba = np.zeros((det_input.shape[0], det_input.shape[1], 4), dtype=np.uint8)
        output_rgba[:, :, :3] = np.clip(rgb_full, 0, 255).astype(np.uint8)
        output_rgba[:, :, 3] = np.clip(alpha_full * 255.0, 0, 255).astype(np.uint8)

        base_name, _ = os.path.splitext(filename)
        dest_path = os.path.join(output_dir, f"{base_name}.png")
        cv2.imwrite(dest_path, output_rgba)

        if idx % 10 == 0 or idx == len(image_files):
            print(f"Processed {idx}/{len(image_files)} images")

    print(f"Segmentation complete. Results saved to {output_dir}.")



# def background_remover(image_path):
#     """Detect objects with YOLO, clean backgrounds via rembg, and overwrite images."""

#     if not os.path.isdir(image_path):
#         print(f"Error: Directory '{image_path}' not found.")
#         return

#     image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
#     image_files = [f for f in os.listdir(image_path) if f.lower().endswith(image_extensions)]

#     if not image_files:
#         print(f"No images found in '{image_path}'.")
#         return

#     print(f"Processing {len(image_files)} images with YOLO-assisted rembg (GPU)...")

#     try:
#         from ultralytics import YOLO
#     except ImportError:
#         print("Installing ultralytics for YOLO detection...")
#         subprocess.run(["pip", "install", "ultralytics"], check=True)
#         from ultralytics import YOLO

#     yolo_model = YOLO("yolov8x.pt")

#     try:
#         session = new_session("u2net", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
#     except Exception as exc:
#         print(f"Warning: CUDA session failed ({exc}). Falling back to default provider.")
#         session = new_session("u2net")

#     for idx, filename in enumerate(image_files, 1):
#         src_path = os.path.join(image_path, filename)

#         original = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
#         if original is None:
#             print(f"Warning: Unable to read '{filename}', skipping.")
#             continue

#         if original.ndim == 2:
#             det_input = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
#         elif original.shape[2] == 4:
#             det_input = cv2.cvtColor(original, cv2.COLOR_BGRA2BGR)
#         else:
#             det_input = original.copy()

#         detections = yolo_model(det_input, verbose=False)[0]
#         boxes = detections.boxes

#         if boxes is not None and len(boxes) > 0:
#             xyxy = boxes.xyxy.cpu().numpy()
#             conf = boxes.conf.cpu().numpy()
#             areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
#             scores = conf * areas
#             best_idx = int(np.argmax(scores))
#             x1, y1, x2, y2 = xyxy[best_idx]
#             x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

#             margin_x = int((x2 - x1) * 0.1)
#             margin_y = int((y2 - y1) * 0.1)
#             x1 = max(0, x1 - margin_x)
#             y1 = max(0, y1 - margin_y)
#             x2 = min(det_input.shape[1], x2 + margin_x)
#             y2 = min(det_input.shape[0], y2 + margin_y)

#             if x2 <= x1 or y2 <= y1:
#                 x1, y1, x2, y2 = 0, 0, det_input.shape[1], det_input.shape[0]
#         else:
#             x1, y1, x2, y2 = 0, 0, det_input.shape[1], det_input.shape[0]

#         roi = det_input[y1:y2, x1:x2]
#         if roi.size == 0:
#             print(f"Warning: Empty ROI for '{filename}', skipping.")
#             continue

#         success, buffer = cv2.imencode(".png", roi)
#         if not success:
#             print(f"Warning: Could not encode ROI for '{filename}', skipping.")
#             continue

#         processed_bytes = remove(
#             buffer.tobytes(),
#             session=session,
#             only_mask=False,
#             alpha_matting=True,
#             alpha_matting_foreground_threshold=240,
#             alpha_matting_background_threshold=10,
#         )

#         processed_array = np.frombuffer(processed_bytes, dtype=np.uint8)
#         roi_result = cv2.imdecode(processed_array, cv2.IMREAD_UNCHANGED)
#         if roi_result is None:
#             print(f"Warning: Could not decode processed ROI for '{filename}', skipping.")
#             continue

#         if roi_result.ndim == 2:
#             roi_result = cv2.cvtColor(roi_result, cv2.COLOR_GRAY2BGRA)
#         elif roi_result.shape[2] == 3:
#             roi_result = cv2.cvtColor(roi_result, cv2.COLOR_BGR2BGRA)

#         if roi_result.shape[0] != (y2 - y1) or roi_result.shape[1] != (x2 - x1):
#             roi_result = cv2.resize(roi_result, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)

#         rgb_full = np.zeros((det_input.shape[0], det_input.shape[1], 3), dtype=np.float32)
#         alpha_full = np.zeros((det_input.shape[0], det_input.shape[1]), dtype=np.float32)

#         roi_rgb = roi_result[:, :, :3].astype(np.float32)
#         roi_alpha = roi_result[:, :, 3].astype(np.float32) / 255.0

#         rgb_full[y1:y2, x1:x2] = roi_rgb
#         alpha_full[y1:y2, x1:x2] = roi_alpha

#         output_rgba = np.zeros((det_input.shape[0], det_input.shape[1], 4), dtype=np.uint8)
#         output_rgba[:, :, :3] = np.clip(rgb_full, 0, 255).astype(np.uint8)
#         output_rgba[:, :, 3] = np.clip(alpha_full * 255.0, 0, 255).astype(np.uint8)

#         base_name, _ = os.path.splitext(filename)
#         dest_path = os.path.join(image_path, f"{base_name}.png")
#         cv2.imwrite(dest_path, output_rgba)

#         if dest_path != src_path and os.path.exists(src_path):
#             try:
#                 os.remove(src_path)
#             except OSError:
#                 pass

#         if idx % 10 == 0 or idx == len(image_files):
#             print(f"Processed {idx}/{len(image_files)} images")

#     print("Background removal complete.")