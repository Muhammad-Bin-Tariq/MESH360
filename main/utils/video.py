import cv2
import os
import subprocess
import math
import io
import json
from rembg import remove, new_session
import numpy as np
from PIL import Image

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
        "2",
        "--overwrite",
    ]

    print(f"\nRunning colmap2nerf with fps={fps} to extract ~{target_frames} images...")
    print(f"Command: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    transforms_path = os.path.join(repo_root, "transforms.json")
    _replace_transforms_image_extensions_to_png(transforms_path)

    return fps


def _replace_transforms_image_extensions_to_png(transforms_path):
    """Replace .jpg/.jpeg frame paths with .png in transforms.json."""
    if not os.path.isfile(transforms_path):
        raise FileNotFoundError(f"transforms.json not found at '{transforms_path}'.")

    with open(transforms_path, "r", encoding="utf-8") as transforms_file:
        transforms_data = json.load(transforms_file)

    frames = transforms_data.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"Invalid transforms format in '{transforms_path}': 'frames' must be a list.")

    updated_count = 0
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        file_path = frame.get("file_path")
        if not isinstance(file_path, str):
            continue

        file_root, file_ext = os.path.splitext(file_path)
        if file_ext.lower() in (".jpg", ".jpeg"):
            frame["file_path"] = f"{file_root}.png"
            updated_count += 1

    with open(transforms_path, "w", encoding="utf-8") as transforms_file:
        json.dump(transforms_data, transforms_file, indent=2)
        transforms_file.write("\n")

    print(f"Updated {updated_count} frame paths to .png in '{transforms_path}'.")


def _enhance_image_for_processing(
    image_bgr,
    contrast_alpha,
    contrast_beta,
    saturation_scale,
    sharpen_strength,
):
    """Boost contrast/colors and sharpen details before rembg."""
    contrasted_bgr = cv2.convertScaleAbs(image_bgr, alpha=contrast_alpha, beta=contrast_beta)

    hsv_image = cv2.cvtColor(contrasted_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_scale, 0, 255)
    saturated_bgr = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

    blurred_bgr = cv2.GaussianBlur(saturated_bgr, (0, 0), sigmaX=1.0, sigmaY=1.0)
    sharpened_bgr = cv2.addWeighted(
        saturated_bgr,
        1.0 + sharpen_strength,
        blurred_bgr,
        -sharpen_strength,
        0,
    )
    return sharpened_bgr


def _resolve_rembg_model_name(model_name):
    """Map legacy BRIA model aliases to rembg session names."""
    model_aliases = {
        "bria-rmbg-1.4": "bria-rmbg",
        "bria-rmbg-2.0": "bria-rmbg",
    }
    resolved_name = model_aliases.get(model_name, model_name)
    if resolved_name != model_name:
        print(f"Using rembg model '{resolved_name}' for requested '{model_name}'.")
    return resolved_name


def _initialize_rembg_session(model_name, available_providers):
    """
    Initialize rembg session with CUDA first, then explicit CPU fallback.

    This avoids hard failures when onnxruntime reports CUDA provider availability
    but runtime libs (e.g., libcudnn) are missing.
    """
    provider_attempts = []
    if "CUDAExecutionProvider" in available_providers:
        provider_attempts.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available_providers:
        provider_attempts.append("CPUExecutionProvider")

    if not provider_attempts:
        raise RuntimeError(
            "No supported onnxruntime providers found. "
            "Expected CUDAExecutionProvider or CPUExecutionProvider."
        )

    if "CUDAExecutionProvider" in provider_attempts:
        try:
            import ctypes

            ctypes.CDLL("libcudnn.so.9")
        except OSError as exc:
            print(
                "CUDAExecutionProvider is present but CUDA runtime is incomplete "
                f"({exc}). Falling back to CPUExecutionProvider."
            )
            provider_attempts = [provider for provider in provider_attempts if provider != "CUDAExecutionProvider"]

    if not provider_attempts:
        raise RuntimeError(
            "CUDAExecutionProvider is unavailable at runtime and CPUExecutionProvider "
            "is not available."
        )

    last_provider_error = None
    for provider_name in provider_attempts:
        try:
            session = new_session(model_name, providers=[provider_name])
            return session, provider_name
        except ValueError:
            raise
        except Exception as exc:
            last_provider_error = exc
            if provider_name == "CUDAExecutionProvider" and "CPUExecutionProvider" in provider_attempts:
                print(
                    f"Failed to initialize CUDAExecutionProvider ({exc}). "
                    "Falling back to CPUExecutionProvider."
                )
                continue
            raise RuntimeError(
                f"Failed to initialize rembg session with {provider_name}: {exc}"
            ) from exc

    raise RuntimeError(f"Failed to initialize rembg session: {last_provider_error}") from last_provider_error


def _resolve_yolo_model_path(yolo_model_path):
    """Resolve YOLO model path from cwd or repo root."""
    if os.path.isabs(yolo_model_path) and os.path.isfile(yolo_model_path):
        return yolo_model_path
    if os.path.isfile(yolo_model_path):
        return os.path.abspath(yolo_model_path)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidate_path = os.path.join(repo_root, yolo_model_path)
    if os.path.isfile(candidate_path):
        return candidate_path

    raise FileNotFoundError(f"YOLO model file '{yolo_model_path}' was not found.")


def _select_primary_bbox_xyxy(yolo_result):
    """Select one primary YOLO bbox as [x1, y1, x2, y2]."""
    boxes = getattr(yolo_result, "boxes", None)
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return None

    boxes_xyxy = boxes.xyxy.detach().cpu().numpy()
    if boxes_xyxy.size == 0:
        return None

    if boxes.conf is not None:
        confidences = boxes.conf.detach().cpu().numpy()
    else:
        confidences = np.ones(len(boxes_xyxy), dtype=np.float32)

    widths = np.clip(boxes_xyxy[:, 2] - boxes_xyxy[:, 0], 0.0, None)
    heights = np.clip(boxes_xyxy[:, 3] - boxes_xyxy[:, 1], 0.0, None)
    areas = widths * heights
    best_index = int(np.argmax(confidences * np.maximum(areas, 1e-6)))
    return boxes_xyxy[best_index].astype(np.float64)


def _bbox_from_alpha_mask(image_path):
    """Fallback bbox from non-zero alpha pixels on rembg output."""
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        return None, None, None

    image_height, image_width = image.shape[:2]
    alpha = image[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if xs.size == 0 or ys.size == 0:
        return None, image_width, image_height

    bbox_xyxy = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)
    return bbox_xyxy, image_width, image_height


def _normalize_pixel_coord(pixel_value, image_size, aabb_scale):
    """Normalize pixel coordinate to instant-ngp coordinate space."""
    return (float(pixel_value) / float(image_size) - 0.5) * float(aabb_scale)


def add_yolo_render_aabb_from_first_last_frames(
    image_paths,
    transforms_path,
    yolo_model_path="yolo11n-seg.pt",
    conf=0.25,
):
    """
    Use YOLOv8 on first/last frame and inject render_aabb crop into transforms.json.

    Pixel bbox values are normalized as:
    (pixel / image_size - 0.5) * aabb_scale
    """
    if not image_paths:
        raise ValueError("No image paths were provided for YOLO render_aabb extraction.")
    if not os.path.isfile(transforms_path):
        raise FileNotFoundError(f"transforms.json not found at '{transforms_path}'.")
    if conf < 0 or conf > 1:
        raise ValueError("conf must be in [0, 1].")

    with open(transforms_path, "r", encoding="utf-8") as transforms_file:
        transforms_data = json.load(transforms_file)

    aabb_scale = transforms_data.get("aabb_scale")
    if not isinstance(aabb_scale, (int, float)) or aabb_scale <= 0:
        raise ValueError(
            f"Invalid or missing aabb_scale in '{transforms_path}'. "
            "Expected a positive number."
        )

    sorted_image_paths = sorted(image_paths)
    first_frame_path = sorted_image_paths[0]
    last_frame_path = sorted_image_paths[-1]
    selected_paths = [first_frame_path] if first_frame_path == last_frame_path else [first_frame_path, last_frame_path]

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required for YOLO-based render_aabb extraction.") from exc

    resolved_model_path = _resolve_yolo_model_path(yolo_model_path)
    yolo_model = YOLO(resolved_model_path)
    yolo_results = yolo_model.predict(source=selected_paths, conf=conf, verbose=False)

    normalized_x = []
    normalized_y = []
    for image_path, yolo_result in zip(selected_paths, yolo_results):
        bbox_xyxy = _select_primary_bbox_xyxy(yolo_result)
        image_width = None
        image_height = None
        if hasattr(yolo_result, "orig_shape") and yolo_result.orig_shape is not None:
            image_height, image_width = yolo_result.orig_shape

        if bbox_xyxy is None:
            fallback_bbox, fallback_width, fallback_height = _bbox_from_alpha_mask(image_path)
            if fallback_bbox is not None:
                bbox_xyxy = fallback_bbox
                image_width = fallback_width
                image_height = fallback_height

        if bbox_xyxy is None:
            raise ValueError(f"YOLO could not detect an object bbox for frame '{image_path}'.")
        if image_width is None or image_height is None:
            raise ValueError(f"Could not resolve image dimensions for frame '{image_path}'.")

        x1, y1, x2, y2 = bbox_xyxy.tolist()
        x1 = float(np.clip(x1, 0.0, image_width - 1))
        y1 = float(np.clip(y1, 0.0, image_height - 1))
        x2 = float(np.clip(x2, 0.0, image_width - 1))
        y2 = float(np.clip(y2, 0.0, image_height - 1))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid YOLO bbox for frame '{image_path}': {[x1, y1, x2, y2]}")

        normalized_x.extend(
            [
                _normalize_pixel_coord(x1, image_width, aabb_scale),
                _normalize_pixel_coord(x2, image_width, aabb_scale),
            ]
        )
        normalized_y.extend(
            [
                _normalize_pixel_coord(y1, image_height, aabb_scale),
                _normalize_pixel_coord(y2, image_height, aabb_scale),
            ]
        )

    render_aabb_min = [min(normalized_x), min(normalized_y), -0.5 * float(aabb_scale)]
    render_aabb_max = [max(normalized_x), max(normalized_y), 0.5 * float(aabb_scale)]
    transforms_data["render_aabb"] = [render_aabb_min, render_aabb_max]

    with open(transforms_path, "w", encoding="utf-8") as transforms_file:
        json.dump(transforms_data, transforms_file, indent=2)
        transforms_file.write("\n")

    print(
        f"Updated render_aabb in '{transforms_path}' from YOLO first/last frames: "
        f"min={render_aabb_min}, max={render_aabb_max}"
    )
    return transforms_data["render_aabb"]


def contrast_segment_and_remove_background(
    image_dir,
    yolo_model_path="yolo11n-seg.pt",
    rembg_model_name="bria-rmbg",
    contrast_alpha=1.6,
    contrast_beta=8,
    saturation_scale=1.65,
    sharpen_strength=1.0,
    conf=0.25,
):
    """
    Apply stronger enhancement, then BRIA rembg background removal.

    `yolo_model_path` and `conf` are kept for backward compatibility with older
    call sites, but YOLO segmentation is intentionally skipped to match
    `rembg p -m bria-rmbg` style behavior.
    """
    if not os.path.isdir(image_dir):
        raise ValueError(f"Image directory '{image_dir}' does not exist.")
    if saturation_scale <= 0:
        raise ValueError("saturation_scale must be greater than 0.")
    if sharpen_strength < 0:
        raise ValueError("sharpen_strength must be non-negative.")

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_files = sorted(
        [
            file_name
            for file_name in os.listdir(image_dir)
            if file_name.lower().endswith(image_extensions)
            and os.path.isfile(os.path.join(image_dir, file_name))
        ]
    )
    if not image_files:
        raise ValueError(f"No images found in '{image_dir}'.")


    import onnxruntime as ort

    available_providers = ort.get_available_providers()
    resolved_rembg_model_name = _resolve_rembg_model_name(rembg_model_name)
    try:
        rembg_session, active_provider = _initialize_rembg_session(
            resolved_rembg_model_name,
            available_providers=available_providers,
        )
    except ValueError as exc:
        raise ValueError(
            f"No session class found for model '{rembg_model_name}'. "
            "For BRIA, use 'bria-rmbg' with this rembg version."
        ) from exc
    processed_count = 0
    output_paths = []

    for file_name in image_files:
        image_path = os.path.join(image_dir, file_name)
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        enhanced_bgr = _enhance_image_for_processing(
            image_bgr=image_bgr,
            contrast_alpha=contrast_alpha,
            contrast_beta=contrast_beta,
            saturation_scale=saturation_scale,
            sharpen_strength=sharpen_strength,
        )

        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        enhanced_image = Image.fromarray(enhanced_rgb)
        image_buffer = io.BytesIO()
        enhanced_image.save(image_buffer, format="PNG")
        rembg_bytes = remove(
            image_buffer.getvalue(),
            session=rembg_session,
            force_return_bytes=True,
        )

        if not isinstance(rembg_bytes, (bytes, bytearray)):
            raise TypeError(f"Unexpected rembg output type: {type(rembg_bytes)}")

        output_image = Image.open(io.BytesIO(rembg_bytes)).convert("RGBA")
        output_path = os.path.splitext(image_path)[0] + ".png"
        output_image.save(output_path, format="PNG")

        if output_path != image_path and image_path.lower().endswith((".jpg", ".jpeg")):
            os.remove(image_path)

        processed_count += 1
        output_paths.append(output_path)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    transforms_path = os.path.join(repo_root, "transforms.json")
    add_yolo_render_aabb_from_first_last_frames(
        image_paths=output_paths,
        transforms_path=transforms_path,
        yolo_model_path=yolo_model_path,
        conf=conf,
    )

    print(
        f"Processed {processed_count} images in '{image_dir}' "
        f"with stronger enhancement + BRIA rembg ({active_provider})."
    )
    return processed_count
