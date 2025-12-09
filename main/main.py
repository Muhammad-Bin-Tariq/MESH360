import os
from utils.video import get_video_length
from utils.video import extract_frames_based_on_time 
from utils.video import background_remover
from utils.video import yolo_segment_and_remove_background

def main():
    # Example: Get video length from a video file
    video_path = "main/data/videos/office_chair.mp4"  # Change this to your video path
    image_output_dir = "main/data/videos/images"
    # Get video length
    duration = get_video_length(video_path)
    
    if duration is not None:
        print(f"Video length: {duration:.2f} seconds")
    else:
        print("Failed to get video length.")

    # Extract 150 frames from the video
    extract_frames_based_on_time(video_path, target_frames=120)

    # YOLOv11 segmentation to remove background from images
    yolo_segment_and_remove_background(image_output_dir)


if __name__ == "__main__":
    main()
