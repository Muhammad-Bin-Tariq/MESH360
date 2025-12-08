import os
from utils.video import get_video_length




def main():
    # Example: Get video length from a video file
    video_path = "main/data/videos/office_chair.mp4"  # Change this to your video path
    # Get video length
    duration = get_video_length(video_path)
    
    if duration is not None:
        print(f"Video length: {duration:.2f} seconds")
    else:
        print("Failed to get video length.")


if __name__ == "__main__":
    main()
