import cv2
import os


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


def extract_frames_based_on_time(video_path, interval_seconds):
    pass