"""
Video Processor - Handles video preprocessing tasks like extracting frames and audio
"""

import os
import cv2
import logging
import subprocess
import numpy as np
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Class for processing video files, extracting frames and audio.
    """
    
    def __init__(self, output_dir=None, frame_rate=1, temp_dir=None):
        """
        Initialize the VideoProcessor.
        
        Args:
            output_dir (str, optional): Directory to save processed files. Defaults to a temp directory.
            frame_rate (float, optional): Frame rate for extraction. Defaults to 1 fps.
            temp_dir (str, optional): Directory for temporary files. Defaults to system temp.
        """
        self.output_dir = Path(output_dir) if output_dir else Path("cookify/data/processed")
        self.frame_rate = frame_rate
        self.temp_dir = Path(temp_dir) if temp_dir else Path("cookify/data/temp")
        
        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def process(self, video_path):
        """
        Process a video file, extracting frames and audio.
        
        Args:
            video_path (str): Path to the video file.
            
        Returns:
            tuple: (frames, audio_path, metadata)
                - frames: List of extracted frames as numpy arrays
                - audio_path: Path to the extracted audio file
                - metadata: Dictionary containing video metadata
        """
        logger.info(f"Processing video: {video_path}")
        
        # Create video-specific output directories
        video_name = Path(video_path).stem
        video_output_dir = self.output_dir / video_name
        frames_dir = video_output_dir / "frames"
        os.makedirs(frames_dir, exist_ok=True)
        
        # Extract metadata
        metadata = self._extract_metadata(video_path)
        logger.info(f"Video metadata: {metadata}")
        
        # Extract frames
        frames = self._extract_frames(video_path, frames_dir)
        logger.info(f"Extracted {len(frames)} frames")
        
        # Extract audio
        audio_path = self._extract_audio(video_path, video_output_dir)
        logger.info(f"Extracted audio to {audio_path}")
        
        return frames, audio_path, metadata
    
    def _extract_metadata(self, video_path):
        """
        Extract metadata from a video file.
        
        Args:
            video_path (str): Path to the video file.
            
        Returns:
            dict: Dictionary containing video metadata
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Extract basic metadata
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        # Use ffprobe to get more detailed metadata
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            ffprobe_data = result.stdout
            
            # Parse ffprobe output if available
            if ffprobe_data:
                import json
                ffprobe_json = json.loads(ffprobe_data)
                # Extract additional metadata if needed
            
        except Exception as e:
            logger.warning(f"Could not extract detailed metadata with ffprobe: {e}")
            ffprobe_json = {}
        
        metadata = {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "path": video_path,
            "ffprobe": ffprobe_json
        }
        
        return metadata
    
    def _extract_frames(self, video_path, output_dir):
        """
        Extract frames from a video file.
        
        Args:
            video_path (str): Path to the video file.
            output_dir (Path): Directory to save extracted frames.
            
        Returns:
            list: List of extracted frames as numpy arrays
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame extraction interval
        interval = int(fps / self.frame_rate)
        interval = max(1, interval)  # Ensure interval is at least 1
        
        frames = []
        frame_idx = 0
        
        with tqdm(total=frame_count, desc="Extracting frames") as pbar:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Extract frame at specified interval
                if frame_idx % interval == 0:
                    # Save frame to disk
                    frame_path = output_dir / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    
                    # Add frame to list
                    frames.append(frame)
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        return frames
    
    def _extract_audio(self, video_path, output_dir):
        """
        Extract audio from a video file.
        
        Args:
            video_path (str): Path to the video file.
            output_dir (Path): Directory to save extracted audio.
            
        Returns:
            Path: Path to the extracted audio file
        """
        audio_path = output_dir / "audio.wav"
        
        # Use ffmpeg to extract audio
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-q:a", "0",
            "-map", "a",
            "-y",  # Overwrite output file if it exists
            str(audio_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except FileNotFoundError:
            logger.error("ffmpeg not found. Please install ffmpeg and make sure it's in your PATH.")
            logger.warning("Continuing without audio extraction.")
            # Create an empty file as a placeholder
            with open(audio_path, 'w') as f:
                f.write("# Audio extraction skipped - ffmpeg not installed")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error extracting audio: {e}")
            logger.error(f"ffmpeg stderr: {e.stderr.decode()}")
            logger.warning("Continuing without audio extraction.")
            # Create an empty file as a placeholder
            with open(audio_path, 'w') as f:
                f.write("# Audio extraction failed")
        
        return audio_path
