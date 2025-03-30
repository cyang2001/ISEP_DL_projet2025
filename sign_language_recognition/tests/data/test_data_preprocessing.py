"""
Tests for the data preprocessing module.
"""
import os
import shutil
import tempfile
import unittest
from unittest import mock
import numpy as np

import pytest
from omegaconf import OmegaConf

from src.data.data_preprocessor import VideoPreprocessor
from src.data_preprocessing import main


class TestVideoPreprocessor(unittest.TestCase):
    """Test cases for VideoPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.processed_dir = os.path.join(self.test_dir, "processed")
        
        # Create a mock configuration
        self.cfg = OmegaConf.create({
            "data": {
                "raw_dir": self.test_dir,
                "processed_dir": self.processed_dir,
                "video": {
                    "resize_shape": [64, 64],
                    "num_frames": 16,
                    "fps": 25,
                    "normalize": True
                }
            }
        })
        
        # Create mock logger
        self.mock_logger = mock.MagicMock()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    @mock.patch('cv2.VideoCapture')
    def test_preprocess_video(self, mock_video_capture):
        """Test video preprocessing functionality."""
        # Setup mock video capture
        mock_cap = mock.MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 30,
            cv2.CAP_PROP_FPS: 30
        }.get(prop, 0)
        
        # Mock reading frames
        mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, mock_frame)
        mock_video_capture.return_value = mock_cap
        
        # Initialize preprocessor and process a video
        preprocessor = VideoPreprocessor(self.cfg, self.mock_logger)
        test_video_path = os.path.join(self.test_dir, "test_video.mp4")
        
        # Create an empty file to satisfy os.path.exists check
        with open(test_video_path, 'w') as f:
            f.write('')
        
        # Process the video
        frames = preprocessor.preprocess_video(test_video_path)
        
        # Check results
        self.assertEqual(frames.shape, (16, 64, 64, 3))  # 16 frames of 64x64 with 3 channels
        self.assertTrue(np.all(frames <= 1.0))  # Check normalization
        
        # Verify logger was called
        self.mock_logger.debug.assert_called()
    
    @mock.patch('src.data.data_preprocessor.VideoPreprocessor.preprocess_video')
    def test_process_dataset(self, mock_preprocess):
        """Test dataset processing functionality."""
        # Setup mock return value for preprocess_video
        mock_frames = np.zeros((16, 64, 64, 3), dtype=np.float32)
        mock_preprocess.return_value = mock_frames
        
        # Create test video files
        os.makedirs(self.test_dir, exist_ok=True)
        test_videos = ["video1.mp4", "video2.mp4"]
        for video in test_videos:
            with open(os.path.join(self.test_dir, video), 'w') as f:
                f.write('')
        
        # Initialize preprocessor and process dataset
        preprocessor = VideoPreprocessor(self.cfg, self.mock_logger)
        results = preprocessor.process_dataset(self.test_dir, self.processed_dir)
        
        # Check results
        self.assertEqual(len(results), 2)
        for video in test_videos:
            base_name = os.path.splitext(video)[0]
            output_path = os.path.join(self.processed_dir, f"{base_name}.npy")
            self.assertIn(video, results)
            self.assertEqual(results[video], output_path)
        
        # Verify logger was called
        self.mock_logger.info.assert_called()


@mock.patch('src.data.data_preprocessor.VideoPreprocessor')
def test_main_function(mock_video_processor):
    """Test the main function."""
    # Setup mock
    mock_processor_instance = mock.MagicMock()
    mock_video_processor.return_value = mock_processor_instance
    mock_processor_instance.process_dataset.return_value = {"video1.mp4": "processed/video1.npy"}
    
    # Create test config
    cfg = OmegaConf.create({
        "data": {
            "raw_dir": "raw_data",
            "processed_dir": "processed_data",
            "video": {
                "resize_shape": [64, 64],
                "num_frames": 16,
                "fps": 25,
                "normalize": True
            }
        }
    })
    
    # Create mock directory to bypass existence check
    with mock.patch('os.path.exists', return_value=True):
        main(cfg)
    
    # Verify processor was created and used
    mock_video_processor.assert_called_once()
    mock_processor_instance.process_dataset.assert_called_once()


if __name__ == '__main__':
    unittest.main() 