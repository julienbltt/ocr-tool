#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Camera Burst Module - Professional OCR burst capture system with optimized performance.

This module provides a high-performance OCR system for burst photo capture and analysis
using the modern TesseractOCR module with confidence-based text consolidation.

Author: Assistant IA
Date: 08/07/2025
Version: 3.0.0
Based on: Modern TesseractOCR module (ocr.py)
"""

import cv2
import numpy as np
import time
import threading
import keyboard
import re
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict
from pathlib import Path
import json
import os
from datetime import datetime
from functools import lru_cache
import tempfile
from concurrent.futures import ThreadPoolExecutor
import gc

# Import modern TesseractOCR class from ocr.py
from ocr import TesseractOCR, EnhancementLevel, PSMMode, OCRResult


class OCRCameraBurst:
    """
    Professional OCR Camera Burst system for real-time text extraction from camera feeds.
    
    This class provides comprehensive functionality for capturing burst photos from cameras
    and performing OCR analysis with confidence-based text consolidation, optimized for
    performance and professional usage using the modern TesseractOCR module.
    
    Attributes:
        burst_count (int): Number of photos per burst capture
        languages (List[str]): Languages for OCR processing
        confidence_threshold (float): Minimum confidence threshold for text acceptance
        enhancement_level (EnhancementLevel): Image preprocessing enhancement level
        camera (cv2.VideoCapture): Active camera object
        camera_index (int): Index of currently selected camera
        is_running (bool): System running state
        logger (logging.Logger): Logger instance for this class
    """
    
    def __init__(
        self, 
        burst_count: int = 5, 
        languages: List[str] = ['eng'], 
        confidence_threshold: float = 60.0,
        enhancement_level: EnhancementLevel = EnhancementLevel.MEDIUM,
        log_level: int = logging.INFO,
        max_workers: int = 4
    ) -> None:
        """
        Initialize OCR Camera Burst system with modern TesseractOCR integration.
        
        Args:
            burst_count: Number of photos per burst (default: 5)
            languages: Languages for OCR processing (default: ['eng'])
            confidence_threshold: Minimum confidence threshold (default: 60.0)
            enhancement_level: Image preprocessing level (default: MEDIUM)
            log_level: Logging level (default: INFO)
            max_workers: Maximum number of worker threads for parallel processing
        
        Raises:
            RuntimeError: If TesseractOCR initialization fails
        """
        self.burst_count = burst_count
        self.languages = languages
        self.confidence_threshold = confidence_threshold
        self.enhancement_level = enhancement_level
        self.camera: Optional[cv2.VideoCapture] = None
        self.camera_index = 0
        self.is_running = False
        self.capture_thread: Optional[threading.Thread] = None
        self.max_workers = max_workers
        
        # Performance optimization: pre-compile regex patterns
        self._compile_cleanup_patterns()
        
        # Setup directory structure
        self._setup_directories()
        
        # Configure logging
        self._setup_logging(log_level)
        
        # Initialize modern OCR engine with error handling
        try:
            self.ocr = TesseractOCR(language_cache=True)
            self.logger.info("Modern TesseractOCR engine initialized successfully")
            
            # Validate requested languages
            available_languages = self.ocr.available_languages
            valid_languages = [lang for lang in self.languages if lang in available_languages]
            
            if not valid_languages:
                self.logger.warning(f"No valid languages found from {self.languages}, using 'eng'")
                self.languages = ['eng']
            elif len(valid_languages) != len(self.languages):
                invalid_langs = set(self.languages) - set(valid_languages)
                self.logger.warning(f"Invalid languages removed: {invalid_langs}")
                self.languages = valid_languages
            
            self.logger.info(f"Using languages: {self.languages}")
            
        except Exception as e:
            self.logger.error(f"OCR initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize OCR engine: {e}")
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        self.logger.info(
            f"System initialized - Burst: {burst_count}, "
            f"Threshold: {confidence_threshold}%, "
            f"Enhancement: {enhancement_level.value}, "
            f"Languages: {self.languages}, "
            f"Workers: {max_workers}"
        )
    
    def _compile_cleanup_patterns(self) -> None:
        """Compile regex patterns for text cleanup optimization."""
        self.cleanup_patterns = [
            re.compile(r'[|\\/_\-~`^]'),  # Common parasitic symbols
            re.compile(r'\s+'),           # Multiple spaces
            re.compile(r'^[\s\W]+'),      # Leading spaces/symbols
            re.compile(r'[\s\W]+$')       # Trailing spaces/symbols
        ]
    
    def _setup_directories(self) -> None:
        """
        Setup directory structure for logs, results and temporary files.
        
        Creates organized directory structure with date-based organization
        for better file management and archival.
        """
        self.base_dir = Path.cwd()
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Define directory structure
        self.dirs = {
            'logs': self.base_dir / 'logs' / self.current_date,
            'results': self.base_dir / 'results' / self.current_date,
            'temp': self.base_dir / 'temp'
        }
        
        # Create directories with proper error handling
        for dir_name, dir_path in self.dirs.items():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"Directory initialized: {dir_path}")
            except Exception as e:
                print(f"Warning: Could not create {dir_name} directory: {e}")
                # Fallback to current directory
                self.dirs[dir_name] = self.base_dir
        
        # Set specific paths for optimized access
        self.log_file = self.dirs['logs'] / f'ocr_camera_{datetime.now().strftime("%Y%m%d")}.log'
        self.temp_dir = self.dirs['temp']
        self.results_dir = self.dirs['results']
    
    def _setup_logging(self, log_level: int) -> None:
        """
        Setup comprehensive logging configuration.
        
        Args:
            log_level: Logging level constant from logging module
        """
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_file, encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Logging system initialized - Log file: {self.log_file}")
        self.logger.debug(f"Directory structure: {dict(self.dirs)}")
    
    @lru_cache(maxsize=32)
    def _get_camera_properties(self, camera_index: int) -> Optional[Dict[str, Any]]:
        """
        Get camera properties with caching for performance optimization.
        
        Args:
            camera_index: Index of camera to query
            
        Returns:
            Dictionary containing camera properties or None if camera unavailable
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return None
        
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return None
        
        properties = {
            'index': camera_index,
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'resolution': f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        }
        
        cap.release()
        return properties
    
    def list_cameras(self) -> List[Dict[str, Any]]:
        """
        List all available cameras with comprehensive information.
        
        Returns:
            List of dictionaries containing camera information
        """
        self.logger.info("Scanning for available cameras")
        available_cameras = []
        
        # Use parallel processing for camera detection
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self._get_camera_properties, i): i for i in range(10)}
            
            for future in futures:
                try:
                    properties = future.result(timeout=2.0)
                    if properties:
                        available_cameras.append(properties)
                        self.logger.info(f"Camera {properties['index']} detected: {properties['resolution']}")
                except Exception as e:
                    self.logger.debug(f"Camera {futures[future]} detection failed: {e}")
        
        if not available_cameras:
            self.logger.warning("No cameras detected")
        else:
            self.logger.info(f"{len(available_cameras)} cameras detected")
        
        return available_cameras
    
    def select_camera(self, camera_index: Optional[int] = None) -> bool:
        """
        Select and initialize a camera with optimized settings.
        
        Args:
            camera_index: Camera index (None for interactive selection)
        
        Returns:
            True if camera successfully initialized, False otherwise
        """
        if camera_index is None:
            cameras = self.list_cameras()
            if not cameras:
                return False
            
            if len(cameras) == 1:
                camera_index = cameras[0]['index']
                self.logger.info(f"Auto-selected camera {camera_index}")
            else:
                print("\nAvailable cameras:")
                for cam in cameras:
                    print(f"   {cam['index']}: Camera {cam['index']} ({cam['resolution']})")
                
                while True:
                    try:
                        choice = input(f"\nSelect camera (0-{len(cameras)-1}): ")
                        camera_index = int(choice)
                        if any(cam['index'] == camera_index for cam in cameras):
                            break
                        else:
                            print("Invalid camera index")
                    except ValueError:
                        print("Please enter a valid number")
        
        # Initialize selected camera with optimal settings
        self.camera = cv2.VideoCapture(camera_index)
        self.camera_index = camera_index
        
        if not self.camera.isOpened():
            self.logger.error(f"Failed to initialize camera {camera_index}")
            return False
        
        # Apply optimal camera settings for OCR
        self._optimize_camera_settings()
        
        # Validate camera functionality
        ret, frame = self.camera.read()
        if ret and frame is not None:
            height, width = frame.shape[:2]
            self.logger.info(f"Camera {camera_index} initialized successfully: {width}x{height}")
            return True
        else:
            self.logger.error(f"Camera {camera_index} initialization test failed")
            return False
    
    def _optimize_camera_settings(self) -> None:
        """Apply optimal camera settings for OCR performance."""
        if not self.camera:
            return
        
        # Set optimal resolution for OCR (balance between quality and performance)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        # Additional optimizations if supported
        try:
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        except Exception as e:
            self.logger.debug(f"Advanced camera settings not supported: {e}")
    
    def capture_burst(self) -> List[np.ndarray]:
        """
        Capture a burst of photos with optimized timing.
        
        Returns:
            List of captured images as numpy arrays
        """
        if not self.camera or not self.camera.isOpened():
            self.logger.error("Camera not initialized for burst capture")
            return []
        
        images = []
        self.logger.info(f"Initiating burst capture: {self.burst_count} photos")
        
        # Camera stabilization period
        time.sleep(0.1)
        
        # Pre-allocate list for performance
        images = [None] * self.burst_count
        
        # Capture images with optimal timing
        for i in range(self.burst_count):
            ret, frame = self.camera.read()
            if ret and frame is not None:
                images[i] = frame.copy()
                self.logger.debug(f"Photo {i+1}/{self.burst_count} captured")
                if i < self.burst_count - 1:  # No delay after last capture
                    time.sleep(0.05)  # Minimal delay between captures
            else:
                self.logger.warning(f"Failed to capture photo {i+1}")
        
        # Filter out None values
        valid_images = [img for img in images if img is not None]
        
        self.logger.info(f"Burst capture completed: {len(valid_images)}/{self.burst_count} photos captured")
        return valid_images
    
    def _process_single_image(self, image_data: Tuple[int, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Process a single image for OCR analysis using modern TesseractOCR module.
        
        Args:
            image_data: Tuple of (image_index, image_array)
            
        Returns:
            List of OCR results for the image
        """
        image_index, image = image_data
        
        # Create temporary file with unique name
        temp_filename = f"temp_burst_{image_index}_{int(time.time()*1000)}.png"
        temp_path = self.temp_dir / temp_filename
        
        try:
            # Save image to temporary file
            cv2.imwrite(str(temp_path), image)
            
            # Perform OCR analysis using modern interface
            ocr_results = self.ocr.extract_text_with_boxes(
                temp_path,
                languages=self.languages,
                confidence_threshold=self.confidence_threshold,
                enhancement_level=self.enhancement_level
            )
            
            # Process results - convert OCRResult objects to dictionaries
            results = []
            for ocr_result in ocr_results:
                word = self.clean_word(ocr_result.text)
                if word and len(word) > 1:  # Filter single characters
                    results.append({
                        'image_index': image_index,
                        'word': word,
                        'confidence': ocr_result.confidence,
                        'coordinates': (ocr_result.x, ocr_result.y, ocr_result.width, ocr_result.height),
                        'level': ocr_result.level
                    })
            
            self.logger.debug(f"Image {image_index}: {len(results)} valid elements detected")
            return results
            
        except Exception as e:
            self.logger.error(f"OCR processing failed for image {image_index}: {e}")
            return []
        
        finally:
            # Clean up temporary file
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to remove temporary file {temp_path}: {e}")
    
    def analyze_image_burst(self, images: List[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze burst of images with parallel OCR processing using modern TesseractOCR.
        
        Args:
            images: List of images as numpy arrays
        
        Returns:
            Dictionary containing consolidated OCR results
        """
        if not images:
            return {'consolidated_text': '', 'word_data': {}, 'stats': {}}
        
        self.logger.info(f"Starting parallel OCR analysis of {len(images)} images")
        
        # Collect all detected words with their confidences
        all_words = defaultdict(list)
        all_boxes_data = []
        
        # Process images in parallel for better performance
        image_data = [(i, img) for i, img in enumerate(images)]
        
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(images))) as executor:
            futures = [executor.submit(self._process_single_image, img_data) for img_data in image_data]
            
            for future in futures:
                try:
                    results = future.result(timeout=30.0)  # 30 second timeout per image
                    
                    # Consolidate results
                    for result in results:
                        word = result['word']
                        all_words[word].append(result['confidence'])
                        all_boxes_data.append(result)
                        
                except Exception as e:
                    self.logger.error(f"Image processing failed: {e}")
        
        # Force garbage collection to free memory
        gc.collect()
        
        # Consolidate and return results
        return self._consolidate_results(all_words, all_boxes_data)
    
    def clean_word(self, word: str) -> str:
        """
        Clean detected word using pre-compiled regex patterns.
        
        Args:
            word: Raw word from OCR
        
        Returns:
            Cleaned word string
        """
        if not word:
            return ""
        
        # Apply pre-compiled cleanup patterns for performance
        cleaned = word
        for pattern in self.cleanup_patterns:
            if pattern.pattern == r'\s+':
                cleaned = pattern.sub(' ', cleaned)
            else:
                cleaned = pattern.sub('', cleaned)
        
        return cleaned.strip()
    
    def _consolidate_results(self, all_words: Dict[str, List[float]], all_boxes_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Consolidate OCR results from all images with optimized processing.
        
        Args:
            all_words: Dictionary mapping words to confidence lists
            all_boxes_data: List of detailed box data
        
        Returns:
            Dictionary containing consolidated results
        """
        self.logger.info("Consolidating OCR results")
        
        # Calculate final confidence for each word (vectorized for performance)
        final_words = {}
        for word, confidences in all_words.items():
            confidences_array = np.array(confidences)
            final_words[word] = {
                'max_confidence': float(confidences_array.max()),
                'avg_confidence': float(confidences_array.mean()),
                'occurrences': len(confidences),
                'confidences': confidences
            }
        
        # Filter words with insufficient confidence
        filtered_words = {
            word: data for word, data in final_words.items()
            if data['max_confidence'] >= self.confidence_threshold
        }
        
        # Sort by maximum confidence (descending)
        sorted_words = sorted(
            filtered_words.items(),
            key=lambda x: x[1]['max_confidence'],
            reverse=True
        )
        
        # Build final text
        final_text_words = [word for word, _ in sorted_words]
        consolidated_text = ' '.join(final_text_words)
        
        # Final text cleaning
        consolidated_text = self._clean_final_text(consolidated_text)
        
        # Calculate statistics
        stats = self._calculate_statistics(final_words, filtered_words, consolidated_text)
        
        self.logger.info(
            f"Consolidation complete: {stats['total_unique_words']} unique words -> "
            f"{stats['filtered_words']} retained (avg confidence: {stats['avg_confidence']:.1f}%)"
        )
        
        return {
            'consolidated_text': consolidated_text,
            'word_data': dict(sorted_words),
            'all_boxes': all_boxes_data,
            'stats': stats
        }
    
    def _calculate_statistics(self, final_words: Dict, filtered_words: Dict, consolidated_text: str) -> Dict[str, Any]:
        """Calculate comprehensive statistics for OCR results."""
        confidences = [data['max_confidence'] for data in filtered_words.values()]
        
        return {
            'total_unique_words': len(final_words),
            'filtered_words': len(filtered_words),
            'final_text_length': len(consolidated_text),
            'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'confidence_range': {
                'min': float(min(confidences)) if confidences else 0.0,
                'max': float(max(confidences)) if confidences else 0.0
            }
        }
    
    def _clean_final_text(self, text: str) -> str:
        """
        Perform final cleaning of consolidated text.
        
        Args:
            text: Raw consolidated text
        
        Returns:
            Cleaned text string
        """
        if not text:
            return ""
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove words that are too short (likely artifacts)
        words = text.split()
        filtered_words = [word for word in words if len(word) > 1 or word.isdigit()]
        
        return ' '.join(filtered_words)
    
    def generate_report(self, results: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report with professional formatting.
        
        Args:
            results: OCR analysis results
            session_id: Optional session identifier
        
        Returns:
            Formatted report dictionary
        """
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        text = results['consolidated_text']
        stats = results['stats']
        word_data = results['word_data']
        
        # Calculate confidence distribution
        confidences = [data['max_confidence'] for data in word_data.values()]
        confidence_distribution = {
            'high_confidence': len([c for c in confidences if c >= 90]),
            'medium_confidence': len([c for c in confidences if 70 <= c < 90]),
            'low_confidence': len([c for c in confidences if c < 70])
        }
        
        # Word length analysis
        word_lengths = [len(word) for word in word_data.keys()]
        word_length_stats = {
            'min_length': min(word_lengths) if word_lengths else 0,
            'max_length': max(word_lengths) if word_lengths else 0,
            'avg_length': float(np.mean(word_lengths)) if word_lengths else 0.0
        }
        
        return {
            'session_info': {
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'system_config': {
                    'burst_count': self.burst_count,
                    'languages': self.languages,
                    'confidence_threshold': self.confidence_threshold,
                    'enhancement_level': self.enhancement_level.value,
                    'camera_index': self.camera_index,
                    'max_workers': self.max_workers
                }
            },
            'ocr_results': {
                'consolidated_text': text,
                'text_length': len(text),
                'word_count': len(word_data),
                'character_count': len(text.replace(' ', ''))
            },
            'quality_metrics': {
                'average_confidence': round(stats['avg_confidence'], 2),
                'confidence_range': stats['confidence_range'],
                'confidence_distribution': confidence_distribution,
                'word_length_statistics': word_length_stats
            },
            'processing_statistics': {
                'unique_words_detected': stats['total_unique_words'],
                'words_after_filtering': stats['filtered_words'],
                'retention_rate': round(
                    (stats['filtered_words'] / stats['total_unique_words']) * 100, 2
                ) if stats['total_unique_words'] > 0 else 0.0
            },
            'top_words': [
                {
                    'word': word,
                    'confidence': data['max_confidence'],
                    'occurrences': data['occurrences']
                }
                for word, data in list(word_data.items())[:10]
            ]
        }
    
    def save_results(self, results: Dict[str, Any], session_id: Optional[str] = None) -> bool:
        """
        Save analysis results to JSON file in organized directory structure.
        
        Args:
            results: Results dictionary to save
            session_id: Optional session identifier
        
        Returns:
            True if save successful, False otherwise
        """
        report = self.generate_report(results, session_id)
        filename = f"ocr_results_{report['session_info']['session_id']}.json"
        filepath = self.results_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Results saved successfully: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            
            # Fallback to current directory
            fallback_path = Path.cwd() / filename
            try:
                with open(fallback_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Results saved to fallback location: {fallback_path}")
                return True
            except Exception as e2:
                self.logger.error(f"Fallback save failed: {e2}")
                return False
    
    def preview_camera(self) -> None:
        """Display real-time camera preview with OCR trigger capability."""
        if not self.camera or not self.camera.isOpened():
            self.logger.error("Camera not initialized for preview")
            return
        
        self.logger.info("Camera preview started - Press 'q' to quit, SPACE to capture")
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            # Add professional overlay information
            cv2.putText(frame, f"Camera {self.camera_index}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE: Capture | Q: Quit", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add enhancement level indicator
            cv2.putText(frame, f"Enhancement: {self.enhancement_level.value}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow('OCR Camera Preview', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.logger.info("Burst capture triggered from preview")
                images = self.capture_burst()
                if images:
                    results = self.analyze_image_burst(images)
                    self.display_results(results)
                    self.save_results(results)
        
        cv2.destroyAllWindows()
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """
        Display OCR analysis results in professional format.
        
        Args:
            results: Analysis results dictionary
        """
        report = self.generate_report(results)
        
        print("\n" + "="*80)
        print("OCR ANALYSIS REPORT")
        print("="*80)
        
        # Session information
        session_info = report['session_info']
        print(f"Session ID: {session_info['session_id']}")
        print(f"Timestamp: {session_info['timestamp']}")
        config = session_info['system_config']
        print(f"Configuration: {config['burst_count']} photos, "
              f"threshold {config['confidence_threshold']}%, "
              f"enhancement {config['enhancement_level']}")
        
        print("\n" + "-"*80)
        print("EXTRACTED TEXT")
        print("-"*80)
        
        ocr_results = report['ocr_results']
        print(f"Text: {ocr_results['consolidated_text']}")
        print(f"Length: {ocr_results['character_count']} characters, {ocr_results['word_count']} words")
        
        print("\n" + "-"*80)
        print("QUALITY METRICS")
        print("-"*80)
        
        quality = report['quality_metrics']
        print(f"Average Confidence: {quality['average_confidence']}%")
        print(f"Confidence Range: {quality['confidence_range']['min']:.1f}% - {quality['confidence_range']['max']:.1f}%")
        
        conf_dist = quality['confidence_distribution']
        print(f"Confidence Distribution:")
        print(f"  High (>=90%): {conf_dist['high_confidence']} words")
        print(f"  Medium (70-89%): {conf_dist['medium_confidence']} words")
        print(f"  Low (<70%): {conf_dist['low_confidence']} words")
        
        print("\n" + "-"*80)
        print("PROCESSING STATISTICS")
        print("-"*80)
        
        proc_stats = report['processing_statistics']
        print(f"Words Detected: {proc_stats['unique_words_detected']}")
        print(f"Words Retained: {proc_stats['words_after_filtering']}")
        print(f"Retention Rate: {proc_stats['retention_rate']}%")
        
        if report['top_words']:
            print("\n" + "-"*80)
            print("TOP DETECTED WORDS")
            print("-"*80)
            
            for i, word_info in enumerate(report['top_words'][:5], 1):
                print(f"{i:2d}. {word_info['word']:<20} "
                      f"Confidence: {word_info['confidence']:6.1f}% "
                      f"Occurrences: {word_info['occurrences']}")
        
        print("="*80)
    
    def run_keyboard_mode(self) -> None:
        """
        Run interactive keyboard control mode with professional interface.
        
        This method provides the main interactive interface for the OCR system,
        allowing users to trigger captures, preview camera, and configure settings.
        """
        print("\nKEYBOARD CONTROL MODE")
        print("="*40)
        print("SPACE: Capture burst")
        print("P: Camera preview")
        print("C: Configure settings")
        print("E: Enhancement level")
        print("I: Directory information")
        print("ESC: Exit")
        print("="*40)
        
        self.is_running = True
        
        try:
            while self.is_running:
                event = keyboard.read_event()
                
                if event.event_type == keyboard.KEY_DOWN:
                    if event.name == 'space':
                        self.logger.info("Burst capture initiated via keyboard")
                        images = self.capture_burst()
                        if images:
                            results = self.analyze_image_burst(images)
                            self.display_results(results)
                            self.save_results(results)
                    
                    elif event.name == 'p':
                        self.preview_camera()
                    
                    elif event.name == 'c':
                        self.configure_settings()
                    
                    elif event.name == 'e':
                        self.configure_enhancement_level()
                    
                    elif event.name == 'i':
                        self.display_directory_info()
                    
                    elif event.name == 'esc':
                        self.logger.info("Shutdown requested via keyboard")
                        self.is_running = False
                        break
                
        except KeyboardInterrupt:
            self.logger.info("Shutdown via Ctrl+C")
        
        finally:
            self.cleanup()
    
    def configure_enhancement_level(self) -> None:
        """Interactive enhancement level configuration."""
        print("\nENHANCEMENT LEVEL CONFIGURATION")
        print("-" * 35)
        print("Available enhancement levels:")
        for i, level in enumerate(EnhancementLevel, 1):
            current = " (current)" if level == self.enhancement_level else ""
            print(f"  {i}. {level.value.upper()}{current}")
        
        try:
            choice = input("\nSelect enhancement level (1-3): ")
            levels = list(EnhancementLevel)
            
            if 1 <= int(choice) <= len(levels):
                self.enhancement_level = levels[int(choice) - 1]
                self.logger.info(f"Enhancement level updated to: {self.enhancement_level.value}")
                print(f"Enhancement level set to: {self.enhancement_level.value}")
            else:
                print("Invalid selection. Configuration unchanged.")
                
        except (ValueError, IndexError):
            print("Invalid input. Configuration unchanged.")
    
    def configure_settings(self) -> None:
        """Interactive configuration interface for system settings."""
        print("\nSYSTEM CONFIGURATION")
        print("-" * 20)
        
        try:
            new_burst = input(f"Photos per burst (current: {self.burst_count}): ")
            if new_burst.strip():
                self.burst_count = max(1, int(new_burst))
                self.logger.info(f"Burst count updated to: {self.burst_count}")
            
            new_threshold = input(f"Confidence threshold % (current: {self.confidence_threshold}): ")
            if new_threshold.strip():
                self.confidence_threshold = max(0.0, min(100.0, float(new_threshold)))
                self.logger.info(f"Confidence threshold updated to: {self.confidence_threshold}%")
            
        except ValueError:
            self.logger.warning("Invalid input provided, configuration unchanged")
            print("Invalid input. Configuration unchanged.")
    
    def display_directory_info(self) -> None:
        """Display comprehensive directory structure and file information."""
        print("\n" + "="*60)
        print("DIRECTORY STRUCTURE INFORMATION")
        print("="*60)
        
        dir_info = self.get_directory_info()
        
        for name, info in dir_info.items():
            print(f"\n{name.upper()} Directory:")
            print(f"  Path: {info['path']}")
            print(f"  Status: {'Ready' if info.get('exists', False) else 'Created'}")
            
            if info.get('exists', False):
                print(f"  Files: {info.get('file_count', 0)}")
                if info.get('files'):
                    print(f"  Recent files: {', '.join(info['files'])}")
            
            if 'error' in info:
                print(f"  Error: {info['error']}")
        
        print(f"\nCurrent session log: {self.log_file}")
        
        # Display OCR engine information
        print(f"\nOCR Engine Information:")
        print(f"  Available languages: {', '.join(self.ocr.available_languages)}")
        print(f"  Active languages: {', '.join(self.languages)}")
        print(f"  Enhancement level: {self.enhancement_level.value}")
        
        # Display disk usage information
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.base_dir)
            print(f"\nDisk Usage:")
            print(f"  Total: {total // (1024**3):.1f} GB")
            print(f"  Used: {used // (1024**3):.1f} GB")
            print(f"  Free: {free // (1024**3):.1f} GB")
        except Exception as e:
            print(f"  Could not retrieve disk usage: {e}")
        
        print("="*60)
    
    def get_directory_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive information about directory structure and contents.
        
        Returns:
            Dictionary containing directory information and file counts
        """
        info = {}
        
        for dir_name, dir_path in self.dirs.items():
            try:
                files = list(dir_path.glob("*"))
                info[dir_name] = {
                    'path': str(dir_path),
                    'exists': dir_path.exists(),
                    'file_count': len(files),
                    'files': [f.name for f in files[:5]]  # Show first 5 files
                }
            except Exception as e:
                info[dir_name] = {
                    'path': str(dir_path),
                    'exists': False,
                    'error': str(e)
                }
        
        return info
    
    def cleanup(self) -> None:
        """
        Comprehensive cleanup of system resources and temporary files.
        
        This method ensures proper cleanup of camera resources, temporary files,
        OCR engine cache, and thread pools to prevent resource leaks.
        """
        # Release camera resources
        if self.camera:
            self.camera.release()
            self.camera = None
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        # Clean up temporary files
        self._cleanup_temp_files()
        
        # Clear OCR engine cache
        if hasattr(self, 'ocr'):
            self.ocr.clear_cache()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        # Clear camera properties cache
        self._get_camera_properties.cache_clear()
        
        # Force garbage collection
        gc.collect()
        
        self.logger.info("System cleanup completed successfully")
    
    def _cleanup_temp_files(self) -> None:
        """Clean up temporary files created during processing."""
        try:
            temp_files = list(self.temp_dir.glob("temp_burst_*.png"))
            removed_count = 0
            
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                    removed_count += 1
                    self.logger.debug(f"Removed temporary file: {temp_file}")
                except Exception as e:
                    self.logger.warning(f"Could not remove temporary file {temp_file}: {e}")
            
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} temporary files")
                
        except Exception as e:
            self.logger.warning(f"Error during temporary file cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()


def main() -> None:
    """
    Main function providing command-line interface for the OCR Camera Burst system.
    
    This function handles system initialization, configuration, and provides
    an interactive interface for users using the modern TesseractOCR module.
    """
    print("OCR CAMERA BURST SYSTEM v3.0")
    print("="*60)
    print("Using Modern TesseractOCR Module")
    
    try:
        # Initial configuration with validation
        print("\nSystem Configuration:")
        burst_count = 3
        languages = ['eng', 'fra', 'glg']
        confidence_threshold = 60.0
        enhancement_level = EnhancementLevel.MEDIUM
        
        # Interactive configuration
        try:
            custom_burst = input(f"Photos per burst (default: {burst_count}): ")
            if custom_burst.strip():
                burst_count = max(1, int(custom_burst))
            
            custom_threshold = input(f"Confidence threshold % (default: {confidence_threshold}): ")
            if custom_threshold.strip():
                confidence_threshold = max(0.0, min(100.0, float(custom_threshold)))
            
            print("\nEnhancement levels:")
            for i, level in enumerate(EnhancementLevel, 1):
                print(f"  {i}. {level.value.upper()}")
            
            custom_enhancement = input(f"Enhancement level (1-3, default: 2): ")
            if custom_enhancement.strip():
                levels = list(EnhancementLevel)
                idx = max(0, min(len(levels) - 1, int(custom_enhancement) - 1))
                enhancement_level = levels[idx]
                
        except ValueError:
            print("Invalid input detected. Using default values.")
        
        # Initialize system with context manager for proper cleanup
        with OCRCameraBurst(
            burst_count=burst_count,
            languages=languages,
            confidence_threshold=confidence_threshold,
            enhancement_level=enhancement_level
        ) as ocr_camera:
            
            # Display system information
            print("\nSystem Information:")
            print(f"  OCR Engine: Modern TesseractOCR v2.0")
            print(f"  Available languages: {', '.join(ocr_camera.ocr.available_languages)}")
            print(f"  Enhancement level: {enhancement_level.value}")
            
            print("\nDirectory Structure:")
            dir_info = ocr_camera.get_directory_info()
            for name, info in dir_info.items():
                print(f"  {name.upper()}: {info['path']}")
                status = "Ready" if info.get('exists', False) else "Created"
                file_count = info.get('file_count', 0)
                print(f"    Status: {status} ({file_count} files)")
            
            # Camera initialization
            if not ocr_camera.select_camera():
                print("Failed to initialize camera. Exiting.")
                return
            
            # Start interactive mode
            ocr_camera.run_keyboard_mode()
        
    except Exception as e:
        print(f"System error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()