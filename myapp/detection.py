import os
import time
import gc
import tracemalloc
import pickle
import logging
import io
from collections import deque
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict

import django
from django.conf import settings

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myapp.settings')
django.setup()

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pygame
from gtts import gTTS
 
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
NUM_LANDMARKS = 42
IMG_SIZE = 224  # Image size for MobileNetV2 and RNN models, chosen to match the input size expected by these models
SEQUENCE_LENGTH = 30
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_CONFIG = {
    'cnn': {
        'path': os.path.join(BASE_DIR, "models/asl_mobilenetv2_model.h5"),
        'weight': 0.6
    },
    'rnn': {
        'path': os.path.join(BASE_DIR, "models/rnn_asl_model.keras"),
        'weight': 0.4
    }
}
LABEL_ENCODER_PATHS = {
    'cnn': os.path.join(BASE_DIR, "data/labels/cnn_label_encoder.pkl"),
    'rnn': os.path.join(BASE_DIR, "data/labels/rnn_label_encoder.pkl")
}

class ASLDetector:
    def __init__(self):
        self.MIN_HAND_DETECTION_CONFIDENCE = 0.5  # Or whatever value you want
        self.FRAME_WIDTH = 1280  # Set default frame width
        self.FRAME_HEIGHT = 720  # Set default frame height
        self.PREDICTION_SMOOTHING_WINDOW = 5
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logger.error("Could not open camera.")
            raise RuntimeError("Could not open camera.")
        self.setup_parameters()  # Call setup_parameters first to define confidence value
        self.initialize_components()

    def setup_parameters(self):
        """Initialize configuration parameters"""
        self.SPEECH_DELAY = 3
        self.CONFIDENCE_THRESHOLD = 0.85
        try:
            pygame.init()
            pygame.mixer.init()
        except Exception as e:
            logger.error(f"Failed to initialize pygame or mixer: {e}")

    def initialize_components(self):
        self.models = self._load_models()
        self.label_encoders = self._load_label_encoders()

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=self.MIN_HAND_DETECTION_CONFIDENCE)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.audio_file_counter = 0
        self.last_speech_time = 0
        self.previous_label = ""
        self.frame_sequence = deque(maxlen=SEQUENCE_LENGTH)
        self.recent_predictions = deque(maxlen=self.PREDICTION_SMOOTHING_WINDOW)
        self.audio_executor = ThreadPoolExecutor(max_workers=2)
        self.previous_landmarks = None  # Initialize previous_landmarks
        self.MIN_HAND_DETECTION_CONFIDENCE = 0.5  # Set a default value for hand detection confidence
        self.FRAME_WIDTH = 1280  # Set default frame width
        self.FRAME_HEIGHT = 720  # Set default frame height
        self.PREDICTION_SMOOTHING_WINDOW = 5  # Set default prediction smoothing window

    def _load_models(self) -> Dict[str, Optional[tf.keras.Model]]:
        """Load models with improved error handling and cleanup"""
        models = {}
        
        # Configure TensorFlow to avoid creating temporary directories
        tf.keras.utils.disable_interactive_logging()
        tf.get_logger().setLevel('ERROR')
        
        for model_name, config in MODELS_CONFIG.items():
            try:
                # Ensure the model file exists
                if not os.path.exists(config['path']):
                    logger.error(f"Model file not found: {config['path']}")
                    models[model_name] = None
                    continue

                # Custom load function for different model types
                if model_name == 'cnn':
                    models[model_name] = tf.keras.models.load_model(
                        config['path'],
                        compile=False  # Don't load optimizer state
                    )
                elif model_name == 'rnn':
                    # Load RNN model with custom options
                    models[model_name] = tf.keras.models.load_model(
                        config['path'],
                        compile=False,
                        options=tf.saved_model.LoadOptions(
                            experimental_io_device='/job:localhost'
                        )
                    )
                
                # Recompile the model with basic optimizer
                if models[model_name] is not None:
                    models[model_name].compile(
                        optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                logger.info(f"Successfully loaded {model_name} model")
                
            except Exception as e:
                logger.error(f"Error loading {model_name} model: {str(e)}")
                models[model_name] = None
                
            finally:
                # Clean up any temporary files
                self._cleanup_temp_files()
        
        return models

    def _cleanup_temp_files(self):
        """Clean up temporary model files"""
        try:
            temp_dirs = [
                os.path.join(BASE_DIR, d) for d in os.listdir(BASE_DIR)
                if d.startswith('tmp') and os.path.isdir(os.path.join(BASE_DIR, d))
            ]
            
            for temp_dir in temp_dirs:
                try:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary directory {temp_dir}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error during temporary file cleanup: {str(e)}")
            
            
    def _load_label_encoders(self) -> Dict[str, Optional[object]]:
        """Load label encoders for each model"""
        encoders = {}
        for model_type, path in LABEL_ENCODER_PATHS.items():
            try:
                with open(path, 'rb') as f:
                    encoders[model_type] = pickle.load(f)
                logger.info(f"Successfully loaded {model_type} label encoder")
            except Exception as e:
                logger.error(f"Error loading {model_type} label encoder: {e}")
                encoders[model_type] = None  # Explicitly set to None
        return encoders

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame with enhancement techniques"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        filtered = cv2.GaussianBlur(enhanced, (5, 5), 0)
        return cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)

    def extract_landmarks(self, hand_landmarks) -> list[list[float]]:
        """Extract and normalize landmarks with velocity calculation."""
        landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
        VELOCITY_THRESHOLD = 0.01  # Define a threshold for significant changes

        if self.previous_landmarks is not None:
            for i in range(len(landmarks)):
                velocity = [landmarks[i][j] - self.previous_landmarks[i][j] for j in range(3)]
                if any(abs(v) > VELOCITY_THRESHOLD for v in velocity):
                    landmarks[i].extend(velocity)
                else:
                    landmarks[i].extend([0.0, 0.0, 0.0])
        else:
            for landmark in landmarks:
                landmark.extend([0.0, 0.0, 0.0])
        return landmarks

    def preprocess_landmarks(self, landmarks: list[list[float]]) -> np.ndarray:
        """Convert landmarks to image representation"""
        landmarks_image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        x_coords = [lm[0] for lm in landmarks]
        y_coords = [lm[1] for lm in landmarks]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        x_range = max(max_x - min_x, 1e-6)  # Prevent division by zero
        y_range = max(max_y - min_y, 1e-6)  # Prevent division by zero

        normalized_landmarks = [
            ((x - min_x) / x_range * (IMG_SIZE - 1),
             (y - min_y) / y_range * (IMG_SIZE - 1))
            for x, y in zip(x_coords, y_coords)
        ]

        for connection in self.mp_hands.HAND_CONNECTIONS:
            start_point = tuple(map(int, normalized_landmarks[connection[0]]))
            end_point = tuple(map(int, normalized_landmarks[connection[1]]))
            cv2.line(landmarks_image, start_point, end_point, (255, 255, 255), 2)

        for point in normalized_landmarks:
            x, y = map(int, point)
            cv2.circle(landmarks_image, (x, y), 5, (0, 0, 255), -1)

        return landmarks_image

    def draw_landmarks(self, frame: np.ndarray, landmarks: list[list[float]], hand_landmarks) -> None:
        """Draw hand landmarks and connections on the frame"""
        self.mp_draw.draw_landmarks(
            frame,
            hand_landmarks
        )

        height, width = frame.shape[:2]
        for lm in landmarks:
            x, y = int(lm[0] * width), int(lm[1] * height)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    def display_prediction(self, frame: np.ndarray, predicted_label: str, confidence: float, fps: float) -> None:
        """Display prediction and FPS information on frame"""
        color = (0, 255, 0) if confidence >= self.CONFIDENCE_THRESHOLD else (0, 165, 255)

        cv2.putText(frame, f"Sign: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def handle_text_to_speech(self, predicted_label: str, confidence: float) -> None:
        """Handle text-to-speech output with timing control and cleanup."""
        current_time = time.time()

        if (predicted_label != self.previous_label and
                current_time - self.last_speech_time >= self.SPEECH_DELAY and
                confidence >= self.CONFIDENCE_THRESHOLD):
            try:
                tts = gTTS(predicted_label, lang='en')
                audio_data = io.BytesIO()
                tts.write_to_fp(audio_data)
                audio_data.seek(0)

                def play_and_cleanup():
                    try:
                        pygame.mixer.music.load(audio_data, 'mp3')
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            pygame.time.Clock().tick(10)
                    except Exception as e:
                        logger.error(f"Error in audio playback: {e}")

                self.audio_executor.submit(play_and_cleanup)
                self.audio_file_counter = (self.audio_file_counter + 1) % 10  # prevent overflow
                self.last_speech_time = current_time
                self.previous_label = predicted_label

            except Exception as e:
                logger.error(f"Error in text-to-speech: {e}")

    def cleanup_old_files(self):
        """Clean up old temporary audio files"""
        temp_dir = os.path.join(settings.BASE_DIR, "temp_audio_files")
        try:
            for filename in os.listdir(temp_dir):
                if filename.startswith("temp_") and filename.endswith(".mp3"):
                    file_path = os.path.join(temp_dir, filename)
                    if time.time() - os.path.getctime(file_path) > 300:  # 5 minutes
                        os.remove(file_path)
        except Exception as e:
            logger.error(f"Error removing old files: {e}")

    def run(self):
        try:
            frame_count = 0
            fps_start_time = time.time()
            tracemalloc.start()  # Start tracing memory allocations
            memory_threshold = 100 * 1024 * 1024  # Set a memory threshold (e.g., 100 MB)

            while self.cap.isOpened():  # Check if capture is opened
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break

                frame_count += 1
                current_time = time.time()

                if current_time - fps_start_time >= 1:
                    frame_count / (current_time - fps_start_time)
                    frame_count = 0
                    fps_start_time = current_time

                frame = cv2.resize(frame, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
                frame = cv2.flip(frame, 1)

                processed_frame = self.preprocess_frame(frame)
                results = self.hands.process(processed_frame)  # Initialize results variable

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        try:
                            landmarks = self.extract_landmarks(hand_landmarks)
                        except Exception as e:
                            logger.error(f"Error extracting landmarks: {e}")
                            continue
                        preprocessed_image = self.preprocess_landmarks(landmarks)
                        self.frame_sequence.append(preprocessed_image)
                        self.draw_landmarks(frame, landmarks, hand_landmarks)

                if len(self.frame_sequence) == SEQUENCE_LENGTH:
                    input_data = np.array(self.frame_sequence)
                    predicted_label, confidence = self.get_ensemble_prediction(input_data)
                    _, peak = tracemalloc.get_traced_memory()
                    self.handle_text_to_speech(predicted_label, confidence)
                    tracemalloc.get_traced_memory()
                cv2.imshow("Real-time ASL Detection", frame)
                if frame_count % 300 == 0:  # Every 300 frames
                    self.cleanup_old_files()
                    _, peak = tracemalloc.get_traced_memory()
                    if peak > memory_threshold:
                        gc.collect()
                        logger.info(f"Garbage collection triggered. Peak memory usage: {peak / 1024 / 1024:.2f} MB")
                    tracemalloc.reset_peak()

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Enhanced cleanup method"""
        try:
            # Release video capture
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
            
            # Close all windows
            cv2.destroyAllWindows()
            
            # Shutdown thread pool
            if hasattr(self, 'audio_executor'):
                self.audio_executor.shutdown(wait=False)
            
            # Cleanup pygame
            try:
                pygame.mixer.quit()
                pygame.quit()
            except Exception:
                pass
            
            # Cleanup temporary files
            self._cleanup_temp_files()
            
            # Clear model references
            if hasattr(self, 'models'):
                self.models.clear()
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def get_frame(self):
        """Get a frame from the video capture."""
        logger.debug("Entering get_frame")  # Add debug log

        try:
            if not self.cap or not self.cap.isOpened():
                logger.error("Camera not opened or initialized in get_frame")
                self.start_capture()  # Attempt to restart capture if not started/opened
                if not self.cap or not self.cap.isOpened():
                    return None, None, 0.0  # Return immediately if still failed

            ret, frame = self.cap.read()

            if not ret or frame is None:
                logger.error("Failed to read frame from camera")
                return None, None, 0.0

            frame = cv2.resize(frame, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
            frame = cv2.flip(frame, 1)

            processed_frame = self.preprocess_frame(frame.copy())  # Create a copy for processing
            results = self.hands.process(processed_frame)  # Use processed frame for detection

            prediction = "No gesture detected"
            confidence = 0.0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = self.extract_landmarks(hand_landmarks)
                    preprocessed_image = self.preprocess_landmarks(landmarks)
                    self.frame_sequence.append(preprocessed_image)
                    self.draw_landmarks(frame, landmarks, hand_landmarks)  # Draw on original frame

                if len(self.frame_sequence) == SEQUENCE_LENGTH:
                    input_data = np.expand_dims(np.array(self.frame_sequence), axis=0)
                    prediction, confidence = self.get_ensemble_prediction(input_data)

            _, jpeg = cv2.imencode('.jpg', frame)  # Encode the original frame with drawings
            if jpeg is None:
                logger.error("Failed to encode frame to JPEG")
                return None, None, 0.0  # Handle encoding failures

            return jpeg.tobytes(), prediction, confidence
        except Exception as e:
            logger.error(f"Error in get_frame: {e}", exc_info=True)
            return None, None, 0.0

    def get_ensemble_prediction(self, input_data):
        """Get ensemble prediction from multiple models, handling errors and mismatches."""
        predictions = []
        
        for model_name, model in self.models.items():
            if model is not None and self.label_encoders.get(model_name) is not None:
                try:
                    if model_name == 'cnn':
                        # For CNN, use only the most recent frame
                        # Take the last frame from the sequence
                        single_frame = input_data[0, -1]  # Shape: (224, 224, 3)
                        # Add batch dimension
                        processed_input = np.expand_dims(single_frame, axis=0)  # Shape: (1, 224, 224, 3)
                    elif model_name == 'rnn':
                        # For RNN, use the full sequence
                        processed_input = input_data  # Shape: (1, 30, 224, 224, 3)
                    else:
                        logger.warning(f"Unknown model type: {model_name}")
                        continue

                    # Make prediction with correctly shaped input
                    prediction_scores = model.predict(processed_input, verbose=0)
                    model_prediction = np.argmax(prediction_scores)
                    confidence_score = np.max(prediction_scores)
                    
                    # Convert prediction to label
                    label = self.label_encoders[model_name].inverse_transform([model_prediction])[0]
                    # Include both the model weight and the confidence score
                    weighted_confidence = confidence_score * MODELS_CONFIG[model_name]['weight']
                    predictions.append((label, weighted_confidence))
                    
                except Exception as e:
                    logger.error(f"Error during prediction with {model_name}: {e}", exc_info=True)
                    continue

        if predictions:
            # Group predictions by label and sum their weighted confidences
            label_confidences = {}
            for label, conf in predictions:
                label_confidences[label] = label_confidences.get(label, 0) + conf
            
            # Get the label with highest combined confidence
            best_label = max(label_confidences.items(), key=lambda x: x[1])
            return best_label[0], best_label[1]
        
        return "No gesture detected", 0.0

    def is_capturing(self):
        """Check if the video capture is currently active and opened."""
        return self.cap is not None and self.cap.isOpened()

    def start_capture(self):
        """
        Start video capture from the default camera.
        This function initializes the video capture object and sets the frame width, height, and FPS.
        """
        if not self.is_capturing():
            self.cap = cv2.VideoCapture(0)  # Try opening the camera
            if not self.is_capturing():
                logger.error("Could not open camera.")
                return

            # Set properties after successfully opening
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            logger.info("Video capture started successfully.")

from django.core.management import execute_from_command_line

if __name__ == "__main__":
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myapp.settings')
    execute_from_command_line(['manage.py', 'runserver'])
