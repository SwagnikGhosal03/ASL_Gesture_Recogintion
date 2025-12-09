import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import time
from collections import deque

ASL_LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'Nothing', 'O', 'P', 'Q', 'R', 'S', 'Space', 'T', 'U', 'V', 
    'W', 'X', 'Y', 'Z'
]

class HandDetector:
    def __init__(self):
        """Initialize MediaPipe Hand detector with optimized settings"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,  
            model_complexity=1
        )
        self.last_bbox = None
        self.tracking_failed_count = 0
        self.max_tracking_fails = 3 
        self.smooth_bbox = None
        self.smoothing_factor = 0.7  
    
    def detect_hand_bbox(self, image):
        """Detect hand and return bounding box coordinates with improved tracking"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        hand_landmarks = None
        bbox = None
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get all landmark coordinates
            h, w, _ = image.shape
            landmarks = []
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append((x, y))
            
            # Calculate bounding box with padding
            x_coords = [lm[0] for lm in landmarks]
            y_coords = [lm[1] for lm in landmarks]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            

            padding_x = int((x_max - x_min) * 0.3)
            padding_y = int((y_max - y_min) * 0.3)
            
            x_min = max(0, x_min - padding_x)
            x_max = min(w, x_max + padding_x)
            y_min = max(0, y_min - padding_y)
            y_max = min(h, y_max + padding_y)
            
 
            new_bbox = (x_min, y_min, x_max, y_max)
            if self.smooth_bbox is None:
                self.smooth_bbox = new_bbox
            else:

                smooth_xmin = int(self.smooth_bbox[0] * self.smoothing_factor + x_min * (1 - self.smoothing_factor))
                smooth_ymin = int(self.smooth_bbox[1] * self.smoothing_factor + y_min * (1 - self.smoothing_factor))
                smooth_xmax = int(self.smooth_bbox[2] * self.smoothing_factor + x_max * (1 - self.smoothing_factor))
                smooth_ymax = int(self.smooth_bbox[3] * self.smoothing_factor + y_max * (1 - self.smoothing_factor))
                self.smooth_bbox = (smooth_xmin, smooth_ymin, smooth_xmax, smooth_ymax)
            
            bbox = self.smooth_bbox
            self.last_bbox = bbox
            self.tracking_failed_count = 0
            
        else:
            if self.last_bbox and self.tracking_failed_count < self.max_tracking_fails:
                bbox = self.last_bbox
                self.tracking_failed_count += 1
            else:
                self.last_bbox = None
                self.smooth_bbox = None
                self.tracking_failed_count = 0
        
        return bbox, hand_landmarks
    
    def draw_hand_bbox(self, image, bbox, landmarks=None):
        """Draw bounding box and hand landmarks on image"""
        if bbox is None:
            return image
            
        x_min, y_min, x_max, y_max = bbox
        

        if self.tracking_failed_count > 0:

            color = (0, 255, 255)
            thickness = 2
            label = "TRACKING"
        else:

            color = (0, 255, 0)
            thickness = 3
            label = "LIVE DETECTION"
        
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        

        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(image, (x_min, y_min - label_size[1] - 10), 
                     (x_min + label_size[0], y_min), color, -1)
        cv2.putText(image, label, (x_min, y_min - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        

        if landmarks:
            self.mp_drawing.draw_landmarks(image, landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return image

class ASLPredictor:
    def __init__(self, model_path):
        """Initialize the ASL predictor with the trained model"""
        try:
            self.model = load_model(model_path)
            st.success("✅ Model loaded successfully!")
            self.img_size = self.model.input_shape[1:3]
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            self.model = None
            self.img_size = (64, 64)
        
        # Initialize hand detector
        self.hand_detector = HandDetector()
    
    def preprocess_image(self, image):
        """Preprocess image for model prediction"""
        try:
            img_array = np.array(image)
            
            if img_array.shape[-1] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            elif len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            
            img_array = cv2.resize(img_array, self.img_size)
            img_array = img_array.astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        except Exception as e:
            st.error(f"Error preprocessing image: {e}")
            return None
    
    def detect_and_crop_hand(self, image):
        """Detect hand in image and return cropped hand region"""
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        bbox, landmarks = self.hand_detector.detect_hand_bbox(img_bgr)
        
        if bbox:
            x_min, y_min, x_max, y_max = bbox
            
            # Ensure bbox has valid dimensions
            if x_max - x_min > 20 and y_max - y_min > 20:  # Increased minimum size
                # Crop hand region
                hand_crop = img_bgr[y_min:y_max, x_min:x_max]
                
                # Draw bounding box on original image
                annotated_image = self.hand_detector.draw_hand_bbox(img_bgr.copy(), bbox, landmarks)
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                
                return hand_crop, annotated_image, bbox
        
        # Return original image if no valid hand detected
        annotated_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return None, annotated_image, None
    
    def predict(self, image):
        """Predict ASL sign from image"""
        if self.model is None:
            return "Model not loaded", 0.0, None, None
        
        # Detect and crop hand
        hand_crop, annotated_image, bbox = self.detect_and_crop_hand(image)
        
        if hand_crop is None:
            return "No hand detected", 0.0, annotated_image, bbox
        
        # Ensure hand crop is valid
        if hand_crop.size == 0:
            return "Invalid hand region", 0.0, annotated_image, bbox
        
        # Convert hand crop to PIL Image for preprocessing
        hand_crop_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
        hand_crop_pil = Image.fromarray(hand_crop_rgb)
        
        processed_image = self.preprocess_image(hand_crop_pil)
        if processed_image is None:
            return "Error processing image", 0.0, annotated_image, bbox
        
        try:
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            
            return ASL_LABELS[predicted_class_idx], confidence, annotated_image, bbox
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return "Prediction error", 0.0, annotated_image, bbox

class WordBuilder:
    def __init__(self):
        self.current_word = []
        self.word_history = []
        self.last_prediction_time = 0
        self.prediction_cooldown = 2.0  # Reduced cooldown for better responsiveness
        self.min_confidence = 0.7
        self.last_prediction = None
        self.stable_prediction_count = 0
        self.stable_threshold = 2  # Require 2 consistent predictions
    
    def add_letter(self, letter, confidence):
        """Add a letter to the current word if conditions are met"""
        current_time = time.time()
        
        # Filter out unwanted predictions
        if letter in ["Nothing", "No hand detected", "Prediction error", "Invalid hand region"]:
            self.stable_prediction_count = 0
            return False
        
        # Check confidence threshold
        if confidence < self.min_confidence:
            self.stable_prediction_count = 0
            return False
        
        # Check for stable prediction
        if letter == self.last_prediction:
            self.stable_prediction_count += 1
        else:
            self.stable_prediction_count = 1
            self.last_prediction = letter
            return False
        
        # Check if prediction is stable enough
        if self.stable_prediction_count < self.stable_threshold:
            return False
        
        # Check cooldown period
        if current_time - self.last_prediction_time < self.prediction_cooldown:
            return False
        
        # Don't add duplicate consecutive letters (except Space)
        if self.current_word and letter == self.current_word[-1] and letter != "Space":
            return False
        
        self.current_word.append(letter)
        self.last_prediction_time = current_time
        self.stable_prediction_count = 0  # Reset after adding
        return True
    
    def add_space(self):
        """Add a space to the current word"""
        if self.current_word and self.current_word[-1] != "Space":
            self.current_word.append("Space")
            self.last_prediction_time = time.time()
            return True
        return False
    
    def backspace(self):
        """Remove the last character"""
        if self.current_word:
            self.current_word.pop()
            self.last_prediction_time = time.time()
            return True
        return False
    
    def clear_word(self):
        """Clear the current word without saving to history"""
        self.current_word = []
        self.last_prediction_time = time.time()
    
    def save_word(self):
        """Save current word to history and clear"""
        if self.current_word:
            current_word_str = self.get_current_word()
            if current_word_str.strip():  # Only save non-empty words
                self.word_history.append(current_word_str)
            self.current_word = []
        self.last_prediction_time = time.time()
    
    def get_current_word(self):
        """Get the current word as a string"""
        if not self.current_word:
            return ""
        
        word_string = ""
        for char in self.current_word:
            if char == "Space":
                word_string += " "
            else:
                word_string += char
        return word_string.strip()
    
    def get_word_display(self):
        """Get formatted display of current word"""
        display_text = ""
        for char in self.current_word:
            if char == "Space":
                display_text += " ␣ "
            else:
                display_text += char
        return display_text.strip() if display_text else "Start signing..."

def main():
    st.set_page_config(
        page_title="ASL Recognition System",
        page_icon="✋",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .confidence-high {
        background: linear-gradient(90deg, #4ecdc4, #44a08d);
        height: 25px;
        border-radius: 12px;
        margin: 10px 0;
    }
    .confidence-medium {
        background: linear-gradient(90deg, #ffd89b, #ff9d6c);
        height: 25px;
        border-radius: 12px;
        margin: 10px 0;
    }
    .confidence-low {
        background: linear-gradient(90deg, #ff6b6b, #ff8e8e);
        height: 25px;
        border-radius: 12px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #2e1739;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 1rem 0;
    }
    .detection-box {
        border: 3px solid #00ff00;
        border-radius: 10px;
        padding: 5px;
        margin: 10px 0;
    }
    .word-display {
        background-color: #2ecc71;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        min-height: 80px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .history-box {
        background-color:#2e1739;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #6c757d;
        margin: 1rem 0;
    }
    .tracking-indicator {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem 0;
    }
    .tracking-good {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .tracking-fair {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .tracking-poor {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .controls-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #dee2e6;
        margin: 1rem 0;
    }
    .control-btn {
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">✋ Real-time ASL Recognition</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Input Method",
        ["🏠 Home", "📁 Image Upload", "📷 Real-time Webcam", "ℹ️ About"]
    )
    
    # Initialize predictor and word builder
    @st.cache_resource
    def load_predictor():
        model_path = "sign_language_mobilenet.h5"  
        return ASLPredictor(model_path)
    
    predictor = load_predictor()
    
    # Initialize session state for word builder
    if 'word_builder' not in st.session_state:
        st.session_state.word_builder = WordBuilder()
    
    if app_mode == "🏠 Home":
        home_interface()
    elif app_mode == "📁 Image Upload":
        image_upload_interface(predictor)
    elif app_mode == "📷 Real-time Webcam":
        realtime_webcam_interface(predictor)
    elif app_mode == "ℹ️ About":
        about_interface()

def home_interface():
    """Home page with overview"""
    st.header("Welcome to ASL Recognition System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Real-time American Sign Language Recognition
        
        This application uses **deep learning** to recognize American Sign Language 
        (ASL) signs from images or webcam captures.
        **Features:**
        - Recognize 28 ASL signs (A-Z, Space, Nothing)
        - Upload images of ASL signs
        - Real-time webcam capture with improved hand detection
        - Word and sentence formation with stable predictions
        - High accuracy predictions with confidence scoring
        
        **Get Started:**
        1. Use **Image Upload** to analyze existing photos
        2. Use **Real-time Webcam** for live sign recognition and word building
        3. View instant predictions and build words in real-time
        """)
    
    with col2:
        st.markdown("""
        ### 🔠 Supported Signs
        """)
        # Display all letters in a compact grid
        cols = st.columns(4)
        for i, letter in enumerate(ASL_LABELS):
            with cols[i % 4]:
                st.info(f"**{letter}**")
    
    st.markdown("---")
    
    # Quick demo section
    st.subheader("🚀 Quick Demo")
    if st.button("Try Real-time Recognition", type="primary"):
        st.info("Go to **Real-time Webcam** to start building words with your signs!")

def image_upload_interface(predictor):
    """Interface for image upload"""
    st.header("📁 Upload ASL Sign Image")
    
    st.markdown("""
    <div class="info-box">
    💡 <strong>Tips for best results:</strong><br>
    • Clear hand visibility<br>
    • Good lighting conditions<br>
    • Plain background<br>
    • Proper ASL sign formation<br>
    • <strong>Hand will be automatically detected and cropped</strong>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image containing an ASL sign"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        original_image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Original Image")
            st.image(original_image, caption="Uploaded Image", use_column_width=True)
        
        # Process image
        if st.button("🔍 Detect Hand and Analyze", type="primary"):
            if predictor.model is not None:
                with st.spinner("🔄 Detecting hand and analyzing sign..."):
                    prediction, confidence, annotated_image, bbox = predictor.predict(original_image)
                
                with col2:
                    st.subheader("🎯 Detection Results")
                    
                    if bbox:
                        st.markdown('<div class="detection-box">', unsafe_allow_html=True)
                        st.image(annotated_image, caption="Hand Detected ✅", use_column_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Show cropped hand
                        hand_crop, _, _ = predictor.detect_and_crop_hand(original_image)
                        if hand_crop is not None:
                            hand_crop_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
                            st.image(hand_crop_rgb, caption="Cropped Hand Region", use_column_width=True)
                    else:
                        st.warning("❌ No hand detected in the image")
                        st.image(annotated_image, caption="No Hand Detected", use_column_width=True)
                
                display_prediction_results(prediction, confidence, original_image, bbox, predictor)
            else:
                st.error("Model not loaded. Please check if 'sign_language_mobilenet.h5' exists.")

def realtime_webcam_interface(predictor):
    """Interface for real-time webcam processing with word building"""
    st.header("📷 Real-time ASL Recognition & Word Building")
    
    st.markdown("""
    <div class="info-box">
    📝 <strong>Improved Real-time Instructions:</strong><br>
    1. Click "Start Webcam" to begin<br>
    2. Show your hand clearly in the frame<br>
    3. <strong>Green box = Live detection, Yellow box = Tracking</strong><br>
    4. Hold signs steady for 1-2 seconds for better recognition<br>
    5. Use controls below to manage your word building<br>
    6. Click "Stop Webcam" when finished
    </div>
    """, unsafe_allow_html=True)
    
    # Display current word
    current_word_display = st.session_state.word_builder.get_word_display()
    st.markdown(f'<div class="word-display">{current_word_display}</div>', unsafe_allow_html=True)
    
    # Display word history
    if st.session_state.word_builder.word_history:
        st.subheader("📚 Word History")
        for i, word in enumerate(reversed(st.session_state.word_builder.word_history[-5:]), 1):
            st.markdown(f'<div class="history-box">{i}. {word}</div>', unsafe_allow_html=True)
    
    # Webcam processing
    st.subheader("🎥 Live Webcam Feed")
    
    # Webcam configuration
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### ⚙️ Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.7, 
            step=0.1,
            help="Minimum confidence required to add a letter"
        )
        st.session_state.word_builder.min_confidence = confidence_threshold
        
        processing_mode = st.selectbox(
            "Processing Mode",
            options=["Balanced (2 FPS)", "Fast (3 FPS)", "Accurate (1 FPS)"],
            index=0,
            help="Balanced mode works best with improved detection"
        )
        
        fps_map = {"Balanced (2 FPS)": 2, "Fast (3 FPS)": 3, "Accurate (1 FPS)": 1}
        target_fps = fps_map[processing_mode]
        
        # Detection settings
        st.markdown("### 🎯 Detection")
        show_landmarks = st.checkbox("Show Hand Landmarks", value=True)
        show_stability = st.checkbox("Show Prediction Stability", value=True)
    
    # Initialize webcam
    run_webcam = st.checkbox("🎥 Start Webcam", value=False)
    FRAME_WINDOW = st.image([])
    
    st.markdown("---")
    st.subheader("🔤 Word Building Controls")
    
    st.markdown('<div class="controls-container">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Clear Word", use_container_width=True, type="primary", key="clear_btn"):
            st.session_state.word_builder.clear_word()
            st.rerun()
    
    with col2:
        if st.button("Backspace", use_container_width=True, type="secondary", key="backspace_btn"):
            if st.session_state.word_builder.backspace():
                st.rerun()
    
    with col3:
        if st.button("Add Space", use_container_width=True, type="secondary", key="space_btn"):
            if st.session_state.word_builder.add_space():
                st.rerun()
    
    with col4:
        if st.button("💾 Save Word", use_container_width=True, type="secondary", key="save_btn"):
            st.session_state.word_builder.save_word()
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if run_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Could not access webcam. Please check permissions.")
            return
        
        # Set camera resolution for better detection
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        st.info("🔴 Webcam is active. You can use the controls above while recording!")
        
        # Prediction state
        last_prediction_time = 0
        prediction_interval = 1.0 / target_fps
        frame_count = 0
        
        # Processing loop
        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to capture frame")
                break
            
            current_time = time.time()
            frame_count += 1
            
            # Process frame for display
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Perform prediction at intervals
            if current_time - last_prediction_time >= prediction_interval:
                # Convert frame to PIL Image for prediction
                pil_image = Image.fromarray(display_frame)
                
                # Get prediction
                prediction, confidence, annotated_image, bbox = predictor.predict(pil_image)
                
                # Update display frame with annotations
                if annotated_image is not None:
                    display_frame = annotated_image
                
                # Add prediction info to frame
                cv2.putText(display_frame, f"Pred: {prediction}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Conf: {confidence:.2f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add stability information
                if show_stability:
                    stability = st.session_state.word_builder.stable_prediction_count
                    cv2.putText(display_frame, f"Stable: {stability}/2", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Try to add letter to word builder
                if prediction not in ["No hand detected", "Prediction error", "Invalid hand region"]:
                    added = st.session_state.word_builder.add_letter(prediction, confidence)
                    if added:
                        # Visual feedback for added letter
                        cv2.putText(display_frame, "LETTER ADDED!", (10, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                last_prediction_time = current_time
            
            # Display current word on frame
            current_word = st.session_state.word_builder.get_current_word()
            cv2.putText(display_frame, f"Word: {current_word}", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show detection status on frame
            status_text = "Detection: "
            if bbox:
                if predictor.hand_detector.tracking_failed_count > 0:
                    status_text += "TRACKING"
                    color = (0, 255, 255)  # Yellow
                else:
                    status_text += "LIVE"
                    color = (0, 255, 0)    # Green
            else:
                status_text += "NONE"
                color = (0, 0, 255)        # Red
            
            cv2.putText(display_frame, status_text, (10, frame.shape[0] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show frame
            FRAME_WINDOW.image(display_frame)
        
        cap.release()
    
    if not run_webcam and 'cap' in locals():
        cap.release()
        st.success("✅ Webcam stopped")

def display_prediction_results(prediction, confidence, original_image, bbox, predictor):
    """Display prediction results in a formatted way"""
    st.markdown("---")
    st.header("🎯 Prediction Results")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if bbox:

            hand_crop, annotated_image, _ = predictor.detect_and_crop_hand(original_image)
            st.image(annotated_image, caption="Hand Detection", use_column_width=True)
        else:
            st.image(original_image, caption="Input Image", use_column_width=True)
    
    with col2:
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        

        if prediction == "Nothing":
            st.subheader("🤚 No Sign Detected")
            st.info("Neutral hand position detected")
        elif prediction == "Space":
            st.subheader("␣ Space Character")
            st.info("Space sign detected")
        elif prediction == "No hand detected":
            st.subheader("❌ No Hand Detected")
            st.error("Please ensure your hand is clearly visible in the image")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        elif prediction == "Prediction error":
            st.subheader("❌ Prediction Error")
            st.error("There was an error processing your image")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        else:
            st.subheader(f"🔤 Predicted: **{prediction}**")
            st.success(f"Letter **{prediction}** recognized")
        

        confidence_percent = confidence * 100
        st.write(f"**Confidence:** {confidence_percent:.1f}%")
        

        if confidence >= 0.8:
            bar_class = "confidence-high"
            confidence_text = "🟢 High Confidence"
        elif confidence >= 0.6:
            bar_class = "confidence-medium"
            confidence_text = "🟡 Medium Confidence"
        else:
            bar_class = "confidence-low"
            confidence_text = "🔴 Low Confidence"
        
        st.markdown(f"""
        <div style="margin: 15px 0;">
            <div style="width: 100%; background-color: #e0e0e0; border-radius: 12px;">
                <div class="{bar_class}" style="width: {confidence_percent}%;"></div>
            </div>
            <div style="text-align: center; margin-top: 8px; font-weight: bold;">
                {confidence_text} ({confidence_percent:.1f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Word building for single images
        if prediction not in ["Nothing", "No hand detected", "Prediction error"] and confidence >= 0.7:
            st.subheader("🔤 Add to Word")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button(f"Add '{prediction}' to Word", type="primary"):
                    st.session_state.word_builder.add_letter(prediction, confidence)
                    st.rerun()
            
            with col2:
                if st.button("Add Space"):
                    st.session_state.word_builder.add_space()
                    st.rerun()
            
            with col3:
                if st.button("Clear Word"):
                    st.session_state.word_builder.clear_word()
                    st.rerun()
            

            current_word = st.session_state.word_builder.get_word_display()
            if current_word:
                st.markdown(f'<div class="word-display" style="font-size: 1.5rem;">{current_word}</div>', unsafe_allow_html=True)
        
        with st.expander("📊 Detailed Information", expanded=False):
            st.write(f"**Predicted Class:** {prediction}")
            st.write(f"**Confidence Score:** {confidence:.4f}")
            st.write(f"**Model:** MobileNet with Improved Hand Detection")
            st.write(f"**Hand Detected:** {'Yes' if bbox else 'No'}")
            
            if confidence < 0.7:
                st.warning("""
                ⚠️ **Suggestions for better results:**
                - Ensure hand is clearly visible and centered
                - Use good, even lighting
                - Try with a plain background
                - Make sure the sign is properly formed
                - Keep hand within the detection box
                - Retake the photo if needed
                """)

def about_interface():
    """About page with information about the project"""
    st.header("ℹ️ About ASL Recognition System")
    
    st.markdown("""
    ## American Sign Language Recognition with Improved Real-time Detection
    
    This application uses a deep learning model based on **MobileNet** combined with 
    **Enhanced MediaPipe Hand Detection** to recognize American Sign Language (ASL) signs.

    ### ✨ Improved Detection Features
    - **Consistent Hand Tracking**: Maintains detection during brief occlusions
    - **Optimized Timing**: Better prediction intervals for medium FPS
    - **Stable Predictions**: Requires consistent signs before adding to words
    - **Visual Feedback**: Different colors for live detection vs tracking
    - **Smoothing**: Smooth bounding box movements for better stability

    ### 🎯 Enhanced Features
    - **28 ASL Classes**: A-Z, Space, and Nothing (neutral position)
    - **Real-time Processing**: Improved webcam feed with consistent detection
    - **Word & Sentence Building**: Stable word formation with confidence filtering
    - **Smart Hand Detection**: Enhanced MediaPipe with tracking fallback and smoothing
    - **Better Controls**: Real-time word editing during recording

    ### 🔧 Technical Improvements
    - **Increased Confidence Thresholds**: Better detection consistency (0.7)
    - **Bounding Box Smoothing**: Smooth movements for stable detection
    - **Stable Prediction System**: Requires 2 consistent predictions before adding letters
    - **Optimized FPS Settings**: Balanced (2 FPS), Fast (3 FPS), Accurate (1 FPS)
    - **Better Visual Feedback**: Color-coded detection status with labels

    ### 🚀 How to Use Improved Real-time Mode
    1. **Start Webcam**: Enable camera access with optimized settings
    2. **Show Signs**: Present ASL signs steadily for 1-2 seconds
    3. **Monitor Detection**: Green box = Live detection, Yellow box = Tracking
    4. **Build Words**: Letters auto-add after stable detection
    5. **Use Controls**: Edit words in real-time with backspace and other controls

    The improved system provides more consistent hand detection and better timing 
    for medium FPS processing, making ASL recognition more reliable and user-friendly.
    """)
    
    st.markdown("---")
    st.caption("Developed with ❤️ using TensorFlow, MediaPipe and Streamlit")

if __name__ == "__main__":
    main()