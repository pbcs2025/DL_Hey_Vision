import os
import sys
import tempfile
import time
from collections import Counter

import cv2
import numpy as np
import streamlit as st

# ---------------------------------------------------------
# Project setup & path alignment
# ---------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.assistant_pipeline import process_query, detect_objects
from src.vision.detection_visualizer import draw_detections, get_scene_summary_data
from src.vision.live_camera import WebcamStream
from src.voice.speech_to_text import listen
from src.voice.text_to_speech import speak
from src.voice.query_normalizer import normalize_query

# Frame storage paths
LIVE_FRAME_PATH = os.path.join(PROJECT_ROOT, "data", "test", "live_ui_frame.jpg")
UPLOAD_FRAME_PATH = os.path.join(PROJECT_ROOT, "data", "test", "upload_ui_frame.jpg")

os.makedirs(os.path.dirname(LIVE_FRAME_PATH), exist_ok=True)


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Vision Assistant",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# Shared Webcam Stream Singleton
# ---------------------------------------------------------

@st.cache_resource
def get_webcam_stream():
    """Maintain a single background webcam stream for Streamlit."""
    return WebcamStream(camera_index=0)


# ---------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "last_response" not in st.session_state:
    st.session_state.last_response = ""

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "voice_active" not in st.session_state:
    st.session_state.voice_active = False


# ---------------------------------------------------------
# Custom CSS for Modern Academic & Accessible UI
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    /* Container styling */
    .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* Main Title Styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #FFFFFF;
        margin-bottom: 0.1rem;
    }

    .main-subtitle {
        font-size: 1.1rem;
        font-weight: 500;
        color: #3498DB;
        margin-bottom: 1.2rem;
    }

    /* Segmented Control styling */
    div[data-baseweb="tab-highlight"] {
        background-color: #3498DB !important;
    }

    /* Card Containers */
    .panel-card {
        background-color: #1E222A;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    .panel-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ECF0F1;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Scene Summary Pills */
    .summary-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }

    .summary-pill {
        background-color: #2C3E50;
        color: #ECF0F1;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.95rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    /* Assistant Response Box */
    .assistant-box {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 2px solid #38BDF8;
        border-radius: 14px;
        padding: 1.4rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }

    .assistant-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #38BDF8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.5rem;
    }

    .assistant-text {
        font-size: 1.3rem;
        font-weight: 600;
        color: #F8FAFC;
        line-height: 1.6;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 650;
        transition: all 0.2s ease-in-out;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
    }

    /* Helper text */
    .helper-text {
        font-size: 0.9rem;
        color: #94A3B8;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.markdown("### 👁️ AI Vision Assistant")
    st.write(
        """
        Voice-first multimodal AI assistant designed for visually impaired users.
        Analyzes live camera feeds or uploaded images using Deep Learning.
        """
    )

    st.divider()

    st.markdown("#### Capabilities")
    st.markdown(
        """
        • **Object Identification & Counting**  
        • **Spatial & Directional Reasoning**  
        • **Front Object Detection**  
        • **OCR Sign & Text Reading**  
        • **Obstacle & Risk Analysis (DQN)**  
        • **Scene Summarization & Description**  
        • **Approaching Motion Detection**  
        """
    )

    st.divider()

    st.caption("Deep Learning Mini-Project | VII Semester")


# ---------------------------------------------------------
# Header Title
# ---------------------------------------------------------

st.markdown('<div class="main-title">👁️ AI Vision Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">See. Understand. Assist.</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# Mode Switcher (Live Camera vs Upload Image)
# ---------------------------------------------------------

mode = st.radio(
    "Select Operating Mode",
    options=["📷 LIVE CAMERA", "🖼 UPLOAD IMAGE"],
    horizontal=True,
    key="selected_mode",
    label_visibility="collapsed"
)

st.divider()


# ---------------------------------------------------------
# Layout Split: Media Feed (Left) & Controls (Right)
# ---------------------------------------------------------

col_media, col_controls = st.columns([1.35, 1], gap="large")

active_image_path = None
active_detections = []


# =========================================================
# MODE 1 — LIVE CAMERA
# =========================================================

if mode == "📷 LIVE CAMERA":

    with col_media:
        st.markdown('<div class="panel-header">📷 Live Camera Feed</div>', unsafe_allow_html=True)

        stream = get_webcam_stream()

        if not stream.is_available():
            st.warning("⚠️ Camera unavailable or in use by another application. Switch to Upload Image mode to test.")
        else:
            # Container for real-time video stream
            video_placeholder = st.empty()
            
            # Read frame
            frame = stream.get_frame()
            if frame is not None:
                # Save frame for reasoning pipeline
                cv2.imwrite(LIVE_FRAME_PATH, frame)
                active_image_path = LIVE_FRAME_PATH

                # Run YOLO detection for visual overlay
                active_detections = detect_objects(frame, conf=0.25)

                # Draw bounding boxes & colored labels
                annotated_frame = draw_detections(frame, active_detections)

                # Convert BGR to RGB for Streamlit
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Render live video feed
                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            # Auto-refresh live camera feed
            st.caption("🟢 Camera active & continuously updating detections.")


# =========================================================
# MODE 2 — UPLOAD IMAGE
# =========================================================

elif mode == "🖼 UPLOAD IMAGE":

    with col_media:
        st.markdown('<div class="panel-header">🖼 Upload Image Scene</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose an image (JPG, PNG, JPEG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            # Read image file buffer into OpenCV array
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is not None:
                # Save to disk for reasoning pipeline
                cv2.imwrite(UPLOAD_FRAME_PATH, img)
                active_image_path = UPLOAD_FRAME_PATH

                # Run YOLO detection for visual overlay
                active_detections = detect_objects(img, conf=0.25)

                # Draw bounding boxes & colored labels
                annotated_img = draw_detections(img, active_detections)

                # Convert BGR to RGB
                rgb_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

                # Display annotated image
                st.image(rgb_img, channels="RGB", use_container_width=True)
            else:
                st.error("Could not decode the uploaded image.")
        else:
            st.info("👆 Upload an image above to analyze with the AI Vision Assistant.")


# =========================================================
# RIGHT COLUMN — SCENE SUMMARY & VOICE CONTROLS
# =========================================================

with col_controls:

    # -----------------------------------------------------
    # Current Scene Summary
    # -----------------------------------------------------
    st.markdown('<div class="panel-header">📊 Current Scene Summary</div>', unsafe_allow_html=True)

    summary_items = get_scene_summary_data(active_detections)

    if summary_items:
        pills_html = '<div class="summary-grid">'
        for item in summary_items:
            pills_html += f'<div class="summary-pill">{item["icon"]} {item["object"]} × {item["count"]}</div>'
        pills_html += '</div>'
        st.markdown(pills_html, unsafe_allow_html=True)
    else:
        st.caption("No objects currently detected in view.")

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # Voice & Query Controls
    # -----------------------------------------------------
    st.markdown('<div class="panel-header">🎤 Ask the Assistant</div>', unsafe_allow_html=True)

    # Primary Interaction: Voice
    if st.button("🎤 Speak Question", use_container_width=True, type="primary"):
        with st.spinner("🎤 Listening... Speak now..."):
            recognized_text = listen()

        if recognized_text:
            st.session_state.last_query = recognized_text
            st.success(f"🗣️ You said: \"{recognized_text}\"")

            # Execute Query Pipeline automatically on voice input
            if active_image_path and os.path.exists(active_image_path):
                norm_query = normalize_query(recognized_text)
                with st.spinner("🧠 Thinking & Analyzing Scene..."):
                    result = process_query(norm_query, active_image_path)

                st.session_state.last_result = result
                response_text = result.get("response", "No response generated.")
                st.session_state.last_response = response_text

                # Automatically speak response via TTS
                speak(response_text)
            else:
                st.warning("No image/camera frame available for analysis.")
        else:
            st.warning("No speech detected. Please try speaking again.")

    st.markdown('<div class="helper-text" style="text-align: center; margin-top: 0.3rem;">Primary voice-first interaction</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Fallback Interaction: Manual Text Input
    typed_query = st.text_input(
        "Type your question (fallback/demo)",
        placeholder="e.g. What is in front of me?",
        label_visibility="collapsed"
    )

    if st.button("🔍 Submit Text Question", use_container_width=True):
        if not typed_query.strip():
            st.warning("Please type a question first.")
        elif not active_image_path or not os.path.exists(active_image_path):
            st.warning("No camera frame or uploaded image available.")
        else:
            st.session_state.last_query = typed_query
            norm_query = normalize_query(typed_query)

            with st.spinner("🧠 Analyzing Scene..."):
                result = process_query(norm_query, active_image_path)

            st.session_state.last_result = result
            response_text = result.get("response", "No response generated.")
            st.session_state.last_response = response_text

            # Automatically speak response via TTS
            speak(response_text)


# =========================================================
# ASSISTANT RESPONSE DISPLAY
# =========================================================

if st.session_state.last_response:
    st.markdown(
        f"""
        <div class="assistant-box">
            <div class="assistant-title">🤖 Assistant Response</div>
            <div class="assistant-text">"{st.session_state.last_response}"</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_listen, col_space = st.columns([1, 2])
    with col_listen:
        if st.button("🔊 Listen to Response Again", use_container_width=True):
            speak(st.session_state.last_response)


# =========================================================
# TECHNICAL DETAILS (EXPANDABLE SECTION)
# =========================================================

if st.session_state.last_result:
    res = st.session_state.last_result

    with st.expander("🛠️ Technical Details & Model Reasoning Diagnostics"):
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.markdown("##### NLP Intent Classification")
            st.write(f"**Query:** `{res.get('query', '')}`")
            st.write(f"**Intent:** `{res.get('intent', 'UNKNOWN')}`")
            conf_pct = res.get('intent_confidence', 0.0) * 100
            st.write(f"**Intent Confidence:** `{conf_pct:.1f}%`")

        with col_t2:
            st.markdown("##### YOLO Vision Detections")
            dets = res.get("detections", [])
            st.write(f"**Total Detections:** `{len(dets)}`")

            if dets:
                for idx, d in enumerate(dets[:5]):
                    st.write(f"• **{d['object']}** ({d['position']}) — conf: `{int(d['confidence']*100)}%`")
                if len(dets) > 5:
                    st.caption(f"...and {len(dets)-5} more detections.")

        # Additional diagnostic outputs
        events = res.get("movement_events", [])
        if events:
            st.markdown("##### Temporal Movement Tracking")
            st.json(events)