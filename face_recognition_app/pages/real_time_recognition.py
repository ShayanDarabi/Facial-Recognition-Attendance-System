import streamlit as st
import cv2
import numpy as np
import pickle
import face_recognition
import threading
from PIL import Image
import os
import base64

# Custom CSS to hide the Streamlit header & footer
st.markdown("""
    <style>
        /* Hide Streamlit's default menu and footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Custom CSS
st.markdown("""
    <style>
        /* Move sidebar to the right */
        section[data-testid="stSidebar"] {
            position: fixed;
            right: 0;
            top: 0;
            height: 100%;
            width: 250px; /* Adjust width if needed */
            background-color: #262730; /* Customize sidebar color */
        }

        /* Center content */
        .stApp {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background-size: cover;
        }

        /* Make text readable */
        .stTextInput, .stButton {
            font-size: 18px !important;
        }

        /* Center align title */
        h1 {
            text-align: center;
            color: white;
            font-weight: bold;
        }

        /* Adjust input box width */
        div[data-baseweb="input"] {
            width: 50% !important;
            margin: auto;
        }

        /* Adjust button positioning */
        .stButton>button {
            display: block;
            margin: auto;
            font-size: 16px;
            padding: 10px 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Background image path
background_image_path = "/mnt/d/University/Master/1st Sem/Data Mining Algorithms and Applications/Face Recognition/Background Images/IEUT.jpeg"

# Function to set the background image and ensure it covers the whole page
def set_background():
    with open(background_image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    background_style = f"""
    <style>
    .stApp {{
        background-image: url('data:image/jpeg;base64,{encoded_image}');
        background-size: cover;
        background-position: center;
    }}
    .input-overlay{{
        display: block;
        background: rgba(0, 0, 0, 0.5); /* Semi-transparent black */
        color: white;
        padding: 5px; /* Reduced padding */
        border-radius: 10px;
        margin-bottom: -10px; /* Reduced bottom margin */
        text-align: center;
    }}
    .stTextInput, .stFileUploader {{
        margin-top: -15px; /* Reduce space above inputs */
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

# Function to encode image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


# Set the background image
set_background()


# Get the absolute path of the current script (real_time_recognition.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the path to the encodings file within the same directory
ENCODINGS_FILE = os.path.join(SCRIPT_DIR, "face_encodings.pkl")

with open(ENCODINGS_FILE, "rb") as f:
    known_face_encodings, known_face_names = pickle.load(f)

# Streamlit Page Config
st.title("📷 Real-Time Face Recognition")

# Input field for IP camera URL
st.markdown(
    """
    <style>
    .input-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
    }

    .input-overlay {
        display: block;
        width: 60%;
        background: rgba(0, 0, 0, 0.5); /* Semi-transparent black */
        color: white;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-size: 16px;
        font-weight: bold;
    }

    .custom-input {
        width: 60%;
        height: 40px;
        padding: 10px;
        font-size: 16px;
        border-radius: 5px;
        border: none;
        outline: none;
        text-align: center;
        background-color: #222;
        color: white;
        margin-top: 5px;
    }
    </style>
    
    <div class="input-container">
        <div class="input-overlay">Enter IP Camera URL</div>
    </div>
    """,
    unsafe_allow_html=True
)

ip_camera_url = st.text_input("", "http://192.168.43.1:8080/video")

# Start button
start_recognition = st.button("Start Recognition")

if start_recognition:
    stframe = st.empty()
    video_capture = cv2.VideoCapture(ip_camera_url)
    # Set camera resolution (Change only these two lines)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_lock = threading.Lock()
    latest_frame = None
    
    def capture_frame():
        global latest_frame
        while True:
            ret, frame = video_capture.read()
            if ret:
                with frame_lock:
                    latest_frame = frame
    
    threading.Thread(target=capture_frame, daemon=True).start()
    
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            name = "Unknown"
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            
            if len(distances) > 0:
                best_match_index = np.argmin(distances)
                if distances[best_match_index] < 0.4:
                    name = known_face_names[best_match_index]

            top, right, bottom, left = [int(coord * 2) for coord in face_location]
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Convert frame to PIL Image and display in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(Image.fromarray(frame_rgb), channels="RGB")
    
    video_capture.release()


    
