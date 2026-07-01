import streamlit as st
import base64


st.set_page_config(page_title="Facial Recognition System Development", layout="wide")
# Custom CSS to hide the Streamlit header & footer
st.markdown("""
    <style>
        /* Hide Streamlit's default menu and footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

image_path = "Fani.jpg"  # Change this to the path of your local image
base64_image = get_base64_image(image_path)


# Add background image
st.markdown(f"""
    <style>
    .stApp {{
        background: url("data:image/jpeg;base64,{base64_image}") no-repeat center center fixed;
        background-size: cover;
    }}
    .centered-text {{
        font-size: 18px;
        font-weight: bold;
        color: #FFFFFF; /* White text */
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7); /* Black shadow for better readability */
        padding: 10px;
        background: rgba(0, 0, 0, 0.5); /* Semi-transparent dark overlay */
        border-radius: 10px;
    }}
    h2 {{
        color: #FFFFFF;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
    }}
    p {{
        color: #EAEAEA;
    }}
    .image-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
        }}
    .image-caption {{
            text-align: center;
            font-weight: bold;
            color: #ADD8E6;
            background: rgba(0, 0, 0, 0.6);
            padding: 5px;
            border-radius: 5px;
        }}
    </style>
""", unsafe_allow_html=True)

st.title("Facial Recognition System Development Process")

def display_section(title, content):
    st.markdown(f"""
        <div class='centered-text'>
            <h2>{title}</h2>
            <p>{content}</p>
        </div>
    """, unsafe_allow_html=True)

display_section("1. Extracting Faces in a Photo", 
               "I explored face detection models like <a href='https://github.com/ipazc/mtcnn' target='_blank'>MTCNN</a> and implemented this model to extract faces from a photo. These extracted faces serve as input for the subsequent phases.")

col1, col2 = st.columns(2)
with col1:
    st.image("DM Class.jpg", caption="Regular Image", use_container_width=True)
with col2:
    st.image("extracted_faces.jpg", caption="Extracted Faces", use_container_width=True)

display_section("2. Data Augmentation", 
               "Since I had only one image per student, I used <a href='https://github.com/genforce/idinvert' target='_blank'>in-domain GAN</a> to generate variations in poses and expressions, including adding sunglasses. Additionally, I applied standard augmentation techniques like rotation, blurring, color jittering, contrast adjustment, grayscale conversion, and noise addition.")

col1, col2, col3 = st.columns(3)
with col1:
    st.image("indomain_realimages.jpg", use_container_width=True)
with col2:
    st.image("indoman_generatedimages.jpg", use_container_width=True)
with col3:
    st.image("standard_augmentation_techniques.jpg", use_container_width=True)

display_section("3. Enhancing Image Resolution", 
               "To improve image quality, I leveraged <a href='https://github.com/TencentARC/GFPGAN' target='_blank'>GFPGAN</a> to enhance resolution, resulting in clearer and more detailed images for better recognition accuracy.")

col1, col2 = st.columns(2)
with col1:
    st.image("GFPGAN_Example.jpg", caption="Original Low-Resolution Image", use_container_width=True)
with col2:
    st.image("GFPGAN_ME.jpg", caption="Enhanced High-Resolution Image", use_container_width=True)
    

display_section("4. Face Recognition Using DeepFace", 
               "I utilized <a href='https://github.com/serengil/deepface' target='_blank'>DeepFace</a> with pre-trained models like Facenet512 and VGG-Face for feature extraction and facial identification, ensuring high accuracy.")

display_section("5. Storing Detected Faces in Attendance.xlsx", 
               "Recognized faces are stored in an Excel sheet named Attendance.xlsx, with the sheet name corresponding to the date the photo was taken.")

display_section("6. Real-Time Face Recognition", 
               "Using <a href='https://github.com/davisking/dlib' target='_blank'>dlib</a> and the <a href='https://github.com/ageitgey/face_recognition' target='_blank'>face_recognition</a> library, I implemented real-time face recognition, enabling live video stream processing and instant identification.")

display_section("7. Streamlit App Integration", 
               "All functionalities are seamlessly integrated into an interactive Streamlit app, providing a user-friendly interface for facial recognition tasks.")

display_section("8. Optimizing GPU Usage", 
               "To maximize performance, I configured CUDA and cuDNN on my local machine, leveraging GPU acceleration for deep learning models.")
col1, col2 = st.columns(2)
with col1:
    st.image("cuda.jpg", use_container_width=True)
with col2:
    st.image("cuDNN.png", use_container_width=True)

