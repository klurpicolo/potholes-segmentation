# import streamlit as st
# import cv2
# import numpy as np
# from ultralytics import YOLO

# # Cache the model to avoid reloading on every run
# @st.cache_resource
# def load_model():
#     return YOLO("yolov8s-seg.pt")

# model = load_model()

# st.title("YOLOv8 Segmentation App")

# # Upload image
# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Convert uploaded file to a NumPy array for OpenCV
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # BGR image

#     # Convert to RGB for display
#     original_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
#     # Display original image immediately (optional)
#     st.image(original_rgb, caption="Original Image", use_container_width=True)

#     if st.button("Detect Objects"):
#         # Run inference with YOLOv8 segmentation model
#         results = model.predict(source=img_cv)
#         res = results[0]
        
#         # Get annotated image (res.plot() returns a BGR image)
#         annotated_img = res.plot()
#         annotated_rgb = annotated_img[..., ::-1]  # convert BGR to RGB

#         st.image(annotated_rgb, caption="YOLOv8 Segmentation", use_container_width=True)

#######################################################################################################################################

# import streamlit as st
# import cv2
# import numpy as np
# import tempfile
# import os
# from ultralytics import YOLO

# # Cache the model to avoid reloading on every run
# @st.cache_resource
# def load_model():
#     return YOLO("yolov8s-seg.pt")

# model = load_model()

# st.title("YOLOv8 Segmentation App")

# # Allow user to upload image or video
# uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mkv"])

# if uploaded_file is not None:
#     filename = uploaded_file.name.lower()
    
#     # If the file is an image
#     if filename.endswith(("jpg", "jpeg", "png")):
#         # Convert uploaded file to a NumPy array for OpenCV
#         file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#         img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # BGR image
        
#         # Convert to RGB for display
#         original_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
#         st.image(original_rgb, caption="Original Image", use_container_width=True)
        
#         if st.button("Detect Objects on Image"):
#             results = model.predict(source=img_cv)
#             res = results[0]
            
#             # Get annotated image (res.plot() returns a BGR image)
#             annotated_img = res.plot()
#             annotated_rgb = annotated_img[..., ::-1]  # convert BGR to RGB
            
#             # Display images side by side using Streamlit columns
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.image(original_rgb, caption="Original Image", use_container_width=True)
#             with col2:
#                 st.image(annotated_rgb, caption="YOLOv8 Segmentation", use_container_width=True)
    
#     # If the file is a video
#     elif filename.endswith(("mp4", "mov", "avi", "mkv")):
#         # Save the uploaded video to a temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
#             tmp_file.write(uploaded_file.read())
#             tmp_video_path = tmp_file.name
        
#         st.video(tmp_video_path, format="video/mp4", start_time=0)
        
#         if st.button("Detect Objects on Video"):
#             # Run inference on the video file
#             results = model.predict(source=tmp_video_path)
#             res = results[0]
#             # YOLOv8's Results object includes a `video_path` attribute when processing video.
#             annotated_video_path = getattr(res, "video_path", None)
#             if annotated_video_path and os.path.exists(annotated_video_path):
#                 st.video(annotated_video_path, format="video/mp4", start_time=0)
#             else:
#                 st.error("No annotated video output available.")

#######################################################################################################################################

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO

# Cache the model to avoid reloading on every run
@st.cache_resource
def load_model():
    return YOLO("yolo11n-seg.pt")

model = load_model()

st.title("YOLOv8 Segmentation App")

# Allow user to upload image or video
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    filename = uploaded_file.name.lower()
    
    # If the file is an image
    if filename.endswith(("jpg", "jpeg", "png")):
        # Convert uploaded file to a NumPy array for OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # BGR image
        
        # Convert to RGB for display
        original_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        st.image(original_rgb, caption="Original Image", use_container_width=True)
        
        if st.button("Detect Objects on Image"):
            results = model.predict(source=img_cv)
            res = results[0]
            
            # Get annotated image (res.plot() returns a BGR image)
            annotated_img = res.plot()
            annotated_rgb = annotated_img[..., ::-1]  # convert BGR to RGB
            
            # Display images side by side using Streamlit columns
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_rgb, caption="Original Image", use_container_width=True)
            with col2:
                st.image(annotated_rgb, caption="YOLOv8 Segmentation", use_container_width=True)
    
    # If the file is a video
    elif filename.endswith(("mp4", "mov", "avi", "mkv")):
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_video_path = tmp_file.name
        
        st.video(tmp_video_path, format="video/mp4", start_time=0)
        
        if st.button("Detect Objects on Video"):
            results = model.track(source=tmp_video_path)
            # Assuming your model saves the annotated video and you know the path:
            annotated_video_path = results[0].video  # or use the appropriate attribute
            st.video(annotated_video_path)