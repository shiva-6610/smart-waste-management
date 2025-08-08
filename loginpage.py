import streamlit as st
import pandas as pd
import numpy as np
import platform
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import hashlib
from roboflow import Roboflow
import supervision as sv
import cv2
import matplotlib.pyplot as plt 
import os
from PIL import Image
import io
from supervision.annotators.utils import Position




def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_password(password, hashed):
    return make_hashes(password) == hashed

user_db = {
    "sathwika": make_hashes("sathwika12"),
}

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""


def login():
    st.title("Login Page")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.button("Login")

    if login_btn:
        if username in user_db and check_password(password, user_db[username]):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success("Login successful! ✅")
            st.rerun()

        else:
            st.error("Incorrect username or password ❌")

def logout():
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.success("Logged out successfully.")
    st.rerun()


if not st.session_state.authenticated:
    login()
    st.stop()
else:
    st.set_page_config(layout="wide")
    st.sidebar.success(f"Logged in as: {st.session_state.username}")
    if st.sidebar.button("Logout"):
        logout()

#========================title===========================
    st.title("Trash Classifier")

#===================model calling ===================================

    rf = Roboflow(api_key = "s92952864nuTEq6I57GW")

    Project = rf.workspace().project("smart-waste-management-h5yif-mwcpw")
    model = Project.version(1).model



#=================upload img ====================
    st.header("Upload Image")
    file = st.file_uploader("Upload image jpg", type=["jpg"])
    if file is not None:
        # Save file temporarily
        with open("uploaded.jpg", "wb") as f:
            f.write(file.read())

        # Show uploaded image
        image = Image.open("uploaded.jpg")
        
        #st.image(image, caption="Uploaded Image", use_column_width=True)
        st.image(image, use_container_width=True)


#===========================prediction===============
        # Predict using Roboflow

        st.header("Predict Class")
        result = model.predict("uploaded.jpg", confidence=40, overlap=30).json()
        predictions = result["predictions"]

        # Display unique detected classes
        unique_classes = set(pred['class'] for pred in predictions)
        st.subheader("Detected Classes:")
        for cls in unique_classes:
            st.write(f"- {cls}")

        # Prepare detection data
        xyxy = []
        confidences = []
        labels = []
        class_id = []

        for pred in predictions:
            x1 = int(pred["x"] - pred["width"] / 2)
            y1 = int(pred["y"] - pred["height"] / 2)
            x2 = int(pred["x"] + pred["width"] / 2)
            y2 = int(pred["y"] + pred["height"] / 2)



            xyxy.append([x1, y1, x2, y2])
            confidences.append(pred["confidence"])
            class_id.append(pred.get("class_id", 0))  # ✅ Corrected
            #labels.append(pred["class"])
            labels = [f"{pred['class']} ({pred['confidence']*100:.1f}%)" for pred in predictions]


        detections = sv.Detections(
            xyxy=np.array(xyxy),
            confidence=np.array(confidences),
            class_id=np.array(class_id),
        )

        # Load and process image for annotation
        img = cv2.imread("uploaded.jpg")
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Annotate with bounding boxes and labels
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator(text_position= Position.BOTTOM_LEFT)
        annotated_image = box_annotator.annotate(scene=image_rgb.copy(), detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)


        # Display final image in Streamlit
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(annotated_image)
        ax.axis("off")
        ax.set_title("Annotated Image")
        st.pyplot(fig)

        # Save annotated output
        cv2.imwrite("output.jpg", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        st.success("Saved annotated image as output.jpg ✅")


    # streamlit run robo.py