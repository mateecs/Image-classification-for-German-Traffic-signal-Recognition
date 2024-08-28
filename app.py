import streamlit as st
from PIL import Image
#import tensorflow as tf
import numpy as np
import zipfile
import os
from model import loadmodel, preprocess_image, classes, entropy, Accuracy_Test
import matplotlib.pyplot as plt

model = loadmodel('model/model.h5')
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_id = np.argmax(prediction, axis=1)[0]  # Get the class ID
    return prediction[0], class_id

def count_images(directory):
    image_extensions = ['.jpg', '.jpeg', '.png']
    count = 0
    for file_name in os.listdir(directory):
        if os.path.splitext(file_name)[1].lower() in image_extensions:
            count += 1
    return count

def Labels(data_dir):
    labels = []
    data_dir = os.path.join(data_dir, "Train")
    for class_id in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_id)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                    labels.append(class_id)
    return np.array(labels)

def process_zip(file):

    with zipfile.ZipFile(file, 'r') as zip_ref:
        # Create temporary directories
        temp_dir = 'temp_zip_dir'
        os.makedirs(temp_dir, exist_ok=True)
        zip_ref.extractall(temp_dir)

        # Separate train and test directories
        train_dir = os.path.join(temp_dir, 'train')
        test_dir = os.path.join(temp_dir, 'test')

        # Count images
        #train_count = count_images(train_dir)
        labels = []
        train_count = 0
        for class_id in os.listdir(train_dir):
            class_dir = os.path.join(train_dir, class_id)
            if os.path.isdir(class_dir):
                train_count += count_images(class_dir)
                labels.append(int(count_images(class_dir)))
        test_count = count_images(test_dir)
        return train_count, test_count, labels

with st.sidebar:

    st.title("Choose Action")
    choice = st.radio("Navigation", ["Predict","Dataset Analysis", "Model Score"])
    st.info("This project application helps you build and explore your data.")

if choice == "Predict":
    st.title("Image Classification App")
    st.write("Upload an image and get the class ID predicted by the model.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        prediction, class_id = predict(image)
        st.write(f"Predicted Class ID: {class_id}")
        #st.write(f"Predicted Class ID: {classes[class_id]}")

        # Plotting the confidence plot
        st.write("Confidence Plot:")
        fig, ax = plt.subplots()
        ax.bar(range(len(classes)), prediction, color='blue')
        ax.set_xlabel('Class ID')
        ax.set_ylabel('Confidence')
        ax.set_title('Confidence Scores for Each Class')
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([classes[i] for i in range(len(classes))], rotation=45)
        st.pyplot(fig)



