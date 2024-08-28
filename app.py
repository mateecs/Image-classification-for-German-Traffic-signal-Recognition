import streamlit as st
from PIL import Image
#import tensorflow as tf
import numpy as np
import cv2
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

if choice == "Dataset Analysis":
    st.title("Dataset Analysis")
    st.write("Upload a Dataset in a specified Format.")
    uploaded_file = st.file_uploader("Choose a Dataset...", type=["rar", "zip"])

    if uploaded_file is not None:
        #train_count, test_count = process_zip(uploaded_file)
        #st.write(f"Number of images in the training set: {train_count}")
        #st.write(f"Number of images in the test set: {test_count}")
        entropies  = entropy(Accuracy_Test('gtsr'))
        st.write("Entropy Distribution")
        st.write(entropies.shape)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(entropies, bins=30, color='blue', edgecolor='black')
        ax.set_xlabel('Entropy')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Entropy')
        ax.grid(True)

        # Display the plot in Streamlit
        st.pyplot(fig)


if choice == "Model Score":
    st.title("Object Detection in Large Images")
    st.write("Upload a large image and the model will detect and highlight regions of interest.")

    uploaded_file = st.file_uploader("Choose a large image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file)
        image_array = np.array(image)

        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("Running object detection...")


        # Define sliding window and detection function (same as explained earlier)
        def sliding_window(image, step_size, window_size):
            for y in range(0, image.shape[0] - window_size[1], step_size):
                for x in range(0, image.shape[1] - window_size[0], step_size):
                    yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


        def detect_objects_in_image(model, image, window_size=(32, 32), step_size=16, threshold=0.9):
            detected_boxes = []
            for (x, y, window) in sliding_window(image, step_size, window_size):
                if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                    continue

                window = Image.fromarray(window)
                window_processed = preprocess_image(window)
                prediction, class_id = predict(window_processed)

                # Check if the prediction confidence is above the threshold
                if max(prediction) > threshold:
                    detected_boxes.append((x, y, x + window_size[0], y + window_size[1], class_id))

            return detected_boxes


        def draw_boxes(image, boxes):
            draw = ImageDraw.Draw(image)
            for (x1, y1, x2, y2, class_id) in boxes:
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, y1), f"Class {class_id}", fill="red")
            return image


        # Detect objects in the image
        detected_boxes = detect_objects_in_image(model, image_array, window_size=(32, 32), step_size=16, threshold=0.9)

        # Draw bounding boxes on the original image
        output_image = draw_boxes(image.copy(), detected_boxes)

        # Display the image with bounding boxes in Streamlit
        st.image(output_image, caption='Detected Regions', use_column_width=True)
