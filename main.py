import os

import cv2
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from model import loadmodel, predict, calcuate_entropy, load_data, accuracy_score, Accuracy_Test
from app import process_zip, Labels
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def Graph(labels):
    # Get unique labels and their counts
    labels = np.sort(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Sort the unique labels and their corresponding counts
    sorted_indices = np.argsort(unique_labels)
    sorted_labels = unique_labels[sorted_indices]
    sorted_counts = counts[sorted_indices]

    # Plot the bar chart using Matplotlib
    plt.figure(figsize=(8, 6))
    plt.bar(sorted_labels, sorted_counts, color='skyblue')
    plt.xlabel('Class Labels')
    plt.ylabel('Frequency')
    plt.title('Frequency of Class Labels')
    plt.xticks(np.arange(0, 43, step=1))  # Ensure x-axis labels from 0 to 42
    plt.show()

def confidence_graph(confidence ,image_name):
    confidence_values = np.array(confidence)  # Replace with your (1, 43) array
    print(confidence_values.shape)
    confidence_values = confidence_values.flatten()
    print(confidence_values.shape)
    num_classes = confidence_values.shape[0]

    # Generate class labels (assuming classes are 0 to 42)
    class_labels = [str(i) for i in range(num_classes)]

    # Verify that the lengths match
    if len(class_labels) != len(confidence_values):
        raise ValueError("The number of class labels does not match the number of confidence values.")

    plt.figure(figsize=(12, 6))
    plt.bar(class_labels, confidence_values, color='skyblue')
    plt.xlabel('Class Labels')
    plt.ylabel('Confidence Values')
    plt.title('Confidence Values for Each Class')
    plt.xticks(rotation=90)  # Rotate x-axis labels if needed
    plt.show()


    # Save the plot to the specified directory
    """""""""""
    plot_path = os.path.join(save_dir, f'{image_name}_confidence_plot.png')
    plt.savefig(plot_path)
    plt.close()
    """""


def image_Predict(image_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        # List all image files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        # Read the image
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        # Predict
        probab,entropy = predict(image)
        # Save the confidence graph
        #confidence_graph(probab, os.path.splitext(image_file)[0], save_dir)



def Predict_entropy():
    image = cv2.imread('gtsr/Test/00000.png')
    probab, entropy,class_id = predict(image)
    confidence_graph(probab, "00001")

    print(probab)
    print(class_id)
    print(entropy)


def main():
    #Model= loadmodel('model/model.h5')
    #train_count, test_count, labels = process_zip("gtsr")
    #image_Predict('gtsr/Test','confidence_plots')
    #Predict_entropy()
    #load_data()
    Accuracy_Test('gtsr')



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
