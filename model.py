import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import tensorflow as tf
import re
import os
import cv2
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from scipy.stats import entropy

data_dir = 'gtsr/'
Image_Width=32
Image_Height=32
Image_Width = 32
Image_Height = 32

classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Veh > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing veh > 3.5 tons' }

def loadmodel(path):
    model = tf.keras.models.load_model(path)
    return  model

def preprocess_image(image):
    """"""""""
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    image = image.resize((IMAGE_Width, IMAGE_Height))  # Resize the image to match the input size of the model
    image = np.array(image)  # Convert the image to a numpy array
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image 
    """""
    if isinstance(image, Image.Image):
        image = np.array(image)
    if image.shape[2] == 4:
        # If the image has an alpha channel, remove it
        image = image[:, :, :3]

        # Ensure Image_Width and Image_Height are defined
    if not (Image_Width > 0 and Image_Height > 0):
        raise ValueError("Image_Width and Image_Height must be positive integers.")

        # Print for debugging
    print(f"Image shape before resize: {image.shape}")

    # Resize image
    image = cv2.resize(image, (Image_Width, Image_Height))

    print(f"Image shape after resize: {image.shape}")

    # Normalize and add batch dimension
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def load_Train_data(data_dir):
    images = []
    labels = []
    for class_id in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_id)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                image = cv2.imread(img_path)
                if image is not None:
                    image = cv2.resize(image, (32, 32))  # Resize to 32x32
                    images.append(image)
                    labels.append(int(class_id))
    return np.array(images), np.array(labels)


def predict(image):
    model = loadmodel('latest model.h5')
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    entropy = calcuate_entropy(prediction)
    class_id = np.argmax(prediction, axis=1)[0]  # Get the class ID
    #return class_id
    return prediction, entropy, class_id

def model_arch():
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu',
                            input_shape=(Image_Height, Image_Width, 3)),
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.3),  # Dropout after first Conv block

        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.BatchNormalization(axis=-1),
        keras.layers.Dropout(0.3),  # Dropout after first Conv block

        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=0.5),

        keras.layers.Dense(43, activation='softmax')
    ])
    return model


def model_train(x_train, y_train ,lr , epoch):

    #Splitting data
    y_train = keras.utils.to_categorical(y_train, num_classes=43)
    X_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2,
                                                      random_state=42, shuffle=True)
    X_train = X_train / 255.
    X_val = x_val / 255.


    model = model_arch()
    lr = lr
    epochs = epoch
    opr = Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opr, metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_val, y_val))
    model.save("model_new1.h5")



def load_data():
    x_train, y_train = load_Train_data(data_dir + 'Train')
    x_train, y_train = Preprocessing(x_train, y_train)
    model = model_train(x_train, y_train, .0001, 19)


def Preprocessing(x_train, y_train):
    # Loading Train data
    x_train, y_train = load_Train_data(data_dir + 'Train')

    datagen = ImageDataGenerator(
        brightness_range=[0.7, 1.3],  # Adjust brightness
        rotation_range=20,  # Rotate images up to 20 degrees
        zoom_range=0.0,  # No zooming
        horizontal_flip=False
    )

    # Augment data for classes with fewer than 500 images
    min_images = 650
    target_images = 1200

    augmented_X_train = []
    augmented_y_train = []

    class_labels, class_counts = np.unique(y_train, return_counts=True)
    for class_label, count in zip(class_labels, class_counts):
        if count < min_images:
            class_indices = np.where(y_train == class_label)[0]
            class_images = x_train[class_indices]

            # Augment the images
            num_to_generate = target_images - count
            print(f"Generating {num_to_generate} images for class {class_label}")

            i = 0
            for x, y in datagen.flow(class_images, [class_label] * len(class_images), batch_size=1):
                augmented_X_train.append(x[0])
                augmented_y_train.append(y[0])
                i += 1
                if i >= num_to_generate:
                    break

    # Convert the augmented data into numpy arrays
    augmented_X_train = np.array(augmented_X_train)
    augmented_y_train = np.array(augmented_y_train)

    # Combine the original and augmented data
    X_train_augmented = np.concatenate((x_train, augmented_X_train), axis=0)
    y_train_augmented = np.concatenate((y_train, augmented_y_train), axis=0)

    return X_train_augmented, y_train_augmented


def calcuate_entropy(predictions):
    entropy = -np.sum(predictions * np.log(predictions + 1e-10), axis=1)
    return entropy


def Accuracy_Test(data_dir):
    model = tf.keras.models.load_model('model/latest model.h5')
    images = []
    filenames = []

    data_dir = os.path.join(data_dir, 'Test')

    # Read PNG images
    for filename in os.listdir(data_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(data_dir, filename)
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if image is not None:
                resized_image = cv2.resize(image, (32, 32))
                images.append(resized_image)
                filenames.append(filename)
            else:
                print(f"Warning: Could not read image {img_path}")

    def extract_number(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else 0

    # Sort filenames based on the extracted numeric part
    sorted_filenames = sorted(filenames, key=extract_number)

    labels_df = pd.read_csv('gtsr/labels.csv')
    sorted_labels = []

    for filename in sorted_filenames:
        lab, _ = os.path.splitext(filename)
        sorted_labels.append(labels_df['label'][int(lab)])

    x_test = np.array(images) / 255.0
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)

    # Calculate and print accuracy
    accuracy = accuracy_score(sorted_labels, predicted_classes)
    print(f'Accuracy: {accuracy}')
    entropy = -np.sum(predictions * np.log(predictions + 1e-7), axis=1)
    plt.figure(figsize=(10, 6))
    plt.hist(entropy, bins=30, color='blue', edgecolor='black')
    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    plt.title('Distribution of Entropy')
    plt.grid(True)
    plt.show()

    return predictions
