import zipfile
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Path to the uploaded .zip file
zip_path = 'C:/Users/user/Desktop/DB4_B.zip'
extract_path = 'C:/Users/user/Desktop/DB4_B'

# Function to extract zip file
def extract_zip(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    return os.listdir(extract_path)  # List files to confirm extraction

# Extract the zip file
extracted_files = extract_zip(zip_path, extract_path)

# Function to load and preprocess images
def load_and_preprocess_images(file_paths, extract_path, target_size=(224, 224)):
    images = []
    labels = []  # To store labels based on file naming convention

    for file_path in file_paths:
        # Open the image file
        with Image.open(f"{extract_path}/{file_path}") as img:
            # Ensure image is in RGB mode
            img = img.convert('RGB')
            # Resize image to match ResNet50 model input
            img = img.resize(target_size)
            # Convert image to numpy array, normalize it, and add a channel dimension
            img_array = np.array(img)
            img_array = preprocess_input(img_array)
            images.append(img_array)

            # Extract label from the filename, assuming format 'ID_X.tif'
            label = int(file_path.split('_')[0])
            labels.append(label)

    return np.array(images), np.array(labels)

# List of file paths for images, excluding any directories
image_files = [f for f in extracted_files if f.endswith('.tif')]
images, labels = load_and_preprocess_images(image_files, extract_path)

# Check the shapes of the arrays
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

# Load the pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Adding custom Layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(np.unique(labels)), activation='softmax')(x)

# Final model setup
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers which you don't want to train
for layer in base_model.layers:
    layer.trainable = False

# Compiling the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Create a label encoder object
label_encoder = LabelEncoder()

# Fit the label encoder to your labels and transform them to normalized labels
labels_encoded = label_encoder.fit_transform(labels)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)
print("Transformed label examples:", labels_encoded[:10])

# Confirm the shapes of training and validation data
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)

# Fit the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on the test set
eval_result = model.evaluate(X_val, y_val)
print(f"Test Loss: {eval_result[0]}, Test Accuracy: {eval_result[1]}")

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model dəqiqliyi')
plt.ylabel('Dəqiqlik')
plt.xlabel('Dövr')
plt.legend(['Təlim', 'Doğrulama'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model itkisi')
plt.ylabel('İtki')
plt.xlabel('Dövr')
plt.legend(['Təlim', 'Doğrulama'], loc='upper left')
plt.show()
