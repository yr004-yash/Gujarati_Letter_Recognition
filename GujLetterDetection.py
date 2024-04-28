import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from google.colab import drive
drive.mount('/content/drive')

# Define dataset directory
dataset_dir = "/content/drive/MyDrive/ml_dataset/DatasetNew"

#/-------------------------------------------------Data preprocessing and augmentation------------------------------------------------/
img_width, img_height = 32, 32
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    validation_split=0.2  # Split data into training and validation sets
)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='grayscale',  # Set color mode to grayscale
    class_mode='categorical',
    subset='training'  # Specify training data
)


validation_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='grayscale',  # Set color mode to grayscale
    class_mode='categorical',
    subset='validation'  # Specify validation data
)
print(train_generator)
#/------------------------------------------------------------------------------------------------------------------------------------/



#/--------------------------------------------- Define CNN model architecture------------------------------------------------------------/
model = Sequential()
model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(img_width, img_height, 1)))  # Adjusted input shape for grayscale images
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(train_generator.num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")

# Save the trained model
model.save("/content/drive/MyDrive/ml_dataset/guj_letter_2.h5")
#/----------------------------------------------------------------------------------------------------------------------------------------/



#/-------------------------Prediction of Letter-------------------------------------------------------------------------------------------/
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import matplotlib.pyplot as plt

# Define image dimensions
img_width, img_height = 32, 32  # Adjust as needed

# Load the example image
example_path = "/content/drive/MyDrive/ml_dataset/sample_testing set/image_25042021_201628.jpg"
example_image = load_img(example_path, target_size=(img_width, img_height), color_mode="grayscale")

# Convert the image to a numpy array
example_image_array = img_to_array(example_image)

# Normalize the pixel values
example_image_array = example_image_array.astype('float32') / 255

# Reshape the image to match the input shape expected by the model
example_image_array = np.expand_dims(example_image_array, axis=0)  # Add batch dimension

# Assuming the model is trained and loaded correctly
prediction = model.predict(example_image_array)
print(prediction)

print("max probability index=",np.argmax(prediction))

# Display the image
plt.imshow(np.squeeze (example_image_array), cmap="gray")
plt.show()

# Dictionary mapping indices to characters
dictionary = {0: 'અ', 1: 'આ', 2: 'ઇ', 3: 'ઈ', 4: 'ઉ', 5: 'ઊ', 6: 'એ', 7: 'ઐ',
              8: 'ઓ', 9: 'ઔ', 10: 'ક', 11: 'ખ', 12: 'ગ', 13: 'ઘ', 14: 'ચ', 15: 'છ',
              16: 'જ', 17: 'ઝ', 18: 'ટ', 19: 'ઠ', 20: 'ડ', 21: 'ઢ', 22: 'ણ', 23: 'ત',
              24: 'થ', 25: 'દ', 26: 'ધ', 27: 'ન', 28: 'પ', 29: 'ફ', 30: 'બ', 31: 'ભ',
              32: 'મ', 33: 'ય', 34: 'ર', 35: 'લ', 36: 'વ', 37: 'શ', 38: 'ષ', 39: 'સ',
              40: 'હ', 41: 'ળ', 42: 'ક્ષ', 43: 'જ્ઞ'}

# Display the predicted character
predicted_character = dictionary[np.argmax(prediction)]
print("Predicted character is:", predicted_character)
#/---------------------------------------------------------------------------------------------------------------------------------------/