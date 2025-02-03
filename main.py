import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # Regularization
    Dense(7, activation='softmax')  # Assuming 7 emotion categories
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


train_datagen = ImageDataGenerator(rescale=1.0/255.0, 
                                   rotation_range=30, 
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Directory paths to your dataset (train and validation)
train_dir = 'C:\\Users\\nsaip\\Documents\\CNN_Projects\\Emotion_Detection\\data\\train'
val_dir = 'C:\\Users\\nsaip\\Documents\\CNN_Projects\\Emotion_Detection\\data\\test'

# Load images from directories
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(48, 48),
                                                    color_mode='grayscale',
                                                    batch_size=64,
                                                    class_mode='categorical')

val_generator = test_datagen.flow_from_directory(val_dir,
                                                 target_size=(48, 48),
                                                 color_mode='grayscale',
                                                 batch_size=64,
                                                 class_mode='categorical')

history = model.fit(train_generator,
                    epochs=81,
                    validation_data=val_generator)


# Evaluate the model on validation/test data
val_loss, val_accuracy = model.evaluate(val_generator)
print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')


model.save('emotion_detection_model.h5')
