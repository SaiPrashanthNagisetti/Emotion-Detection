import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ------------------------------
# Step 1: Build the CNN Model
# ------------------------------
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

    # Additional dense layer for better learning capacity
    Dense(512, activation='relu'),
    Dropout(0.5),

    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(7, activation='softmax')  # 7 emotion categories
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ------------------------------
# Step 2: Prepare Data Generators
# ------------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_dir = 'C:\\Users\\nsaip\\Documents\\CNN_Projects\\Emotion_Detection\\data\\train'  # Replace with your training data path
val_dir = 'C:\\Users\\nsaip\\Documents\\CNN_Projects\\Emotion_Detection\\data\\test'  # Replace with your validation data path

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(48, 48), color_mode='grayscale', batch_size=64, class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(48, 48), color_mode='grayscale', batch_size=64, class_mode='categorical'
)

# ------------------------------
# Step 3: Implement Early Stopping
# ------------------------------
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ------------------------------
# Step 4: Train the Model
# ------------------------------
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[early_stopping]
)

# ------------------------------
# Step 5: Visualize Accuracy and Loss
# ------------------------------
def plot_learning_curves(history):
    plt.figure(figsize=(12, 4))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training vs Validation Accuracy')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')

    plt.show()

plot_learning_curves(history)

# ------------------------------
# Step 6: Evaluate the Model
# ------------------------------
val_loss, val_accuracy = model.evaluate(val_generator)
print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# ------------------------------
# Step 7: Save the Trained Model
# ------------------------------
model.save('emotion_detection_model.h5')

print("Model saved as 'emotion_detection_model.h5'.")
