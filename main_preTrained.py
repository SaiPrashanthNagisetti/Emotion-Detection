import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ------------------------------
# Step 1: Load Pre-trained ResNet50 without Top Layers
# ------------------------------
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Freeze the base layers to avoid training them initially
for layer in base_model.layers:
    layer.trainable = False

# ------------------------------
# Step 2: Add Custom Layers for Emotion Detection
# ------------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
output = Dense(7, activation='softmax')(x)  # 7 emotion categories

# Create the final model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ------------------------------
# Step 3: Prepare Data Generators
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
    train_dir, target_size=(48, 48), batch_size=64, class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(48, 48), batch_size=64, class_mode='categorical'
)

# ------------------------------
# Step 4: Implement Early Stopping
# ------------------------------
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ------------------------------
# Step 5: Train the Model
# ------------------------------
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[early_stopping]
)

# ------------------------------
# Step 6: Visualize Training Performance
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
# Step 7: Evaluate the Model
# ------------------------------
val_loss, val_accuracy = model.evaluate(val_generator)
print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# ------------------------------
# Step 8: Save the Trained Model
# ------------------------------
model.save('resnet_emotion_detection_model.h5')

print("Model saved as 'resnet_emotion_detection_model.h5'.")
