# train.py
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import image_dataset_from_directory
from efficientnet.tfkeras import EfficientNetB0
import os

# --- Configuration ---
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 1 # Binary classification (real=0, fake=1)
DATA_DIR = 'datasets/faces/'
MODEL_SAVE_PATH = 'models/deepfake_detector_model.keras'

# --- 1. Build the Model ---
def build_model():
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = False  # Freeze base layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

print("--- Building Model ---")
model = build_model()
model.summary()


# --- 2. Create Data Generators ---
print("\n--- Loading Datasets ---")
train_dataset = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary' # Important for binary_crossentropy
)

validation_dataset = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# Found class names will be ['fake', 'real']. Keras maps 'fake' to 1 and 'real' to 0.
print("Class Names:", train_dataset.class_names)


# --- 3. Compile and Train the Model ---
print("\n--- Compiling Model ---")
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n--- Starting Training ---")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset
)


# --- 4. Evaluate and Save ---
print("\n--- Evaluating Model ---")
# Create a separate test set from the validation data for a final check
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 2)
validation_dataset = validation_dataset.skip(val_batches // 2)

loss, accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\n--- Saving Model ---")
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"✅ Model saved successfully to {MODEL_SAVE_PATH}")