import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
from PIL import Image

INPUT_SHAPE = (64, 64, 3)
NUM_CLASSES = 3          

def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

GLOBAL_MODEL = None

def load_or_get_model():

    global GLOBAL_MODEL

    if GLOBAL_MODEL is None:
        GLOBAL_MODEL = tf.keras.models.load_model('cnn_image/models/cnn_rps.keras')
        
    return GLOBAL_MODEL

def preprocess_for_cnn(image_np: np.ndarray) -> np.ndarray:

    image_pil = Image.fromarray(image_np)
    image_resized = image_pil.resize(INPUT_SHAPE[:2])
    
    img_array = np.array(image_resized, dtype=np.float32) / 255.0

    return np.expand_dims(img_array, axis=0)

def classify_image(image_np: np.ndarray, classes: list) -> tuple[str, float]:
    cnn_model = load_or_get_model()
    
    processed_input = preprocess_for_cnn(image_np)
    
    predictions = cnn_model.predict(processed_input, verbose=0)[0]
    
    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index]
    predicted_class = classes[predicted_index]
    
    return predicted_class, float(confidence)