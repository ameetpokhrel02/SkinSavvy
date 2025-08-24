import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf
import io

# Load your trained model
# IMPORTANT: Replace 'path/to/your/acne_detection_model.h5' with your model's file path.
model = load_model('path/to/your/acne_detection_model.h5')

# Define the image size your model was trained on
IMG_SIZE = (128, 128)

def preprocess_image(image_bytes):
    """
    Takes raw image bytes, converts them to a PIL Image, resizes,
    and converts to a NumPy array suitable for model prediction.
    """
    try:
        # Open image from bytes
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        # Resize to the target size
        img = img.resize(IMG_SIZE)
        # Convert to a numpy array and normalize
        img_array = np.array(img) / 255.0
        # Expand dimensions to create a batch (1, height, width, channels)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None

def detect_acne(image_bytes):
    """
    Preprocesses an image and uses the model to predict if it contains acne.
    """
    preprocessed_img = preprocess_image(image_bytes)
    if preprocessed_img is None:
        return "invalid_image"

    # Make a prediction
    prediction = model.predict(preprocessed_img)
    
    # The output is a probability score. We need to classify it.
    # A simple threshold is used here (e.g., > 0.5)
    acne_probability = prediction[0][0]

    # Return a status and the probability
    if acne_probability > 0.5:
        return "acne_detected", acne_probability
    else:
        return "no_acne", acne_probability