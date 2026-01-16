import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

IMAGE_SIZE = 224


def load_class_names(json_path: str):
    """Load label -> flower name mapping from JSON."""
    with open(json_path, "r") as f:
        class_names = json.load(f)
    return class_names


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Resize and normalize an image for the model.

    Input:
      image: NumPy array (H, W, 3) with dtype uint8 or similar

    Returns:
      NumPy array (224, 224, 3) float32 in [0, 1]
    """
    image_tf = tf.convert_to_tensor(image)
    image_tf = tf.image.convert_image_dtype(image_tf, tf.float32)  # -> [0,1]
    image_tf = tf.image.resize(image_tf, (IMAGE_SIZE, IMAGE_SIZE))
    return image_tf.numpy()


def load_model(model_path: str) -> tf.keras.Model:
    """
    Load a saved Keras model that may include a TF Hub KerasLayer.
    """
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"KerasLayer": hub.KerasLayer}
    )
    return model


def predict(image_path: str, model: tf.keras.Model, top_k: int = 5):
    """
    Predict top_k classes for an image.

    Returns:
      probs: 1D NumPy array of probabilities (descending)
      classes: list of class ids as strings
    """
    im = Image.open(image_path).convert("RGB")
    image = np.asarray(im)

    processed = process_image(image)              # (224,224,3)
    processed = np.expand_dims(processed, axis=0) # (1,224,224,3)

    preds = model.predict(processed, verbose=0)[0]  # (num_classes,)
    topk = tf.nn.top_k(preds, k=top_k)

    probs = topk.values.numpy()
    classes = [str(int(i)) for i in topk.indices.numpy()]
    return probs, classes

