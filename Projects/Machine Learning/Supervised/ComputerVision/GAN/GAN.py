"""
SRGAN is a combination of generative adversarial networks (GANs)
and deep convolutional neural networks (CNNs) and it produces highly
realistic high-resolution images from low-resolution images. 
"""
import os
import time
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

# Environment variable to enable download progress for TensorFlow Hub models
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

# Constants for image paths and model URL
IMAGE_PATH = "C:\\temp\\data\\Te-glTr_0000.jpg"
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

def preprocess_image(image_path):
    """
    Load and preprocess an image to make it suitable for the model.
    
    Args:
        image_path (str): Path to the image file.

    Returns:
        tf.Tensor: Preprocessed image tensor.
    """
    hr_image = tf.image.decode_image(tf.io.read_file(image_path), channels=3)
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)

def save_image(image, filename):
    """
    Save an image tensor to a file.

    Args:
        image (tf.Tensor): 3D image tensor [height, width, channels].
        filename (str): File path where the image will be saved.
    """
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save(f"{filename}.jpg")
    print(f"Saved as {filename}.jpg")

def plot_image(image, title=""):
    """
    Plot an image using matplotlib.

    Args:
        image (tf.Tensor): 3D image tensor [height, width, channels].
        title (str): Title for the plot.
    """
    image = np.asarray(tf.clip_by_value(image, 0, 255))
    plt.imshow(Image.fromarray(tf.cast(image, tf.uint8).numpy()))
    plt.axis("off")
    plt.title(title)
    plt.show()

def downscale_image(image):
    """
    Downscale an image using bicubic downsampling.

    Args:
        image (tf.Tensor): 3D tensor of the preprocessed image.

    Returns:
        tf.Tensor: Downscaled image tensor.
    """
    if len(image.shape) != 3:
        raise ValueError("Function works only on single image (3D tensor).")

    lr_image = np.asarray(
        Image.fromarray(tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8).numpy())
        .resize((image.shape[1] // 4, image.shape[0] // 4), Image.BICUBIC)
    )
    lr_image = tf.expand_dims(lr_image, 0)
    lr_image = tf.cast(lr_image, tf.float32)
    return lr_image

# Load the pre-trained ESRGAN model
model = hub.load(SAVED_MODEL_PATH)

# Process the high-resolution image
hr_image = preprocess_image(IMAGE_PATH)
plot_image(tf.squeeze(hr_image), title="Original Image")
save_image(tf.squeeze(hr_image), filename="C:\\temp\\data\\OriginalImage")

# Downscale the image to create a low-resolution version
lr_image = downscale_image(tf.squeeze(hr_image))
plot_image(tf.squeeze(lr_image), title="Low Resolution")
save_image(tf.squeeze(lr_image), filename="C:\\temp\\data\\LowResolutionImage")

# Apply the super-resolution model
start_time = time.time()
super_res_image = model(lr_image)
super_res_image = tf.squeeze(super_res_image)
print(f"Time Taken: {time.time() - start_time:.2f} seconds")

# Plot and save the super-resolution image
plot_image(super_res_image, title="Super Resolution")
save_image(super_res_image, filename="C:\\temp\\data\\SuperResolution")