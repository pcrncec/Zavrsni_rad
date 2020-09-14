from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.models import load_model
from skimage.color import rgb2lab, lab2rgb
import numpy as np


def predict_rgb_image(image_path, model_path):
    model = load_model(model_path)
    loaded_img = load_img(image_path, target_size=(256, 256))
    img_array = img_to_array(loaded_img)
    img_array = 1.0/255 * img_array
    lab_img = rgb2lab(img_array)
    lab_img_bw = lab_img[:, :, 0]
    lab_img_bw = np.expand_dims(lab_img_bw, axis=0)
    colorized_img = model.predict(lab_img_bw)
    colorized_img = (128 * colorized_img)[0]
    merged_lab_img = np.zeros((256, 256, 3))
    merged_lab_img[:, :, 0] = lab_img_bw
    merged_lab_img[:, :, 1:] = colorized_img
    rgb_img = lab2rgb(merged_lab_img)
    return array_to_img(rgb_img)

