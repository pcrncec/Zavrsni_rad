import os
import sys
from shutil import move
from random import sample
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2lab


def generator(batch_size, images_path):
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        images_path,
        class_mode=None,
        batch_size=batch_size
    )
    for batch in train_generator:
        lab_batch = rgb2lab(batch)
        x_batch = lab_batch[:, :, :, 0]
        y_batch = lab_batch[:, :, :, 1:] / 128
        yield x_batch, y_batch


def get_number_of_data(path):
    try:
        n = len(os.listdir(path))
    except FileNotFoundError:
        print('Ne postoji direktorij:', path)
        sys.exit(1)
    if n == 0:
        raise Exception('Greška, nedostaju podaci!')
    return n


def split_data(source_images_dir, validation_images_dir, split_ratio):
    if os.path.exists(validation_images_dir):
        val_images = os.listdir(validation_images_dir)
        if len(val_images) > 0:
            for image in val_images:
                image_path = os.path.join(validation_images_dir, image)
                move(image_path, source_images_dir)
    else:
        os.mkdir(validation_images_dir)
    all_images = os.listdir(source_images_dir)
    val_images = sample(all_images, int(split_ratio*len(all_images)))
    for image in val_images:
        source_image_path = os.path.join(source_images_dir, image)
        if os.path.getsize(source_image_path) > 0:
            val_image_path = os.path.join(validation_images_dir, image)
            move(source_image_path, val_image_path)


