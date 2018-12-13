import os
import numpy as np
import cv2

MAX_LABEL = 609 # index from 0

IMAGE_SHAPE = 200, 200, 3


def read_image_directory_to_str(directory):
    """Read VeRi image directory (train, gallery, query).

    Parameters
    ----------
    directory : str
        Path to image directory.

    Returns
    -------
    (List[str], List[int], List[int])
        Returns a tuple of image filenames, corresponding unique IDs for the
        individuals in the images, and camera identifiers.

    """
    images, ids = [], []
    for filename in sorted(os.listdir(directory)):
        filename_base, ext = os.path.splitext(filename)
        if '.' in filename_base:
            # Some images have double filename extensions.
            filename_base, ext = os.path.splitext(filename_base)
        if ext != ".jpg":
            continue  # Not an image.
        vehicle_id, _ = filename_base.split('_')

        images.append(os.path.join(directory, filename))
        ids.append(int(vehicle_id))
    return images, ids

def read_image_directory_to_image(directory):
    """Read images from VeRi image directory.

    Parameters
    ----------
    directory : str
        Path to image directory (e.g., 'resources/VeRi/image_train')

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray)
       Returns a tuple of images, associated IDs for the individuals in the
       images, and camera indices.

    """

    filenames, ids = (read_image_directory_to_str(directory))

    images = np.zeros((len(filenames), ) + IMAGE_SHAPE, np.uint8)
    for i, filename in enumerate(filenames):
        if i % 1000 == 0:
            print("Reading %s, %d / %d" % (directory, i, len(filenames)))
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        images[i] = cv2.resize(image, IMAGE_SHAPE[:2][::-1])
    ids = np.asarray(ids, dtype=np.int64)
    return images, ids

def read_train_split_to_str(dataset_dir):
    """Read training data to list of filenames.

    Parameters
    ----------
    dataset_dir : str
        Path to VeRi dataset directory; 'image_train' should be a
        subdirectory of this folder.

    Returns
    -------
    (List[str], List[int], List[int])
        Returns a tuple of image filenames, unique vehicle ids, and
        camera indices.

    """
    video_directory = os.path.join(dataset_dir,"")
    return read_image_directory_to_str(video_directory)


def read_train_split_to_image(dataset_dir):
    """Read training data into memory.

    Parameters
    ----------
    dataset_dir : str
        Path to VeRi dataset directory; 'image_train' should be a
        subdirectory of this folder.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray)
        Returns a tuple of images, unique vehicle ids, and camera indices.

    """
    video_directory = os.path.join(dataset_dir,"")
    return read_image_directory_to_image(video_directory)


def read_test_split_to_str(dataset_dir):
    """Read gallery data to list of filenames.

    Parameters
    ----------
    dataset_dir : str
        Path to VeRi dataset directory; 'image_test' should be a
        subdirectory of this folder.

    Returns
    -------
    (List[str], List[int], List[int])
        Returns a tuple of image filenames, unique vehicle Ids, and
        camera indices.

    """
    video_directory = os.path.join(dataset_dir, "everything")
    return read_image_directory_to_str(video_directory)


def read_test_split_to_image(dataset_dir):
    """Read test data into memory.

    Parameters
    ----------
    dataset_dir : str
        Path to VeRi dataset directory; 'image_test' should be a
        subdirectory of this folder.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray)
        Returns a tuple of images, unique vehicle ids, and camera indices.

    """
    video_directory = os.path.join(dataset_dir, "everything")
    return read_image_directory_to_image(video_directory)