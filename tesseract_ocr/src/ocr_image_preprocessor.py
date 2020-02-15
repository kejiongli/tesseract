import numpy as np
import cv2

MORPH_KERNEL = 2
MAX_X, MAX_Y = 4150, 4150
MIN_DM = 300

def _validate_image(image: np.ndarray, min_dm: int = MIN_DM) -> None:
    """
    Validate if image format is correct. OpenCV doesn't through an error if an empty or non-existent file is loaded
    Additionaly, the size of the image is also checked to make sure the image is not too small
    :param image: np.ndarray
    :param min_dm: minimum dimesion of image
    :return: None
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("Image file has incorrect format or does not exist")

    if min(image.shape[0], image.shape[1]) < min_dm:
        raise ValueError(
            f"Image is too small to be used (Image is of dimension {image.shape[0]} x {image.shape[1]}. Minimum allowed dimension size is {min_dm}"
        )


def sharpen_image(
        image_dir: str,
        image_dest: str,
        morph_kernel: int = MORPH_KERNEL,
        max_x: int = MAX_X,
        max_y: int = MAX_Y,
        min_dm: int = MIN_DM
) -> None:

    image = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)

    _validate_image(image, min_dm)

    image_sz_y, image_sz_x = image.shape[0], image.shape[1]
    resize_mult = min(max_x/image_sz_x, max_y/image_sz_y)

    # Taking a matrix of size MORPH_KERNEL = 2 as the kernel
    opening_kernel = np.ones((morph_kernel, morph_kernel), np.uint8)

    # upscale
    image = cv2.resize(image, None, fx=resize_mult, fy=resize_mult, interpolation=cv2.INTER_AREA)

    # opening
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, opening_kernel)

    # # noise reduce
    # image = cv2.medianBlur(image, 3)
    #
    # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    cv2.imwrite(image_dest, image)