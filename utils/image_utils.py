import cv2
import numpy as np
from sklearn.cluster import KMeans


# region Read

def read_image_with_alpha(file_path: str):
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        b, g, r = cv2.split(img)
        alpha = np.ones_like(b) * 255
        img = cv2.merge([b, g, r, alpha])
    return img


# region Display

def show_image(img: np.ndarray):
    cv2.imshow("Image", img)
    cv2.waitKey(0)


def show_mask(mask: np.ndarray):
    show_image(mask.astype(np.uint8) * 255)


def display_color_list_as_img(color_list: list):
    '''
    Creates a Image where each color is listed as a 10x10 pixel rectangle.
    '''

    # Anzahl der Farben
    num_colors = len(color_list)

    pixels_per_color = 15

    # Create a blank image
    img = np.zeros((pixels_per_color, pixels_per_color *
                   num_colors, 3), dtype=np.uint8)

    for i, color in enumerate(color_list):
        # Ignore alpha if present
        img[:, i * pixels_per_color:(i + 1) * pixels_per_color] = color[:3]

    cv2.imshow("Color List", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# region Mask


def get_visible_mask(img: np.ndarray) -> np.ndarray:
    alpha_channel = img[:, :, 3]
    visible_mask = alpha_channel == 0
    return visible_mask


def get_only_visible_pixels(img: np.ndarray) -> np.ndarray:
    visible_mask = get_visible_mask(img)
    visible_pixels = img[:, :, :3][visible_mask]
    return visible_pixels


def create_mask(img: np.ndarray) -> np.ndarray:
    # Graustufenkonvertierung
    # R, G, B nach Graustufen konvertieren
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)

    # Bessere Schwelle (automatisch mit Otsu)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Maske direkt verwenden (kein Filter)
    mask = thresh.copy()

    return mask


def apply_mask_for_image(mask: np.ndarray, img: np.ndarray, fill_with_green: bool = False) -> np.ndarray:
    # Alpha-Kanal = Maske
    b, g, r = cv2.split(img[:, :, :3])
    if fill_with_green:
        g = np.where(mask == 0, 0, 255)
        g = g.astype(np.uint8)
    rgba = cv2.merge([b, g, r, mask])
    return rgba


def create_outline_mask(input_mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        input_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline_mask = np.zeros_like(input_mask)
    if contours:
        for cnt in contours:
            cv2.drawContours(outline_mask, [cnt], 0, 255, -1)
    return outline_mask


def blur_mask(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    blurred_mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, smooth_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)
    return smooth_mask


# region Color

def get_color_list(img: np.ndarray, mask: np.ndarray = None, ignore_transparent: bool = True, return_counts: bool = False):
    """
    deprecated
    img: HxWx3 or HxWx4 (BGR[A])
    mask: optional HxW binary mask or HxWx3/4 image (nonzero -> include)
    ignore_transparent: if True and img has alpha channel, exclude alpha==0 pixels
    return_counts: if True, return list of (color_tuple, count) sorted by count desc
    """
    # build visibility boolean mask
    vis = np.ones(img.shape[:2], dtype=bool)
    if ignore_transparent and img.shape[2] == 4:
        vis &= (img[:, :, 3] > 0)

    if mask is not None:
        if mask.ndim == 3:
            # if mask has channels prefer alpha if present else convert to gray
            if mask.shape[2] == 4:
                m = mask[:, :, 3] > 0
            else:
                m = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY) > 0
        else:
            m = mask > 0
        vis &= m

    # select visible RGB pixels
    pixels = img[:, :, :3][vis]
    if pixels.size == 0:
        return [] if not return_counts else ([], [])

    # get unique colors and counts
    pixels_2d = pixels.reshape(-1, 3)
    unique_colors, counts = np.unique(pixels_2d, axis=0, return_counts=True)

    # sort by frequency desc
    order = np.argsort(-counts)
    unique_colors = unique_colors[order]
    counts = counts[order]

    # convert to tuples if useful
    color_tuples = [tuple(map(int, c)) for c in unique_colors]  # (B,G,R)

    if return_counts:
        return list(zip(color_tuples, counts.tolist()))
    return color_tuples


def reduce_colors_of_img_to_colors_of_color_list(img: np.ndarray, color_list: list) -> np.ndarray:
    '''
    Reduces the number of colors in the image to the colors in the color list.
    For each color, the nearest color in the color list shall be choosen.
    '''

    # Find the nearest color for each pixel
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = img[i, j]
            # Find the nearest color in the color list
            nearest_color = min(
                color_list, key=lambda c: np.linalg.norm(pixel - c))
            img[i, j] = nearest_color

    return img


def rgb_to_rgba(img: np.ndarray) -> np.ndarray:
    if img.shape[2] == 4:
        return img
    b, g, r = cv2.split(img)
    alpha = np.ones_like(b) * 255
    rgba = cv2.merge([b, g, r, alpha])
    return rgba


def find_main_colors_of_img(img: np.ndarray, num_colors: int = 6) -> list:
    """
    Find the main colors in the image using k-means clustering.
    """

    # Reshape the image to be a list of pixels
    pixels = img.reshape(-1, 4)
    print(img.shape)
    print(pixels.shape)

    # Remove transparent pixels
    pixels = pixels[pixels[:, 3] > 0] if img.shape[2] == 4 else pixels

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)

    # Get the cluster centers (main colors)
    main_colors = kmeans.cluster_centers_.astype(int)

    return [tuple(color) for color in main_colors]


def find_most_common_color(img: np.ndarray) -> tuple:
    """
    Find the most common color in the image.
    """
    # Reshape the image to be a list of pixels
    pixels = img.reshape(-1, 4)

    # Remove transparent pixels
    pixels = pixels[pixels[:, 3] > 0] if img.shape[2] == 4 else pixels

    # Find the most common color
    if pixels.size == 0:
        return None
    most_common_color = max(set(map(tuple, pixels)),
                            key=list(map(tuple, pixels)).count)

    return most_common_color
