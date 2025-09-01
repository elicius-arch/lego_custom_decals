import cv2
import numpy as np
from sklearn.cluster import KMeans

import cv2
import numpy as np


def read_image_with_alpha(file_path: str):
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        b, g, r = cv2.split(img)
        alpha = np.ones_like(b) * 255
        img = cv2.merge([b, g, r, alpha])
    return img


def show_image(img: np.ndarray):
    cv2.imshow("Image", img)
    cv2.waitKey(0)


def show_mask(mask: np.ndarray):
    show_image(mask.astype(np.uint8) * 255)


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


def extract_face_expression(input_path: str, output_path: str, smiley_path: str, smiley_path_2: str):
    img = read_image_with_alpha(input_path)

    # Reduziere die Farben des Bildes
    most_common_colors = find_main_colors_of_img(img, num_colors=4)
    # print(len(most_common_colors), 'Hauptfarben:')
    # print(most_common_colors)
    # display_color_list_as_img(most_common_colors)
    img = reduce_colors_of_img_to_colors_of_color_list(img, most_common_colors)

    show_image(img)

    mask = create_mask(img)

    # show_image(mask)

    # Invert the mask
    mask = cv2.bitwise_not(mask)

    # show_image(mask)

    # region Zuschneiden

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
    else:
        print('No contours found')
        return

    # Zuschneiden auf Maske
    cropped_img = img[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]

    # Alpha-Kanal = Maske
    applied_image = apply_mask_for_image(cropped_mask, cropped_img)

    # Speichern
    cv2.imwrite(output_path, applied_image)

    # region Bereiche außen entfernen

    # show_image(cropped_mask)

    # Füge alle Bereiche zwischen dem Bildrand und der Maske zur Maske hinzu
    outline_mask = create_outline_mask(cropped_mask)
    # show_image(outline_mask)

    # Hole die Farbauflistung
    face_content_mask = apply_mask_for_image(
        outline_mask, cropped_img, fill_with_green=True)
    color_list = get_color_list(cropped_img, face_content_mask)
    # print(color_list)
    # print(len(color_list))
    display_color_list_as_img(color_list)

    # Finde die häufigste Farbe
    most_common_color = find_most_common_color(cropped_img)
    if most_common_color is not None:
        print("Häufigste Farbe:", most_common_color)

    # Erstelle eine Maske aus der häufigsten Farbe
    if most_common_color is not None:
        # Debugging: Print shapes and types
        print("most_common_color:", most_common_color)
        # print("most_common_color shape:", most_common_color.shape)
        print("cropped_img shape:", cropped_img.shape)
        print("cropped_img dtype:", cropped_img.dtype)

        # Ensure most_common_color is a NumPy array with the same dtype as cropped_img
        most_common_color = np.array(
            most_common_color, dtype=cropped_img.dtype)

        # Debugging: Print after conversion
        print("most_common_color (converted):", most_common_color)

        # Create a mask for the most common color
        most_common_color_mask = np.all(
            cropped_img[:, :, :len(most_common_color)] == most_common_color, axis=-1
        ).astype(np.uint8) * 255

        # Debugging: Print mask stats
        print("most_common_color_mask shape:", most_common_color_mask.shape)
        print("most_common_color_mask unique values:",
              np.unique(most_common_color_mask))

        # Maske weichzeichnen (zum Entfernen von einzelnen Pixeln)
        most_common_color_mask = blur_mask(most_common_color_mask)
    else:
        most_common_color_mask = np.zeros_like(cropped_img[:, :, 0])

    most_common_color_outline_mask = create_outline_mask(
        most_common_color_mask)
    most_common_color_mask = cv2.bitwise_or(
        most_common_color_mask, cv2.bitwise_not(most_common_color_outline_mask))
    most_common_color_mask = cv2.bitwise_not(most_common_color_mask)
    # most_common_color_mask = blur_mask(most_common_color_mask)
    show_image(most_common_color_mask)

    # Invertiere die Maske
    outline_mask = cv2.bitwise_not(outline_mask)

    # Füge die Maske mit den Außenbereichen zur Maske hinzu
    face_mask = cv2.bitwise_or(cropped_mask, outline_mask)

    # Invertiere die Maske
    face_mask = cv2.bitwise_not(face_mask)

    # show_image(face_mask)

    smiley_img = apply_mask_for_image(face_mask, cropped_img)
    smiley_img_2 = apply_mask_for_image(most_common_color_mask, cropped_img)

    cv2.imwrite(smiley_path, smiley_img)

    cv2.imwrite(smiley_path_2, smiley_img_2)


def blur_mask(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    blurred_mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, smooth_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)
    return smooth_mask


def extend_mask_by_n_pixels(mask: np.ndarray, n: int) -> np.ndarray:
    """
    Extend the mask by n pixels using morphological dilation.
    """
    kernel = np.ones((n, n), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def get_color_list(img: np.ndarray, mask: np.ndarray = None, ignore_transparent: bool = True, return_counts: bool = False):
    """
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


def remove_most_common_color(input_file: str, output_file: str):
    img = read_image_with_alpha(input_file)

    # print(img.shape)

    # # Only consider pixels where alpha == 0
    # visible_mask = get_visible_mask(img)
    # visible_pixels = get_only_visible_pixels(img)

    # # Find the most common color among visible pixels
    # pixels_list = [tuple(pixel) for pixel in visible_pixels]
    # if not pixels_list:
    #     print("No visible pixels found.")
    #     return

    # most_common_color = max(set(pixels_list), key=pixels_list.count)

    # # Create a mask for the most common color (only where alpha > 0)
    # mask = np.all(img[:, :, :3] == most_common_color, axis=-1) & visible_mask
    # mask = cv2.bitwise_not(mask)

    '''mask = create_mask(img)
    mask = cv2.bitwise_not(mask)

    show_mask(mask)

    # Set alpha to 0 for the most common color
    # img[mask, 3] = 0

    # Save the result
    cv2.imwrite(output_file, img)'''


# Beispiel-Aufruf:
folder = "data/output/arya"
temp_file = f"{folder}/face_expression.png"
temp_file_2 = f"{folder}/face_color_removed.png"
smiley_file = f"{folder}/face_smiley.png"
smiley_file_2 = f"{folder}/face_smiley_2.png"

extract_face_expression(f"{folder}/part_10.png",
                        temp_file, smiley_file, smiley_file_2)

remove_most_common_color(temp_file, temp_file_2)
