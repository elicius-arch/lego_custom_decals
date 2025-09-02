import cv2
import numpy as np

from utils.image_utils import read_image_with_alpha, \
    show_image, display_color_list_as_img, find_main_colors_of_img, \
    reduce_colors_of_img_to_colors_of_color_list, create_mask, \
    apply_mask_for_image, get_color_list, find_most_common_color, \
    create_outline_mask, blur_mask


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


# Beispiel-Aufruf:
folder = "data/output/arya"
temp_file = f"{folder}/face_expression.png"
temp_file_2 = f"{folder}/face_color_removed.png"
smiley_file = f"{folder}/face_smiley.png"
smiley_file_2 = f"{folder}/face_smiley_2.png"

extract_face_expression(f"{folder}/part_10.png",
                        temp_file, smiley_file, smiley_file_2)
