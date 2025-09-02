from PIL import Image
import numpy as np
import os
import cv2

from utils.image_utils import show_image, \
    read_image_with_alpha


INPUT_FILE = 'arya.png'


def extract_sub_decals(input_file: str):

    # Eingabedatei
    filename_without_extension = os.path.splitext(input_file)[0]
    input_path = f"./data/input/{input_file}"
    output_dir = f"./data/output/{filename_without_extension}"
    os.makedirs(output_dir, exist_ok=True)

    # Bild laden
    img = read_image_with_alpha(input_path)

    # show_image(img)

    # Hintergrundfarbe finden
    # Für schnellere Entwicklung nehmen wir den ersten Pixel
    # oben links
    background_color = img[0, 0, :]

    # Entscheiden, ob der Hintergrund transparent ist
    background_is_transparent: bool = False
    if background_color[3] == 0:
        background_is_transparent = True

    if background_is_transparent:
        print("Hintergrund ist transparent.")
        # Alpha-Kanal extrahieren
        alpha = img[:, :, 3]

        # Binärmaske für nicht-transparente Pixel
        mask = (alpha > 0).astype(np.uint8) * 255
        print(mask)

    else:
        print("Hintergrund ist nicht transparent.")

        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        _, mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Konturen finden
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Gefundene Konturen: {len(contours)}")

    part_index = 1
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        # Sehr kleine Teile überspringen (Rauschen)
        if area < 5000:  # <- hier Grenze anpassen
            continue

        # Teil ausschneiden
        cropped = img[y:y+h, x:x+w].copy()

        # Transparenz für Hintergrund erhalten
        part_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(part_mask, [cnt - [x, y]], -
                         1, 255, thickness=cv2.FILLED)
        cropped[:, :, 3] = cv2.bitwise_and(cropped[:, :, 3], part_mask)

        # Convert BGR to RGB
        if cropped.shape[2] == 4:
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGRA2RGBA)
        else:
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGBA)

        # Speichern
        out_path = os.path.join(output_dir, f"part_{part_index}.png")
        Image.fromarray(cropped).save(out_path)
        part_index += 1

    print(f"{part_index-1} Teile gespeichert in '{output_dir}'.")


extract_sub_decals(INPUT_FILE)
