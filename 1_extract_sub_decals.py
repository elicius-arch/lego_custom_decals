from PIL import Image
import numpy as np
import os
import cv2

INPUT_FILE = 'arya.png'

def extract_sub_decals(input_file: str):

    # Eingabedatei
    filename_without_extension = os.path.splitext(input_file)[0]
    input_path = f"./data/input/{input_file}"
    output_dir = f"./data/output/{filename_without_extension}"
    os.makedirs(output_dir, exist_ok=True)

    # Bild laden
    img = Image.open(input_path).convert("RGBA")
    arr = np.array(img)

    # Alpha-Kanal extrahieren
    alpha = arr[:, :, 3]

    # Binärmaske für nicht-transparente Pixel
    mask = (alpha > 0).astype(np.uint8) * 255

    # Konturen finden
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Gefundene Konturen: {len(contours)}")

    part_index = 1
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        # Sehr kleine Teile überspringen (Rauschen)
        if area < 5000:  # <- hier Grenze anpassen
            continue

        # Teil ausschneiden
        cropped = arr[y:y+h, x:x+w].copy()

        # Transparenz für Hintergrund erhalten
        part_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(part_mask, [cnt - [x, y]], -1, 255, thickness=cv2.FILLED)
        cropped[:, :, 3] = cv2.bitwise_and(cropped[:, :, 3], part_mask)

        # Speichern
        out_path = os.path.join(output_dir, f"part_{part_index}.png")
        Image.fromarray(cropped).save(out_path)
        part_index += 1

    print(f"{part_index-1} Teile gespeichert in '{output_dir}'.")


extract_sub_decals(INPUT_FILE)