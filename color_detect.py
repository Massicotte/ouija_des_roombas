"""Détection des 2 Roombas par AUTOCOLLANTS DE COULEUR (HSV).
Roumba A = sticker ROUGE, Roumba B = sticker BLEU (modifiable dans la config).
Seuils de saturation / luminosité / aire réglables en live dans l'interface."""

import cv2
import numpy as np

from config import load_config

CFG = load_config()
KERNEL = np.ones((5, 5), np.uint8)


def detect(frame):
    """Renvoie {robot_name: (cx, cy, rayon_px)} pour chaque couleur détectée.
    Le rayon est celui du disque équivalent au contour trouvé."""
    sat_min = int(CFG.get("sat_min", 120))
    lumi_min = int(CFG.get("lumi_min", 70))
    aire_min = max(1, int(CFG.get("aire_min", 500)))
    robots = [
        ("Roumba A", CFG.get("robot_a_colors", [[0, 10], [170, 180]])),   # rouge
        ("Roumba B", CFG.get("robot_b_colors", [[100, 130]])),            # bleu
    ]
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (7, 7), 0), cv2.COLOR_BGR2HSV)
    out = {}
    for name, ranges in robots:
        masque = np.zeros(hsv.shape[:2], np.uint8)
        for (h_min, h_max) in ranges:
            masque |= cv2.inRange(hsv, (h_min, sat_min, lumi_min), (h_max, 255, 255))
        masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN, KERNEL)
        masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, KERNEL)
        contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= aire_min]
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        rayon = int(np.sqrt(cv2.contourArea(c) / np.pi))
        out[name] = (cx, cy, rayon)
    return out
