"""
Détection des 2 Roombas par marqueurs ArUco (identités garanties, zéro faux positif).
Chaque Roomba porte un marqueur ArUco imprimé et collé dessus :
  - Roumba A -> id `robot_a_id` (0 par défaut)
  - Roumba B -> id `robot_b_id` (1 par défaut)
Imprime les marqueurs une fois avec : python make_markers.py
"""

import cv2
import numpy as np

from config import load_config

CFG = load_config()

DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
}


def _make_detector():
    """Compat OpenCV 4.3-4.7 (fonctions _create) et 4.8+/5.x (classes)."""
    a = cv2.aruco
    dic = a.getPredefinedDictionary(DICTS[CFG["aruco_dict"]])
    if hasattr(a, "ArucoDetector"):
        params = a.DetectorParameters()
        det = a.ArucoDetector(dic, params)
        return lambda gray: det.detectMarkers(gray)
    params = a.DetectorParameters_create()
    return lambda gray: a.detectMarkers(gray, dic, parameters=params)


_detect = None


def _get_dict(name=None):
    return cv2.aruco.getPredefinedDictionary(DICTS[name or CFG["aruco_dict"]])


ROBOT_A_ID = CFG["robot_a_id"]
ROBOT_B_ID = CFG["robot_b_id"]


def detect(frame):
    """Renvoie {id: (cx, cy, radius_px)} pour les marqueurs A et B seulement.
    radius_px = distance max coin-centre du marqueur (pour le dessin)."""
    global _detect, ROBOT_A_ID, ROBOT_B_ID
    ROBOT_A_ID, ROBOT_B_ID = CFG["robot_a_id"], CFG["robot_b_id"]
    if _detect is None:
        _detect = _make_detector()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _detect(gray)
    out = {}
    if ids is None:
        return out
    for (pts, mid) in zip(corners, ids.flatten()):
        mid = int(mid)
        if mid not in (ROBOT_A_ID, ROBOT_B_ID):
            continue
        pts = np.asarray(pts).reshape(-1, 2)  # (4,2) quelle que soit la version d'OpenCV
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
        radius = float(max(abs(pts[:, 0] - cx).max(), abs(pts[:, 1] - cy).max()))
        out[mid] = (cx, cy, radius)
    return out
