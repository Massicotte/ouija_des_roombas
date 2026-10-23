"""
Détection des 2 Roombas par marqueurs ArUco (identités garanties, zéro faux positif).
Chaque Roomba porte un marqueur ArUco imprimé et collé dessus :
  - Roumba A -> id `robot_a_id` (0 par défaut)
  - Roumba B -> id `robot_b_id` (1 par défaut)
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


def _make_params(a):
    """Paramètres tolérants pour caméra de qualité médiocre
    (flou, bruit, faible contraste)."""
    if hasattr(a, "DetectorParameters"):
        p = a.DetectorParameters()
    else:
        p = a.DetectorParameters_create()
    p.adaptiveThreshConstant = 7
    p.minMarkerPerimeterRate = 0.02
    p.maxMarkerPerimeterRate = 8.0
    p.polygonalApproxAccuracyRate = 0.06
    p.minCornerDistanceRate = 0.02
    p.minDistanceToBorder = 2
    p.perspectiveRemovePixelPerCell = 8
    p.errorCorrectionRate = 1.0
    if hasattr(a, "CORNER_REFINE_SUBPIX"):
        p.cornerRefinementMethod = a.CORNER_REFINE_SUBPIX
    return p


def _make_detector():
    """Compat OpenCV 4.3-4.7 (fonctions _create) et 4.8+/5.x (classes)."""
    a = cv2.aruco
    dic = a.getPredefinedDictionary(DICTS[CFG["aruco_dict"]])
    params = _make_params(a)
    if hasattr(a, "ArucoDetector"):
        det = a.ArucoDetector(dic, params)
        return lambda gray: det.detectMarkers(gray)
    return lambda gray: a.detectMarkers(gray, dic, parameters=params)


_detect = None


def _get_dict(name=None):
    return cv2.aruco.getPredefinedDictionary(DICTS[name or CFG["aruco_dict"]])


ROBOT_A_ID = CFG["robot_a_id"]
ROBOT_B_ID = CFG["robot_b_id"]


def _detect_all_scales(gray):
    """Détection multi-échelle avec prétraitement pour caméra médiocre :
    1) frame originale, 2) upscale x2, 3) upscale x2 + CLAHE (contraste)."""
    global _detect
    results = []
    corners, ids, _ = _detect(gray)
    if ids is not None and len(ids):
        return corners, ids, 1.0
    # upscale x2 : les petits marqueurs flous deviennent détectables
    up = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    corners, ids, _ = _detect(up)
    if ids is not None and len(ids):
        return corners, ids, 0.5
    # upscale + CLAHE : rescue des basses lumières / faible contraste
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    corners, ids, _ = _detect(clahe.apply(up))
    if ids is not None and len(ids):
        return corners, ids, 0.5
    # dernier recours : débruitage + sharpening sur l'upscale
    den = cv2.bilateralFilter(up, 5, 40, 40)
    sharp = cv2.addWeighted(up, 1.5, cv2.GaussianBlur(up, (0, 0), 3), -0.5, 0)
    corners, ids, _ = _detect(sharp)
    if ids is not None and len(ids):
        return corners, ids, 0.5
    return [], None, 1.0


def detect(frame):
    """Renvoie {id: (cx, cy, radius_px)} pour les marqueurs A et B seulement.
    radius_px = distance max coin-centre du marqueur (pour le dessin)."""
    global _detect, ROBOT_A_ID, ROBOT_B_ID
    ROBOT_A_ID, ROBOT_B_ID = CFG["robot_a_id"], CFG["robot_b_id"]
    if _detect is None:
        _detect = _make_detector()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, scale = _detect_all_scales(gray)
    out = {}
    if ids is None:
        return out
    for (pts, mid) in zip(corners, ids.flatten()):
        mid = int(mid)
        if mid not in (ROBOT_A_ID, ROBOT_B_ID):
            continue
        pts = np.asarray(pts).reshape(-1, 2) * scale  # retour aux pixels originaux
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
        radius = float(max(abs(pts[:, 0] - cx).max(), abs(pts[:, 1] - cy).max()))
        out[mid] = (cx, cy, radius)
    return out
