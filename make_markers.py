"""
Génère et imprime les 2 marqueurs ArUco à coller sur les Roombas.

Usage:
    python make_markers.py            # affiche les tailles et sauve en PNG
    python make_markers.py --print    # ouvre une fenêtre d'aperçu
"""

import argparse
from pathlib import Path

import cv2

from aruco_detect import _get_dict
from config import load_config

CFG = load_config()
OUT_DIR = Path(__file__).parent / "markers"


def generate():
    OUT_DIR.mkdir(exist_ok=True)
    aruco_dict = _get_dict(CFG["aruco_dict"])
    side_px = 400  # résolution du PNG
    for robot_id, name in [(CFG["robot_a_id"], "A"), (CFG["robot_b_id"], "B")]:
        img = cv2.aruco.generateImageMarker(aruco_dict, robot_id, side_px)
        path = OUT_DIR / f"marker_{name}_id{robot_id}.png"
        cv2.imwrite(str(path), img)
        print(f"Marqueur {name} (id {robot_id}) -> {path}")
        print(f"  À imprimer à {CFG['marker_side_cm']} cm de côté, bien à plat sur le Roomba.")
    return OUT_DIR


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()
    out = generate()
    if args.print:
        for p in sorted(out.glob("*.png")):
            cv2.imshow(str(p), cv2.imread(str(p)))
            cv2.waitKey(0)
        cv2.destroyAllWindows()
