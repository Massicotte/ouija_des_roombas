"""Configuration partagée du Ouija des Roumbas. Sauvegardée dans grid_config.json."""

import json
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "grid_config.json"

DEFAULTS = {
    "cols": 8,
    "rows": 8,
    "words": [],
    "homography": None,
    # --- ArUco ---
    "aruco_dict": "DICT_4X4_50",
    "robot_a_id": 0,
    "robot_b_id": 1,
    "marker_side_cm": 8.0,
    # --- collision ---
    "collision_dist_cells": 0.5,
    "collision_cooldown_s": 2.0,
    # --- détection par couleur (autocollants) ---
    "sat_min": 120,
    "lumi_min": 70,
    "aire_min": 500,
    "robot_a_colors": [[0, 10], [170, 180]],
    "robot_b_colors": [[100, 130]],
    # --- explosion ---
    "explosion_scale": 1.0,
    "explosion_sprites": "assets/explosion_sprites.png",
    "explosion_fps": 15,
    # --- caméra / sortie ---
    "cam_index": 0,
    "cam_width": 1920,
    "cam_height": 1080,
    "warp_size": [800, 800],
    "record": True,
    "record_dir": "captures",
    "events_csv": "collisions.csv",
}

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_config():
    cfg = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            cfg.update(json.load(f))
    return cfg


def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def cell_name(col, row, cols, rows):
    c = max(0, min(int(col), cols - 1))
    r = max(0, min(int(row), rows - 1))
    return f"{LETTERS[c]}{r + 1}"
