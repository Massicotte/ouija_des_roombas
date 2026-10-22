"""Configuration partagée du Ouija des Roumbas. Sauvegardée dans grid_config.json."""

import json
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "grid_config.json"

DEFAULTS = {
    "cols": 8,
    "rows": 8,
    "words": [],
    "homography": None,
    "dark_on_light": True,
    "min_area": 1200,
    "max_area": 250000,
    "min_circularity": 0.6,
    "collision_dist_cells": 0.5,
    "collision_cooldown_s": 2.0,
    "cam_index": 0,
    "cam_width": 1920,
    "cam_height": 1080,
    "warp_size": [800, 800],
    "record": True,
    "record_dir": "captures",
    "events_csv": "collisions.csv",
    "explosion_duration_s": 0.8,
    "n_particles": 24,
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
