"""
OUIJA DES ROOMBAS

Usage:
    python main.py

Commandes vidéo : cliquer 4 coins (mode calib), 'r' reset liste, 'q'/ESC quitter
"""

import csv
import time
from collections import deque
import datetime
import threading
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

from config import load_config, save_config, cell_name, LETTERS
import aruco_detect
import color_detect

ROBOT_A = {"name": "Roumba A", "id_key": "robot_a_id", "draw": (60, 60, 230), "trace": (0, 0, 255)}
ROBOT_B = {"name": "Roumba B", "id_key": "robot_b_id", "draw": (230, 120, 40), "trace": (255, 0, 0)}
ROBOTS = [ROBOT_A, ROBOT_B]

TRAJ_LENGTH = 2000  # points max gardés par robot pour la trajectoire

DIRECTIONS = {"Horizontal (vers la droite)": "H", "Vertical (vers le bas)": "V"}
DIR_LABEL = {v: k for k, v in DIRECTIONS.items()}

PANEL_FRACTION = 1 / 8
VIDEO_W = 960
VIDEO_H = 540


def draw_grid_on_view(view, mapper, cols, rows):
    """Dessine la grille sur la vue caméra via l'homographie inverse.
    Les lettres (A, B, C..) sont placées HORS de la grille, au-dessus des colonnes,
    et les chiffres (1, 2, 3..) HORS de la grille, à gauche des lignes."""
    for i in range(cols + 1):
        u = i / cols
        p0 = mapper.to_pixels(u, 0.0)
        p1 = mapper.to_pixels(u, 1.0)
        cv2.line(view, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                 (255, 255, 255), 1, cv2.LINE_AA)
    for j in range(rows + 1):
        v = j / rows
        p0 = mapper.to_pixels(0.0, v)
        p1 = mapper.to_pixels(1.0, v)
        cv2.line(view, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                 (255, 255, 255), 1, cv2.LINE_AA)
    # taille du texte proportionnelle à la taille d'une case (en pixels)
    cell_px = mapper.to_pixels(0.5 / cols, 0.5 / rows)
    edge = mapper.to_pixels(0.0, 0.0)
    cell_w = abs(cell_px[0] - edge[0]) * 2
    font_scale = max(0.4, min(1.0, cell_w / 80.0))
    # lettres AU-DESSUS de la grille (v négatif = hors grille), centrées sur chaque colonne
    for i in range(cols):
        px = mapper.to_pixels((i + 0.5) / cols, -0.22 / rows)
        cv2.putText(view, LETTERS[i],
                    (int(px[0]) - 8, int(px[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    # chiffres À GAUCHE de la grille (u négatif = hors grille), centrés sur chaque ligne
    for j in range(rows):
        px = mapper.to_pixels(-0.22 / cols, (j + 0.5) / rows)
        cv2.putText(view, str(j + 1),
                    (int(px[0]) - 6, int(px[1]) + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)


def word_cell(word, start, cols, rows):
    """Un mot occupe UNE SEULE case : sa case de départ (ex: 'B3').
    Renvoie (col, row) de cette case, ou None si la case n'existe pas."""
    import re
    sm = re.match(r"^([A-Z])(\d+)$", str(start).upper())
    if not sm:
        return None
    sc = ord(sm.group(1)) - ord("A")
    sr = int(sm.group(2)) - 1
    if sc < 0 or sc >= cols or sr < 0 or sr >= rows:
        return None
    return sc, sr


def draw_words_on_view(view, mapper, cols, rows, words):
    """Dessine chaque mot ENTIER dans SA case (une lettre ne prend plus une case).
    La police est réduite pour que le mot tienne dans la case, quelle que soit sa longueur."""
    for word, start, direction in words:
        cell = word_cell(word, start, cols, rows)
        if cell is None:
            continue
        c, r = cell
        text = str(word).replace(" ", "").upper()
        # géométrie de la case en pixels (par homographie inverse)
        top_left = mapper.to_pixels(c / cols, r / rows)
        bot_right = mapper.to_pixels((c + 1) / cols, (r + 1) / rows)
        cw = abs(bot_right[0] - top_left[0])
        ch = abs(bot_right[1] - top_left[1])
        center = mapper.to_pixels((c + 0.5) / cols, (r + 0.5) / rows)
        # police la plus grande qui tient dans la case (avec une petite marge)
        thickness = 1 if cw < 120 else 2  # trait fin si la case est petite
        scale = 0.1
        while scale < 5.0:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale + 0.05, thickness)
            if tw > cw * 0.92 or th > ch * 0.92:
                break
            scale += 0.05
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        org = (int(center[0] - tw / 2), int(center[1] + th / 2))
        cv2.putText(view, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (120, 220, 255), thickness, cv2.LINE_AA)


def word_for_cell(cell, words, cols=8, rows=8):
    """Le mot n'est actif QUE dans sa propre case (une case = un mot)."""
    import re
    m = re.match(r"^([A-Z])(\d+)$", cell)
    if not m:
        return None
    col = ord(m.group(1)) - ord("A")
    row = int(m.group(2)) - 1
    for word, start, direction in words:
        c = word_cell(word, start, cols, rows)
        if c is None:
            continue
        if c == (col, row):
            return str(word).upper()
    return None


def load_sprites(path):
    sheet = cv2.imread(str(Path(__file__).parent / path), cv2.IMREAD_UNCHANGED)
    if sheet is None:
        raise SystemExit(f"Sprite sheet introuvable : {path}")
    if sheet.ndim == 2:
        sheet = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGRA)
    elif sheet.shape[2] == 3:
        sheet = cv2.cvtColor(sheet, cv2.COLOR_BGR2BGRA)
    h = sheet.shape[0]
    n = sheet.shape[1] // h
    return [sheet[:, i * h:(i + 1) * h] for i in range(n)]


class ExplosionAnimation:
    def __init__(self, sprites, pos, scale, fps):
        self.frames = sprites
        self.pos = (int(pos[0]), int(pos[1]))
        self.scale = scale
        self.t0 = time.time()
        self.frame_dur = 1.0 / max(fps, 1)

    def alive(self):
        return self._idx() < len(self.frames)

    def _idx(self):
        return int((time.time() - self.t0) / self.frame_dur)

    def draw(self, img):
        idx = self._idx()
        if idx >= len(self.frames):
            return
        spr = self.frames[idx]
        h, w = spr.shape[:2]
        size = max(8, int(w * self.scale))
        spr = cv2.resize(spr, (size, size), interpolation=cv2.INTER_NEAREST)
        x, y = self.pos[0] - size // 2, self.pos[1] - size // 2
        h2, w2 = spr.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(img.shape[1], x + w2), min(img.shape[0], y + h2)
        if x1 <= x0 or y1 <= y0:
            return
        region = img[y0:y1, x0:x1]
        sub = spr[y0 - y:y0 - y + (y1 - y0), x0 - x:x0 - x + (x1 - x0)]
        alpha = sub[:, :, 3:4].astype(np.float32) / 255.0
        img[y0:y1, x0:x1] = (sub[:, :, :3] * alpha + region * (1 - alpha)).astype(np.uint8)


class GridMapper:
    def __init__(self, H):
        self.H = np.array(H, dtype=np.float32)
        self.Hinv = np.linalg.inv(self.H)  # unités grille [0..1] -> pixels

    def to_grid(self, pt, cols, rows):
        p = np.array([[pt]], dtype=np.float32)
        g = cv2.perspectiveTransform(p, self.H)[0][0]
        return g[0] * cols, g[1] * rows

    def to_pixels(self, gu, gv):
        """coordonnées grille normalisées [0..1] -> pixels caméra"""
        p = np.array([[[gu, gv]]], dtype=np.float32)
        q = cv2.perspectiveTransform(p, self.Hinv)[0][0]
        return float(q[0]), float(q[1])

    def warped(self, frame, cols, rows):
        w, h = 800, 800
        S = np.array([[w, 0, 0], [0, h, 0], [0, 0, 1]], dtype=np.float32)
        return cv2.warpPerspective(frame, S @ self.H, (w, h))


class App:
    def __init__(self, root):
        self.root = root
        root.title("Ouija des Roumbas 🤖💥")
        self.cfg = load_config()
        self.sprites = load_sprites(self.cfg["explosion_sprites"])
        self.mapper = GridMapper(self.cfg["homography"]) if self.cfg.get("homography") else None

        # état live
        self.words = [tuple(w) for w in self.cfg["words"]]
        self.corner_pts = []
        self.calib_mode = False
        self.show_grid = True
        self.show_words = True
        self.explosions = []
        self.collision_events = []  # [(cell, ts)]
        self.last_collision_t = -999.0
        self.collision_count = 0
        self.video_pos = {}  # {robot_name: (cx, cy)} dernières positions
        self.writer = None
        self.log = None
        self.log_file = None
        self.frame_w = self.cfg["cam_width"]
        self.frame_h = self.cfg["cam_height"]
        self.lock = threading.Lock()
        self.running = True
        self.cap = None
        self.latest_frame = None
        self.camera_error = None
        self.camera_running = True
        self._available_cams = []
        self._disp_offset = (0, 0)
        # trajectoires : deque par robot (dessinées derrière chaque Roomba)
        self.trails = {ROBOT_A["name"]: deque(maxlen=TRAJ_LENGTH),
                       ROBOT_B["name"]: deque(maxlen=TRAJ_LENGTH)}
        # réglage IMAGE (visuel) : saturation / luminosité / contraste de la
        # vidéo affichée et enregistrée, ajustables en live
        self.img_sat_gain = float(self.cfg.get("img_sat_gain", 1.0))
        self.img_val_gain = float(self.cfg.get("img_val_gain", 1.0))
        self.img_contrast = float(self.cfg.get("img_contrast", 1.0))

        self._build_ui()
        self._open_outputs()
        self._sync_writer()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        # la caméra s'ouvre en ARRIÈRE-PLAN : la fenêtre s'affiche immédiatement,
        # avec un statut visible pendant l'ouverture (qui peut être lente sous Windows)
        threading.Thread(target=self._open_camera, daemon=True).start()
        self.root.after(10, self._display_loop)

    # ================= UI =================
    def _build_ui(self):
        left = ttk.Frame(self.root, padding=5)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- vidéo (canvas Tkinter, cliquable) ---
        self.canvas = tk.Canvas(left, width=VIDEO_W, height=VIDEO_H, bg="black")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_video_click)
        self.hint = tk.StringVar(value="Prêt. Clique 'Calibrer les 4 coins' pour commencer.")
        ttk.Label(left, textvariable=self.hint).pack(anchor="w")

        right = ttk.Frame(self.root, padding=5)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        # --- Grille ---
        g = ttk.LabelFrame(right, text="Grille", padding=8)
        g.pack(fill=tk.X)
        ttk.Label(g, text="Colonnes:").grid(row=0, column=0)
        self.cols_var = tk.IntVar(value=self.cfg["cols"])
        ttk.Spinbox(g, from_=2, to=26, width=4, textvariable=self.cols_var,
                    command=self._apply_grid).grid(row=0, column=1, padx=4)
        ttk.Label(g, text="Lignes:").grid(row=0, column=2)
        self.rows_var = tk.IntVar(value=self.cfg["rows"])
        ttk.Spinbox(g, from_=2, to=30, width=4, textvariable=self.rows_var,
                    command=self._apply_grid).grid(row=0, column=3, padx=4)
        self.calib_btn = ttk.Button(g, text="📍 Calibrer les 4 coins", command=self._start_calib)
        self.calib_btn.grid(row=1, column=0, columnspan=4, pady=(6, 0), sticky="we")
        self.calib_status = tk.StringVar(
            value="Homographie : " + ("calibrée ✔" if self.mapper else "non calibrée ✘"))
        ttk.Label(g, textvariable=self.calib_status).grid(row=2, column=0, columnspan=4, sticky="w")

        # --- Mots ---
        w = ttk.LabelFrame(right, text="Mots de la grille", padding=8)
        w.pack(fill=tk.X, pady=6)
        row = ttk.Frame(w); row.pack(fill=tk.X)
        ttk.Label(row, text="Mot:").pack(side=tk.LEFT)
        self.word_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.word_var, width=12).pack(side=tk.LEFT, padx=3)
        ttk.Label(row, text="Case:").pack(side=tk.LEFT)
        self.start_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.start_var, width=4).pack(side=tk.LEFT, padx=3)
        ttk.Button(row, text="+", width=3, command=self._add_word).pack(side=tk.LEFT, padx=3)
        self.word_list = tk.Listbox(w, height=5)
        self.word_list.pack(fill=tk.X, pady=4)
        ttk.Button(w, text="Supprimer le mot sélectionné", command=self._remove_word).pack(fill=tk.X)
        vis = ttk.Frame(w); vis.pack(fill=tk.X, pady=(4, 0))
        self.show_words_var = tk.BooleanVar(value=self.show_words)
        ttk.Checkbutton(vis, text="Afficher les mots sur la grille", variable=self.show_words_var,
                        command=lambda: self._set_show("words")).pack(side=tk.LEFT)
        self.show_grid_var = tk.BooleanVar(value=self.show_grid)
        ttk.Checkbutton(vis, text="Afficher la grille", variable=self.show_grid_var,
                        command=lambda: self._set_show("grid")).pack(side=tk.LEFT, padx=6)

        # --- Explosion ---
        e = ttk.LabelFrame(right, text="Explosion 💥", padding=8)
        e.pack(fill=tk.X, pady=6)
        ttk.Label(e, text="Taille:").grid(row=0, column=0)
        self.scale_var = tk.DoubleVar(value=float(self.cfg["explosion_scale"]))
        ttk.Scale(e, from_=0.2, to=5.0, variable=self.scale_var,
                  command=lambda *_: self._apply_cfg("explosion_scale", round(self.scale_var.get(), 2))).grid(row=0, column=1, sticky="we", padx=6)
        self.scale_label = tk.StringVar()
        ttk.Label(e, textvariable=self.scale_label).grid(row=0, column=2)
        ttk.Label(e, text="Vitesse (fps):").grid(row=1, column=0)
        self.fps_var = tk.IntVar(value=self.cfg["explosion_fps"])
        ttk.Spinbox(e, from_=5, to=30, width=4, textvariable=self.fps_var,
                    command=lambda: self._apply_cfg("explosion_fps", self.fps_var.get())).grid(row=1, column=1, sticky="w", padx=6)
        ttk.Button(e, text="💥 Aperçu", command=self._preview_explosion).grid(row=2, column=0, columnspan=2, pady=4)
        self._update_scale_label()

        # --- Collisions ---
        c = ttk.LabelFrame(right, text="Collisions", padding=8)
        c.pack(fill=tk.X, pady=6)
        self.count_var = tk.StringVar(value="0 collision")
        ttk.Label(c, textvariable=self.count_var).pack(anchor="w")
        ttk.Button(c, text="♻ Reset de la liste", command=self._reset_events).pack(fill=tk.X, pady=3)
        self.events_list = tk.Listbox(c, height=6)
        self.events_list.pack(fill=tk.X)

        # --- Réglages divers ---
        s = ttk.LabelFrame(right, text="Réglages", padding=8)
        s.pack(fill=tk.X, pady=6)
        ttk.Label(s, text="Source vidéo:").pack(anchor="w")
        cam_row = ttk.Frame(s); cam_row.pack(fill=tk.X, pady=(2, 4))
        self.cam_combo = ttk.Combobox(cam_row, state="readonly", width=14)
        self.cam_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.cam_combo.bind("<<ComboboxSelected>>", self._switch_camera)
        ttk.Button(cam_row, text="⟳", width=3,
                   command=self._refresh_cam_list).pack(side=tk.LEFT, padx=3)
        self._refresh_cam_list()
        self.rec_var = tk.BooleanVar(value=self.cfg["record"])
        ttk.Checkbutton(s, text="Enregistrer la vidéo", variable=self.rec_var,
                        command=lambda: self._apply_cfg("record", self.rec_var.get())).pack(anchor="w")
        ttk.Label(s, text="Distance collision (cases):").pack(anchor="w")
        self.dist_var = tk.DoubleVar(value=self.cfg["collision_dist_cells"])
        ttk.Spinbox(s, from_=0.1, to=2.0, increment=0.1, width=5,
                    textvariable=self.dist_var,
                    command=lambda: self._apply_cfg("collision_dist_cells", float(self.dist_var.get()))).pack(anchor="w")
        ttk.Label(s, text="Délai entre collisions (s):").pack(anchor="w")
        cd = ttk.Frame(s); cd.pack(fill=tk.X)
        self.cooldown_var = tk.DoubleVar(value=float(self.cfg["collision_cooldown_s"]))
        self.cooldown_label = tk.StringVar()
        ttk.Scale(cd, from_=0.5, to=30.0, variable=self.cooldown_var,
                  command=lambda *_: self._on_cooldown_slider()).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(cd, textvariable=self.cooldown_label, width=6).pack(side=tk.LEFT)
        self._update_cooldown_label()
        ttk.Button(s, text="💾 Sauvegarder la config", command=self._save_cfg).pack(fill=tk.X, pady=(6, 0))

        # --- Détection couleur (seuils réglables en live) ---
        d = ttk.LabelFrame(right, text="Détection couleur 🎨", padding=8)
        d.pack(fill=tk.X, pady=6)
        ttk.Label(d, text="Saturation min:").grid(row=0, column=0, sticky="w")
        self.sat_var = tk.IntVar(value=int(self.cfg.get("sat_min", 120)))
        self.sat_label = tk.StringVar()
        ttk.Scale(d, from_=0, to=255, variable=self.sat_var,
                  command=lambda *_: self._apply_color_cfg("sat_min", self.sat_var.get())).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Label(d, textvariable=self.sat_label, width=4).grid(row=0, column=2)
        ttk.Label(d, text="Luminosité min:").grid(row=1, column=0, sticky="w")
        self.lumi_var = tk.IntVar(value=int(self.cfg.get("lumi_min", 70)))
        self.lumi_label = tk.StringVar()
        ttk.Scale(d, from_=0, to=255, variable=self.lumi_var,
                  command=lambda *_: self._apply_color_cfg("lumi_min", self.lumi_var.get())).grid(row=1, column=1, sticky="we", padx=6)
        ttk.Label(d, textvariable=self.lumi_label, width=4).grid(row=1, column=2)
        ttk.Label(d, text="Aire min (px):").grid(row=2, column=0, sticky="w")
        self.aire_var = tk.IntVar(value=int(self.cfg.get("aire_min", 500)))
        self.aire_label = tk.StringVar()
        ttk.Scale(d, from_=10, to=20000, variable=self.aire_var,
                  command=lambda *_: self._apply_color_cfg("aire_min", self.aire_var.get())).grid(row=2, column=1, sticky="we", padx=6)
        ttk.Label(d, textvariable=self.aire_label, width=7).grid(row=2, column=2)
        ttk.Button(d, text="🧹 Effacer les traces", command=self._clear_trails).grid(
            row=3, column=0, columnspan=3, pady=(6, 0), sticky="we")
        d.columnconfigure(1, weight=1)

        # --- Réglage de l'IMAGE (visuel, en live) ---
        im = ttk.LabelFrame(right, text="Image 🖼", padding=8)
        im.pack(fill=tk.X, pady=6)
        ttk.Label(im, text="Saturation:").grid(row=0, column=0, sticky="w")
        self.img_sat_var = tk.DoubleVar(value=self.img_sat_gain)
        self.img_sat_label = tk.StringVar(value=f"x{self.img_sat_gain:.1f}")
        ttk.Scale(im, from_=0.0, to=3.0, variable=self.img_sat_var,
                  command=lambda *_: self._apply_img_cfg()).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Label(im, textvariable=self.img_sat_label, width=4).grid(row=0, column=2)
        ttk.Label(im, text="Luminosité:").grid(row=1, column=0, sticky="w")
        self.img_val_var = tk.DoubleVar(value=self.img_val_gain)
        self.img_val_label = tk.StringVar(value=f"x{self.img_val_gain:.1f}")
        ttk.Scale(im, from_=0.2, to=2.5, variable=self.img_val_var,
                  command=lambda *_: self._apply_img_cfg()).grid(row=1, column=1, sticky="we", padx=6)
        ttk.Label(im, textvariable=self.img_val_label, width=4).grid(row=1, column=2)
        ttk.Label(im, text="Contraste:").grid(row=2, column=0, sticky="w")
        self.img_cts_var = tk.DoubleVar(value=self.img_contrast)
        self.img_cts_label = tk.StringVar(value=f"x{self.img_contrast:.1f}")
        ttk.Scale(im, from_=0.5, to=2.5, variable=self.img_cts_var,
                  command=lambda *_: self._apply_img_cfg()).grid(row=2, column=1, sticky="we", padx=6)
        ttk.Label(im, textvariable=self.img_cts_label, width=4).grid(row=2, column=2)
        im.columnconfigure(1, weight=1)

    def _apply_img_cfg(self):
        """Applique instantanément les réglages d'image et les sauvegarde."""
        self.img_sat_gain = round(self.img_sat_var.get(), 2)
        self.img_val_gain = round(self.img_val_var.get(), 2)
        self.img_contrast = round(self.img_cts_var.get(), 2)
        self.img_sat_label.set(f"x{self.img_sat_gain:.1f}")
        self.img_val_label.set(f"x{self.img_val_gain:.1f}")
        self.img_cts_label.set(f"x{self.img_contrast:.1f}")
        self.cfg["img_sat_gain"] = self.img_sat_gain
        self.cfg["img_val_gain"] = self.img_val_gain
        self.cfg["img_contrast"] = self.img_contrast

    def _apply_color_cfg(self, key, value):
        self.cfg[key] = int(value)
        if key == "sat_min":
            self.sat_label.set(str(int(self.sat_var.get())))
        elif key == "lumi_min":
            self.lumi_label.set(str(int(self.lumi_var.get())))
        elif key == "aire_min":
            self.aire_label.set(str(int(self.aire_var.get())))

    def _clear_trails(self):
        for tr in self.trails.values():
            tr.clear()
        self.hint.set("Traces effacées")

        self.refresh_words()
        self._refresh_events()

    # ================= config live =================
    def _set_show(self, what):
        if what == "words":
            self.show_words = bool(self.show_words_var.get())
        else:
            self.show_grid = bool(self.show_grid_var.get())

    def _apply_grid(self):
        self._apply_cfg("cols", int(self.cols_var.get()))
        self._apply_cfg("rows", int(self.rows_var.get()))

    def _apply_cfg(self, key, value):
        self.cfg[key] = value
        if key == "record":
            self._sync_writer()

    def _update_scale_label(self, *_):
        self.scale_label.set(f"x{self.scale_var.get():.1f}")

    def _on_cooldown_slider(self):
        self._apply_cfg("collision_cooldown_s", round(self.cooldown_var.get(), 1))
        self._update_cooldown_label()

    def _update_cooldown_label(self, *_):
        self.cooldown_label.set(f"{self.cooldown_var.get():.1f}s")

    def _save_cfg(self):
        self.cfg["words"] = [list(w) for w in self.words]
        save_config(self.cfg)
        self.hint.set("Config sauvegardée dans grid_config.json")

    # ================= calibration coins =================
    def _start_calib(self):
        self.calib_mode = True
        self.corner_pts = []
        self.hint.set("CALIBRATION : clique les 4 coins sur la vidéo (haut-gauche → haut-droit → bas-droit → bas-gauche)")

    def _on_video_click(self, event):
        """Pose un point de calibration là OÙ L'UTILISATRICE CLIQUE, dans
        n'importe quel ordre (les coins sont ré-orientés automatiquement
        au 4e clic). Conversion : clic canvas -> pixels réels de la frame."""
        if not self.calib_mode:
            return
        if len(self.corner_pts) < 4:
            # tenir compte du letterbox : l'image ne remplit pas tout le canvas
            ox, oy = getattr(self, "_disp_offset", (0, 0))
            canvas_w = max(1, self.canvas.winfo_width())
            canvas_h = max(1, self.canvas.winfo_height())
            if canvas_w < 10 or canvas_h < 10:  # canvas pas encore rendu
                canvas_w, canvas_h = VIDEO_W, VIDEO_H
            # taille affichée de l'image (identique au calcul du letterbox)
            scale = min(canvas_w / self.frame_w, canvas_h / self.frame_h)
            disp_w = max(1, int(self.frame_w * scale))
            disp_h = max(1, int(self.frame_h * scale))
            # clic relatif à l'image affichée -> pixels frame réelle
            fx = (event.x - ox) / scale
            fy = (event.y - oy) / scale
            fx = max(0, min(self.frame_w - 1, fx))
            fy = max(0, min(self.frame_h - 1, fy))
            self.corner_pts.append((int(fx), int(fy)))
        if len(self.corner_pts) == 4:
            # ré-orientation automatique : peu importe l'ordre des clics,
            # on classe les points en haut-gauche / haut-droit / bas-droit / bas-gauche
            pts = np.array(self.corner_pts, dtype=np.float32)
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            ordered = sorted(pts.tolist(), key=lambda p: (p[1] > cy, p[0] > cx))
            # sorted donne : HG, HD, BG, BD -> réordonner en HG, HD, BD, BG
            src = np.array([ordered[0], ordered[1], ordered[3], ordered[2]], dtype=np.float32)
            dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
            self.cfg["homography"] = cv2.getPerspectiveTransform(src, dst).tolist()
            self.mapper = GridMapper(self.cfg["homography"])
            self.calib_status.set("Homographie : calibrée ✔")
            self.calib_mode = False
            self.corner_pts = [(int(p[0]), int(p[1])) for p in src.tolist()]  # points ré-orientés pour l'affichage
            self._save_cfg()
            self.hint.set("Grille calibrée ✔ — tracking actif")

    # ================= mots =================
    def refresh_words(self):
        self.word_list.delete(0, tk.END)
        for word, start, direction in self.words:
            self.word_list.insert(tk.END, f"{word.upper()}  — case {start.upper()}")

    def _add_word(self):
        word = self.word_var.get().strip().upper()
        start = self.start_var.get().strip().upper()
        if not word or not start:
            messagebox.showerror("Champ manquant", "Remplis le mot ET sa case.")
            return
        if word_cell(word, start, int(self.cols_var.get()), int(self.rows_var.get())) is None:
            messagebox.showerror(
                "Case invalide",
                f'La case {start} n\'existe pas dans la grille '
                f'({self.cols_var.get()} colonnes x {self.rows_var.get()} lignes).')
            return
        self.words.append((word, start, "H"))  # direction ignorée : 1 mot = 1 case
        self.word_var.set(""); self.start_var.set("")
        self.refresh_words()
        self._save_cfg()

    def _remove_word(self):
        sel = self.word_list.curselection()
        if sel:
            self.words.pop(sel[0])
            self.refresh_words()
            self._save_cfg()

    # ================= explosion =================
    def _preview_explosion(self):
        test = np.zeros((400, 400, 3), np.uint8)
        ExplosionAnimation(self.sprites, (200, 200),
                           float(self.scale_var.get()), int(self.fps_var.get())).draw(test)
        cv2.imwrite(".preview_explosion.png", test)
        self.hint.set("Aperçu explosion dessiné (à la taille/vitesse choisies)")

    # ================= collisions =================
    def _reset_events(self):
        self.collision_events.clear()
        self._refresh_events()

    def _refresh_events(self):
        self.events_list.delete(0, tk.END)
        for cell, ts in reversed(self.collision_events[-40:]):
            self.events_list.insert(tk.END, f"{cell}  {ts}")
        self.count_var.set(f"{self.collision_count} collision{'s' if self.collision_count > 1 else ''}")

    def _on_collision(self, cell, word, mid_px):
        """Explosion à chaque collision, mais la liste ne garde QUE les
        collisions tombées sur les cases d'un mot (le mot en question)."""
        ts = datetime.datetime.now().isoformat(timespec="milliseconds")
        self.collision_count += 1
        self.explosions.append(ExplosionAnimation(self.sprites, mid_px,
                                                  self.cfg["explosion_scale"],
                                                  self.cfg["explosion_fps"]))
        if word:
            self.collision_events.append((f"{word}  ({cell})", ts[11:19]))
            self.root.after(0, self._refresh_events)
        if self.log:
            self.log.writerow([ts, "Roumba A", "Roumba B", cell, word or ""])
            self.log_file.flush()
        print(f"💥 BOOM n°{self.collision_count} — case {cell}" + (f" (mot: {word})" if word else ""))

    # ================= sorties =================
    def _open_outputs(self):
        Path(self.cfg["record_dir"]).mkdir(exist_ok=True)
        events_path = Path(self.cfg["events_csv"])
        log_exists = events_path.exists()
        self.log_file = open(events_path, "a", newline="")
        self.log = csv.writer(self.log_file)
        if not log_exists:
            self.log.writerow(["timestamp", "robot_a", "robot_b", "cellule", "mot"])

    def _probe_cameras(self):
        """Sonde les index 0..3 en arrière-plan pour lister les sources disponibles."""
        import sys
        found = []
        for i in range(4):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if sys.platform == "win32" \
                    else cv2.VideoCapture(i)
                ok, frame = cap.read()
                cap.release()
                if ok and frame is not None:
                    found.append(i)
            except Exception:
                continue
        return found

    def _refresh_cam_list(self):
        """Sonde les caméras dans un thread et met à jour la liste déroulante."""
        self.cam_combo['values'] = ['(scan en cours...)']
        self.cam_combo.set('(scan en cours...)')
        def work():
            cams = self._probe_cameras()
            labels = [f"Caméra {i}" + ("  (actuelle)" if i == self.cfg["cam_index"] else "")
                      for i in cams]
            self.root.after(0, lambda: self._update_cam_combo(cams, labels))
        threading.Thread(target=work, daemon=True).start()

    def _update_cam_combo(self, cams, labels):
        self._available_cams = cams
        self.cam_combo['values'] = labels or ['(aucune caméra trouvée)']
        if self.cfg["cam_index"] in cams:
            idx = cams.index(self.cfg["cam_index"])
            self.cam_combo.set(labels[idx])
        elif labels:
            self.cam_combo.set(labels[0])
        else:
            self.cam_combo.set('(aucune caméra trouvée)')

    def _switch_camera(self, event=None):
        """Changement de source vidéo À CHAUD : arrête le thread caméra,
        rouvre la nouvelle source, sans redémarrer l'application."""
        sel = self.cam_combo.get()
        if not sel or sel.startswith('('):
            return
        try:
            new_index = int(sel.split()[1])
        except (ValueError, IndexError):
            return
        if new_index == self.cfg["cam_index"] and self.cap is not None:
            return
        self.cfg["cam_index"] = new_index
        self._save_cfg()
        # arrêt propre du thread caméra actuel
        self.camera_running = False
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        self.latest_frame = None
        self.camera_error = None
        self.hint.set(f"Ouverture de la caméra {new_index}...")
        # relance
        self.camera_running = True
        threading.Thread(target=self._open_camera, daemon=True).start()

    def _open_camera(self):
        """Thread séparé : ouvre la caméra sans bloquer l'interface.
        Sous Windows, cv2.VideoCapture peut mettre plusieurs secondes
        (voire échouer silencieusement) — d'où le backend DirectShow."""
        try:
            import sys
            cap = cv2.VideoCapture(self.cfg["cam_index"], cv2.CAP_DSHOW) \
                if sys.platform == "win32" else cv2.VideoCapture(self.cfg["cam_index"])
            ok, frame = cap.read()
            if not ok or frame is None:
                # 2e tentative sans backend spécifique (webcam virtuelle, etc.)
                cap.release()
                cap = cv2.VideoCapture(self.cfg["cam_index"])
                ok, frame = cap.read()
            if not ok or frame is None:
                self.camera_error = ("Caméra introuvable. Vérifie qu'elle est branchée et "
                                      "autorisée (Confidentialité Windows > Caméra).")
                self.hint.set(self.camera_error)
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg["cam_width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg["cam_height"])
            # relire une frame APRÈS le changement de résolution : la première
            # frame était dans la résolution par défaut, pas celle demandée
            ok2, frame2 = cap.read()
            if ok2 and frame2 is not None:
                frame = frame2
            self.frame_h, self.frame_w = frame.shape[:2]
            with self.lock:
                self.cap = cap
                self.latest_frame = frame
            self.hint.set(f"Caméra {self.cfg['cam_index']} prête ✔")
            while self.camera_running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                # la résolution peut changer en cours de route : la mesurer en continu
                if frame.shape[1] != self.frame_w or frame.shape[0] != self.frame_h:
                    self.frame_h, self.frame_w = frame.shape[:2]
                with self.lock:
                    self.latest_frame = frame
            cap.release()
        except Exception as e:
            self.camera_error = f"Erreur caméra : {e}"
            self.hint.set(self.camera_error)

    # ================= boucle d'affichage =================
    def _display_loop(self):
        if not self.running:
            return
        frame = getattr(self, "latest_frame", None)
        if frame is None or not isinstance(frame, np.ndarray) or frame.shape[0] < 2 or frame.shape[1] < 2:
            # caméra pas encore prête (ou en erreur) : écran d'attente
            waiting = np.full((VIDEO_H, VIDEO_W, 3), (30, 30, 30), np.uint8)
            msg = self.camera_error or "Ouverture de la caméra..."
            cv2.putText(waiting, msg[:60], (20, VIDEO_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            ok_buf, buf = cv2.imencode(".ppm", waiting)
            if ok_buf:
                self._photo = tk.PhotoImage(data=buf.tobytes())
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        else:
            if self.writer is None and self.cfg["record"]:
                self._sync_writer()  # caméra prête -> on connaît la taille des frames
            view = self._process(frame)
            with self.lock:
                if self.writer:
                    self.writer.write(view)
            # affichage SANS déformation (letterbox) : la frame garde son ratio,
            # des bandes noires comblent le reste du canvas
            canvas_w = max(1, self.canvas.winfo_width())
            canvas_h = max(1, self.canvas.winfo_height())
            if canvas_w < 10 or canvas_h < 10:  # canvas pas encore rendu
                canvas_w, canvas_h = VIDEO_W, VIDEO_H
            scale = min(canvas_w / view.shape[1], canvas_h / view.shape[0])
            disp_w = max(1, int(view.shape[1] * scale))
            disp_h = max(1, int(view.shape[0] * scale))
            disp = cv2.resize(view, (disp_w, disp_h))
            self._disp_offset = ((canvas_w - disp_w) // 2, (canvas_h - disp_h) // 2)
            ok_buf, buf = cv2.imencode(".ppm", disp)
            if ok_buf:
                self._photo = tk.PhotoImage(data=buf.tobytes())
                self.canvas.delete("all")
                self.canvas.create_image(self._disp_offset[0], self._disp_offset[1],
                                         anchor="nw", image=self._photo)
        self.root.after(10, self._display_loop)

    def _process(self, frame):
        view = frame.copy()
        # réglage IMAGE (saturation/luminosité/contraste) appliqué en direct
        if (self.img_sat_gain != 1.0 or self.img_val_gain != 1.0
                or self.img_contrast != 1.0):
            hsv = cv2.cvtColor(view, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.img_sat_gain, 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * self.img_val_gain, 0, 255)
            view = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            if self.img_contrast != 1.0:
                view = np.clip((view.astype(np.float32) - 128) * self.img_contrast + 128,
                               0, 255).astype(np.uint8)
        # grille + mots dessinés PAR-DESSUS la vidéo (calibrés sur les 4 coins)
        if self.mapper:
            cols, rows = self.cfg["cols"], self.cfg["rows"]
            if self.show_grid:
                draw_grid_on_view(view, self.mapper, cols, rows)
            if self.show_words:
                draw_words_on_view(view, self.mapper, cols, rows, self.words)
        # détection PAR COULEUR (autocollants rouge = A, bleu = B)
        color_detect.CFG.update(self.cfg)
        detections = color_detect.detect(frame)
        posA = detections.get(ROBOT_A["name"])
        posB = detections.get(ROBOT_B["name"])
        for robot, pos in [(ROBOT_A, posA), (ROBOT_B, posB)]:
            if pos:
                # trajectoire : on garde le point et on dessine la trace DERRIÈRE le robot
                self.trails[robot["name"]].append((int(pos[0]), int(pos[1])))
                pts = list(self.trails[robot["name"]])
                for a, b in zip(pts, pts[1:]):
                    cv2.line(view, a, b, robot["trace"], 2)
                cv2.circle(view, (int(pos[0]), int(pos[1])), max(6, int(pos[2])),
                           (255, 255, 255), 1)
                cv2.circle(view, (int(pos[0]), int(pos[1])), 6, (255, 255, 255), -1)
                cv2.putText(view, robot["name"], (int(pos[0]) + 12, int(pos[1]) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, robot["trace"], 2)

        if posA and posB and self.mapper:
            cols, rows = self.cfg["cols"], self.cfg["rows"]
            gA = self.mapper.to_grid((posA[0], posA[1]), cols, rows)
            gB = self.mapper.to_grid((posB[0], posB[1]), cols, rows)
            dist = np.hypot(gA[0] - gB[0], gA[1] - gB[1])
            now = time.time()
            if (dist < self.cfg["collision_dist_cells"]
                    and now - self.last_collision_t > self.cfg["collision_cooldown_s"]):
                self.last_collision_t = now
                mid = ((gA[0] + gB[0]) / 2, (gA[1] + gB[1]) / 2)
                cell = cell_name(mid[0], mid[1], cols, rows)
                word = word_for_cell(cell, self.words, cols, rows)
                mid_px = ((posA[0] + posB[0]) / 2, (posA[1] + posB[1]) / 2)
                self._on_collision(cell, word, mid_px)

        # explosions animées
        self.explosions = [e for e in self.explosions if e.alive()]
        for e in self.explosions:
            e.draw(view)

        cv2.putText(view, f"Collisions: {self.collision_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # coins de calibration en cours
        for p in self.corner_pts:
            cv2.circle(view, p, 8, (0, 255, 255), 2)
        if self.calib_mode:
            cv2.putText(view, f"CLIC {4 - len(self.corner_pts)}/4 coins",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return view

    # ================= writer =================
    def _sync_writer(self, start=False):
        """Le writer vidéo attend que la caméra fournisse la taille réelle des frames."""
        if not self.cfg["record"] and self.writer is not None:
            with self.lock:
                self.writer.release()
                self.writer = None
            return
        if self.cfg["record"] and self.writer is None and self.cap is not None:
            with self.lock:
                stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"{self.cfg['record_dir']}/session_{stamp}.mp4"
                self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                              25.0, (self.frame_w, self.frame_h))
                self.hint.set(f"Enregistrement : {path}")

    # ================= fermeture =================
    def _on_close(self):
        self.running = False
        self.camera_running = False
        time.sleep(0.1)
        self._save_cfg()
        if self.writer:
            self.writer.release()
        self.log_file.close()
        self.root.destroy()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=None,
                    help="index de la caméra (0, 1, 2... ; défaut: cam_index de la config)")
    args = ap.parse_args()
    if args.cam is not None:
        import config as _config
        _cfg = _config.load_config()
        _cfg["cam_index"] = args.cam
        _config.save_config(_cfg)
    root = tk.Tk()
    App(root)
    root.mainloop()
