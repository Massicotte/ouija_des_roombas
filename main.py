"""
OUIJA DES ROOMBAS — version KISS (Keep It Simple, Stupid)
Un fichier. Une boucle. La caméra est lue à chaque frame, tout le reste
(calibration, mots, réglages) est relu depuis des variables simples.

    python main.py            # lance tout

Le principe : à chaque frame ->
  1. lire la caméra
  2. détecter les 2 taches de couleur (rouge = A, bleu = B)
  3. dessiner la grille + mots + traces + explosion
  4. si collision : enregistrer le mot touché
  5. afficher dans Tkinter
"""

import csv
import re
import time
import datetime
import json
import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
HERE = Path(__file__).parent
CONFIG_FILE = HERE / "grid_config.json"

# ---------- réglages par défaut (tous éditables dans l'interface) ----------
DEFAULTS = {
    "cam_index": 0,
    "cols": 8,
    "rows": 8,
    "words": [],                # [[mot, case], ...]  (1 mot = 1 case)
    "homography": None,         # 3x3, calculée par les 4 clics
    # détection couleur
    "sat_min": 120,
    "lumi_min": 70,
    "aire_min": 500,
    # image affichée
    "img_sat": 1.0,
    "img_lum": 1.0,
    # explosion
    "boom_scale": 1.0,
    # collision
    "cooldown": 2.0,
    "dist_cells": 0.5,
    # traces
    "trail_len": 2000,
}

# couleurs des taches à détecter (plages de teinte HSV, 0-180)
RED_RANGES = [(0, 10), (170, 180)]
BLUE_RANGES = [(100, 130)]


def load_cfg():
    cfg = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        cfg.update(json.loads(CONFIG_FILE.read_text()))
    # accepter les mots des ANCIENS formats : [mot, case, direction] -> [mot, case]
    cfg["words"] = [[w[0], w[1]] for w in cfg.get("words", []) if len(w) >= 2]
    return cfg


def save_cfg(cfg):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def cell_name(col, row, cols, rows):
    """(col, row) -> 'B3' ; hors grille -> None."""
    if col < 0 or row < 0 or col >= cols or row >= rows:
        return None
    return f"{LETTERS[int(col)]}{int(row) + 1}"


def parse_cell(name):
    """'B3' -> (1, 2) ; invalide -> None."""
    m = re.match(r"^([A-Z])(\d+)$", str(name).strip().upper())
    if not m:
        return None
    return ord(m.group(1)) - ord("A"), int(m.group(2)) - 1


# ============================================================
# caméra : ouverture avec fallback de backends (Windows capricieux)
# ============================================================
def open_camera(index):
    for backend in ([cv2.CAP_DSHOW, cv2.CAP_MSMF] if os.name == "nt" else []):
        try:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    return cap
            cap.release()
        except Exception:
            pass
    cap = cv2.VideoCapture(index)
    return cap if cap.isOpened() else None


# ============================================================
# détection : la plus grosse tache de la couleur demandée
# ============================================================
KERNEL = np.ones((5, 5), np.uint8)


def find_blob(frame, ranges, sat_min, lumi_min, aire_min):
    """Renvoie (cx, cy, rayon) de la plus grosse tache, ou None."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (7, 7), 0), cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for h_min, h_max in ranges:
        mask |= cv2.inRange(hsv, (h_min, sat_min, lumi_min), (h_max, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= aire_min]
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    m = cv2.moments(c)
    if m["m00"] == 0:
        return None
    return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]), int(np.sqrt(cv2.contourArea(c) / np.pi))


# ============================================================
# explosion : les frames du sprite, centrées sur le point de collision
# ============================================================
def load_sprites():
    sheet = cv2.imread(str(HERE / "assets/explosion_sprites.png"), cv2.IMREAD_UNCHANGED)
    if sheet is None:
        return []
    h = sheet.shape[0]
    n = sheet.shape[1] // h
    return [sheet[:, i * h:(i + 1) * h] for i in range(n)]


def draw_sprite(img, sprite, center, scale):
    size = max(8, int(sprite.shape[0] * scale))
    spr = cv2.resize(sprite, (size, size), interpolation=cv2.INTER_NEAREST)
    x, y = center[0] - size // 2, center[1] - size // 2
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + size), min(img.shape[0], y + size)
    if x1 <= x0 or y1 <= y0:
        return
    sub = spr[y0 - y:y0 - y + (y1 - y0), x0 - x:x0 - x + (x1 - x0)]
    alpha = sub[:, :, 3:4].astype(np.float32) / 255.0
    img[y0:y1, x0:x1] = (sub[:, :, :3] * alpha + img[y0:y1, x0:x1] * (1 - alpha)).astype(np.uint8)


# ============================================================
# grille : homographie pixels <-> cases
# ============================================================
class Grid:
    def __init__(self, corners, cols, rows):
        src = np.array(corners, np.float32)
        dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
        self.H = cv2.getPerspectiveTransform(src, dst)
        self.Hinv = np.linalg.inv(self.H)
        self.cols, self.rows = cols, rows

    def to_cell(self, pt):
        """pixel -> (col, row) continues en cases."""
        g = cv2.perspectiveTransform(np.array([[pt]], np.float32), self.H)[0][0]
        return g[0] * self.cols, g[1] * self.rows

    def to_px(self, u, v):
        """[0..1] normalisé -> pixel caméra."""
        q = cv2.perspectiveTransform(np.array([[[u, v]]], np.float32), self.Hinv)[0][0]
        return float(q[0]), float(q[1])

    def draw(self, view):
        c, r = self.cols, self.rows
        for i in range(c + 1):
            p0, p1 = self.to_px(i / c, 0), self.to_px(i / c, 1)
            cv2.line(view, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                     (255, 255, 255), 1, cv2.LINE_AA)
        for j in range(r + 1):
            p0, p1 = self.to_px(0, j / r), self.to_px(1, j / r)
            cv2.line(view, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                     (255, 255, 255), 1, cv2.LINE_AA)
        # lettres au-dessus des colonnes, chiffres à gauche des lignes (HORS grille)
        for i in range(c):
            px = self.to_px((i + 0.5) / c, -0.22 / r)
            cv2.putText(view, LETTERS[i], (int(px[0]) - 8, int(px[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        for j in range(r):
            px = self.to_px(-0.22 / c, (j + 0.5) / r)
            cv2.putText(view, str(j + 1), (int(px[0]) - 6, int(px[1]) + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def draw_word(self, view, word, col, row):
        """Dessine le mot ENTIER dans SA case, police adaptée à la case."""
        tl = self.to_px(col / self.cols, row / self.rows)
        br = self.to_px((col + 1) / self.cols, (row + 1) / self.rows)
        cw, ch = abs(br[0] - tl[0]), abs(br[1] - tl[1])
        center = self.to_px((col + 0.5) / self.cols, (row + 0.5) / self.rows)
        text = str(word).upper()
        scale = 0.1
        while scale < 5.0:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale + 0.05, 1)
            if tw > cw * 0.92 or th > ch * 0.92:
                break
            scale += 0.05
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.putText(view, text, (int(center[0] - tw / 2), int(center[1] + th / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (120, 220, 255), 1, cv2.LINE_AA)


# ============================================================
# L'APPLICATION : une fenêtre, une boucle
# ============================================================
class App:
    def __init__(self, root):
        self.cfg = load_cfg()
        self.root = root
        root.title("Ouija des Rouombas 🤖💥")
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.sprites = load_sprites()
        self.booms = []            # [(t0, x, y)]
        self.trails = [deque(maxlen=self.cfg["trail_len"]), deque(maxlen=self.cfg["trail_len"])]
        self.corners = []          # clics de calibration
        self.events = []           # [(texte, heure)]
        self.last_boom = -1e9
        self.n_booms = 0
        self.cap = None
        self.frame = None
        self.photo = None
        self.calibrating = False
        self.running = True

        self._build_ui()
        self._open_camera()
        self._tick()

    # ---------------- interface : vidéo à gauche, réglages à droite ----------------
    def _build_ui(self):
        left = ttk.Frame(self.root, padding=5)
        left.pack(side=tk.LEFT)
        self.canvas = tk.Canvas(left, width=960, height=540, bg="black")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._click)
        self.hint = tk.StringVar(value="Clique '📍 Calibrer' puis les 4 coins de la grille (dans l'ordre que tu veux).")
        ttk.Label(left, textvariable=self.hint).pack(anchor="w")

        # panneau droit DÉFILABLE (sinon les derniers réglages sont invisibles)
        wrap = ttk.Frame(self.root, padding=(0, 5, 5, 5))
        wrap.pack(side=tk.RIGHT, fill=tk.Y)
        self.panel = tk.Canvas(wrap, width=310, highlightthickness=0)
        bar = ttk.Scrollbar(wrap, orient="vertical", command=self.panel.yview)
        self.panel.configure(yscrollcommand=bar.set)
        bar.pack(side=tk.RIGHT, fill=tk.Y)
        self.panel.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        inner = ttk.Frame(self.panel, padding=5)
        win = self.panel.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: self.panel.configure(
            scrollregion=self.panel.bbox("all")))
        self.panel.bind("<Configure>", lambda e: self.panel.itemconfigure(
            win, width=self.panel.winfo_width()))
        self.panel.bind_all("<MouseWheel>",
                            lambda e: self.panel.yview_scroll(int(-e.delta / 120), "units"))
        self._build_controls(inner)

    def _build_controls(self, p):
        # --- caméra ---
        g = ttk.LabelFrame(p, text="Caméra", padding=8)
        g.pack(fill=tk.X)
        ttk.Button(g, text="Réouvrir la caméra", command=self._reopen_camera).pack(fill=tk.X)
        row = ttk.Frame(g); row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(row, text="Index:").pack(side=tk.LEFT)
        self.cam_var = tk.IntVar(value=self.cfg["cam_index"])
        ttk.Spinbox(row, from_=0, to=5, width=3, textvariable=self.cam_var).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="OK", width=4, command=self._reopen_camera).pack(side=tk.LEFT)

        # --- grille ---
        g = ttk.LabelFrame(p, text="Grille", padding=8)
        g.pack(fill=tk.X, pady=6)
        row = ttk.Frame(g); row.pack(fill=tk.X)
        ttk.Label(row, text="Colonnes:").pack(side=tk.LEFT)
        self.cols_var = tk.IntVar(value=self.cfg["cols"])
        ttk.Spinbox(row, from_=2, to=26, width=3, textvariable=self.cols_var).pack(side=tk.LEFT, padx=4)
        ttk.Label(row, text="Lignes:").pack(side=tk.LEFT)
        self.rows_var = tk.IntVar(value=self.cfg["rows"])
        ttk.Spinbox(row, from_=2, to=30, width=3, textvariable=self.rows_var).pack(side=tk.LEFT, padx=4)
        ttk.Button(g, text="📍 Calibrer les 4 coins", command=self._start_calib).pack(fill=tk.X, pady=(6, 0))

        # --- mots ---
        w = ttk.LabelFrame(p, text="Mots (1 mot = 1 case)", padding=8)
        w.pack(fill=tk.X, pady=6)
        row = ttk.Frame(w); row.pack(fill=tk.X)
        ttk.Label(row, text="Mot:").pack(side=tk.LEFT)
        self.word_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.word_var, width=10).pack(side=tk.LEFT, padx=3)
        ttk.Label(row, text="Case:").pack(side=tk.LEFT)
        self.cell_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.cell_var, width=4).pack(side=tk.LEFT, padx=3)
        ttk.Button(row, text="+", width=3, command=self._add_word).pack(side=tk.LEFT)
        self.words_list = tk.Listbox(w, height=4)
        self.words_list.pack(fill=tk.X, pady=4)
        ttk.Button(w, text="Supprimer le mot sélectionné", command=self._del_word).pack(fill=tk.X)

        # --- détection couleur ---
        d = ttk.LabelFrame(p, text="Détection couleur 🎨", padding=8)
        d.pack(fill=tk.X, pady=6)
        self._slider(d, "Saturation min", "sat_min", 0, 255)
        self._slider(d, "Luminosité min", "lumi_min", 0, 255)
        self._slider(d, "Aire min (px)", "aire_min", 10, 20000)
        ttk.Button(d, text="🧹 Effacer les traces", command=lambda: [t.clear() for t in self.trails]).pack(fill=tk.X, pady=(4, 0))

        # --- image ---
        i = ttk.LabelFrame(p, text="Image 🖼", padding=8)
        i.pack(fill=tk.X, pady=6)
        self._slider(i, "Saturation", "img_sat", 0.0, 3.0)
        self._slider(i, "Luminosité", "img_lum", 0.2, 2.5)
        self._slider(i, "Taille explosion", "boom_scale", 0.2, 5.0)
        self._slider(i, "Délai entre collisions (s)", "cooldown", 0.5, 30.0)

        # --- collisions ---
        c = ttk.LabelFrame(p, text="Collisions 💥", padding=8)
        c.pack(fill=tk.X, pady=6)
        self.count_var = tk.StringVar(value="0 collision")
        ttk.Label(c, textvariable=self.count_var).pack(anchor="w")
        ttk.Button(c, text="♻ Reset de la liste", command=self._reset_events).pack(fill=tk.X, pady=3)
        self.events_list = tk.Listbox(c, height=6)
        self.events_list.pack(fill=tk.X)

    def _slider(self, parent, label, key, lo, hi):
        """Slider simple : écrit directement dans self.cfg, sauvegardé à la volée."""
        row = ttk.Frame(parent); row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label).pack(side=tk.LEFT)
        var = tk.DoubleVar(value=float(self.cfg[key]))
        lbl = tk.StringVar(value=self._fmt(var.get()))
        def set_val(v):
            self.cfg[key] = float(var.get())
            lbl.set(self._fmt(self.cfg[key]))
        ttk.Scale(row, from_=lo, to=hi, variable=var, command=set_val).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Label(row, textvariable=lbl, width=6).pack(side=tk.LEFT)
        var.trace_add("write", lambda *_: save_cfg(self.cfg))

    @staticmethod
    def _fmt(v):
        return str(int(v)) if v == int(v) else f"{v:.1f}"

    # ---------------- caméra ----------------
    def _open_camera(self):
        index = int(self.cam_var.get()) if hasattr(self, "cam_var") else self.cfg["cam_index"]
        self.cfg["cam_index"] = index
        if self.cap:
            self.cap.release()
        self.cap = open_camera(index)
        self.hint.set("Caméra prête ✔" if self.cap else "❌ Caméra introuvable (vérifie branchement / index)")

    def _reopen_camera(self):
        self._open_camera()
        save_cfg(self.cfg)

    # ---------------- calibration ----------------
    def _start_calib(self):
        self.calibrating = True
        self.corners = []
        self.hint.set("Clique les 4 coins de la grille sur la vidéo — dans l'ordre que tu veux.")

    def _click(self, event):
        """Clic sur la vidéo : pose un coin de calibration, converti via le letterbox."""
        if not self.calibrating or self.frame is None:
            return
        fh, fw = self.frame.shape[:2]
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10:
            cw, ch = 960, 540
        scale = min(cw / fw, ch / fh)
        dw, dh = int(fw * scale), int(fh * scale)
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        fx = max(0, min(fw - 1, int((event.x - ox) / scale)))
        fy = max(0, min(fh - 1, int((event.y - oy) / scale)))
        self.corners.append((fx, fy))
        self.hint.set(f"Coin {len(self.corners)}/4 posé.")
        if len(self.corners) == 4:
            # ré-orientation automatique : peu importe l'ordre des clics
            pts = np.array(self.corners, np.float32)
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            ordered = sorted(pts.tolist(), key=lambda q: (q[1] > cy, q[0] > cx))
            src = np.array([ordered[0], ordered[1], ordered[3], ordered[2]], np.float32)
            self.corners = [tuple(p) for p in src.tolist()]
            self.cfg["corners"] = self.corners
            self.calibrating = False
            save_cfg(self.cfg)
            self.hint.set("Grille calibrée ✔")

    # ---------------- mots ----------------
    def _refresh_words(self):
        self.words_list.delete(0, tk.END)
        for word, cell in self.cfg["words"]:
            self.words_list.insert(tk.END, f"{word}  — case {cell}")

    def _add_word(self):
        word = self.word_var.get().strip().upper()
        cell = self.cell_var.get().strip().upper()
        parsed = parse_cell(cell)
        if not word or not parsed:
            messagebox.showerror("Champ manquant", "Remplis le mot ET une case valide (ex: B3).")
            return
        if parsed[0] >= int(self.cols_var.get()) or parsed[1] >= int(self.rows_var.get()):
            messagebox.showerror("Case invalide", f"{cell} n'existe pas dans la grille.")
            return
        self.cfg["words"].append([word, cell])
        self.word_var.set(""); self.cell_var.set("")
        save_cfg(self.cfg)
        self._refresh_words()

    def _del_word(self):
        sel = self.words_list.curselection()
        if sel:
            self.cfg["words"].pop(sel[0])
            save_cfg(self.cfg)
            self._refresh_words()

    def _word_at(self, col, row):
        """Le mot placé dans CETTE case, ou None."""
        for word, cell in self.cfg["words"]:
            if parse_cell(cell) == (col, row):
                return word
        return None

    # ---------------- collisions ----------------
    def _boom(self, x, y, cell, word):
        self.booms.append((time.time(), int(x), int(y)))
        self.n_booms += 1
        if word:
            self.events.append((f"{word} ({cell})", datetime.datetime.now().strftime("%H:%M:%S")))
            self.events_list.delete(0, tk.END)
            for txt, h in reversed(self.events[-40:]):
                self.events_list.insert(tk.END, f"{txt}  {h}")
        self.count_var.set(f"{self.n_booms} collision{'s' if self.n_booms > 1 else ''}")
        with open(HERE / "collisions.csv", "a", newline="") as f:
            csv.writer(f).writerow([datetime.datetime.now().isoformat(timespec="milliseconds"),
                                    cell, word or ""])

    def _reset_events(self):
        self.events.clear()
        self.events_list.delete(0, tk.END)

    # ---------------- la boucle : tout se passe ici ----------------
    def _tick(self):
        if not self.running:
            return
        if self.cap is None or not self.cap.isOpened():
            self.root.after(300, self._tick)
            return
        ok, frame = self.cap.read()
        if not ok:
            self.root.after(50, self._tick)
            return
        self.frame = frame
        cfg = self.cfg
        cols, rows = int(self.cols_var.get()), int(self.rows_var.get())

        # 1. détection des 2 taches (rouge = A, bleu = B)
        posA = find_blob(frame, RED_RANGES, cfg["sat_min"], cfg["lumi_min"], cfg["aire_min"])
        posB = find_blob(frame, BLUE_RANGES, cfg["sat_min"], cfg["lumi_min"], cfg["aire_min"])

        # 2. réglage image (saturation / luminosité)
        view = frame.copy()
        if cfg["img_sat"] != 1.0 or cfg["img_lum"] != 1.0:
            hsv = cv2.cvtColor(view, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * cfg["img_sat"], 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * cfg["img_lum"], 0, 255)
            view = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 3. grille + mots
        if cfg.get("corners"):
            grid = Grid(np.array(cfg["corners"], np.float32), cols, rows)
            grid.draw(view)
            for word, cell in cfg["words"]:
                parsed = parse_cell(cell)
                if parsed and parsed[0] < cols and parsed[1] < rows:
                    grid.draw_word(view, word, parsed[0], parsed[1])

        # 4. traces + marqueurs
        for i, (pos, color) in enumerate([(posA, (0, 0, 255)), (posB, (255, 0, 0))]):
            if pos:
                self.trails[i].append((pos[0], pos[1]))
                pts = list(self.trails[i])
                for a, b in zip(pts, pts[1:]):
                    cv2.line(view, a, b, color, 2)
                cv2.circle(view, (pos[0], pos[1]), 6, (255, 255, 255), -1)
                cv2.putText(view, "A" if i == 0 else "B", (pos[0] + 12, pos[1] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 5. collision
        if posA and posB and cfg.get("corners"):
            grid = Grid(np.array(cfg["corners"], np.float32), cols, rows)
            ca, ra = grid.to_cell((posA[0], posA[1]))
            cb, rb = grid.to_cell((posB[0], posB[1]))
            dist = float(np.hypot(ca - cb, ra - rb))
            now = time.time()
            if dist < cfg["dist_cells"] and now - self.last_boom > cfg["cooldown"]:
                self.last_boom = now
                mx, my = (posA[0] + posB[0]) / 2, (posA[1] + posB[1]) / 2
                cc, rc = int((ca + cb) / 2), int((ra + rb) / 2)
                cell = cell_name(cc, rc, cols, rows)
                self._boom(mx, my, cell or "?", self._word_at(cc, rc) if cell else None)

        # 6. explosions
        self.booms = [b for b in self.booms if time.time() - b[0] < 1.0]
        for t0, bx, by in self.booms:
            idx = int((time.time() - t0) * 15)
            if idx < len(self.sprites):
                draw_sprite(view, self.sprites[idx], (bx, by), cfg["boom_scale"])

        cv2.putText(view, f"Collisions: {self.n_booms}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # coins de calibration en cours
        for p in self.corners:
            cv2.circle(view, (int(p[0]), int(p[1])), 8, (0, 255, 255), 2)

        # 7. affichage letterbox (sans déformation)
        fh, fw = view.shape[:2]
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10:
            cw, ch = 960, 540
        scale = min(cw / fw, ch / fh)
        disp = cv2.resize(view, (max(1, int(fw * scale)), max(1, int(fh * scale))))
        ok2, buf = cv2.imencode(".ppm", disp)
        if ok2:
            self.photo = tk.PhotoImage(data=buf.tobytes())
            self.canvas.delete("all")
            ox, oy = (cw - disp.shape[1]) // 2, (ch - disp.shape[0]) // 2
            self.canvas.create_image(ox, oy, anchor="nw", image=self.photo)

        self.root.after(10, self._tick)

    def close(self):
        self.running = False
        save_cfg(self.cfg)
        if self.cap:
            self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
