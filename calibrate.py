"""
OUTIL DE CALIBRATION 🔧
Fenêtre séparée pour tout régler SANS toucher au code :
  - nombre de colonnes / lignes de la grille
  - les mots placés dans la grille (+ leur case de départ et direction)
  - les 4 coins de la grille (clics sur un aperçu caméra) -> homographie

Usage:
    python calibrate.py
"""

import json
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np

from config import load_config, save_config, CONFIG_FILE

DIRECTIONS = {"H": "Horizontal (vers la droite)", "V": "Vertical (vers le bas)"}


class CalibrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ouija des Roumbas — Calibration de la grille")
        self.cfg = load_config()
        self.corner_pts = []
        if self.cfg.get("homography"):
            self.root.after(300, self._try_show_existing)

        main = ttk.Frame(root, padding=15)
        main.pack(fill=tk.BOTH, expand=True)

        # --- Grille ---
        grid_frame = ttk.LabelFrame(main, text="Grille", padding=10)
        grid_frame.pack(fill=tk.X)
        ttk.Label(grid_frame, text="Colonnes (lettres A..):").grid(row=0, column=0, sticky="w")
        self.cols_var = tk.IntVar(value=self.cfg["cols"])
        ttk.Spinbox(grid_frame, from_=2, to=26, width=5, textvariable=self.cols_var).grid(row=0, column=1, sticky="w", padx=10)
        ttk.Label(grid_frame, text="Lignes (chiffres 1..):").grid(row=0, column=2, sticky="w")
        self.rows_var = tk.IntVar(value=self.cfg["rows"])
        ttk.Spinbox(grid_frame, from_=2, to=30, width=5, textvariable=self.rows_var).grid(row=0, column=3, sticky="w", padx=10)

        # --- Calibration caméra ---
        cam_frame = ttk.LabelFrame(main, text="Coins de la grille (caméra)", padding=10)
        cam_frame.pack(fill=tk.X, pady=8)
        self.calib_status = tk.StringVar(
            value="Homographie : " + ("déjà calibrée ✔" if self.cfg.get("homography") else "non calibrée ✘"))
        ttk.Label(cam_frame, textvariable=self.calib_status).pack(anchor="w")
        ttk.Button(cam_frame, text="📷 Calibrer les 4 coins (fenêtre caméra)",
                   command=self.calibrate_corners).pack(pady=5)

        # --- Mots ---
        words_frame = ttk.LabelFrame(main, text="Mots placés dans la grille", padding=10)
        words_frame.pack(fill=tk.BOTH, expand=True)
        row = ttk.Frame(words_frame)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Mot:").pack(side=tk.LEFT)
        self.word_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.word_var, width=15).pack(side=tk.LEFT, padx=6)
        ttk.Label(row, text="Case de départ (ex: B3):").pack(side=tk.LEFT)
        self.start_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.start_var, width=5).pack(side=tk.LEFT, padx=6)
        ttk.Label(row, text="Direction:").pack(side=tk.LEFT)
        self.dir_var = tk.StringVar(value="H")
        ttk.Combobox(row, textvariable=self.dir_var, width=25, state="readonly",
                     values=list(DIRECTIONS.values())).pack(side=tk.LEFT, padx=6)
        ttk.Button(row, text="Ajouter", command=self.add_word).pack(side=tk.LEFT, padx=6)

        self.word_list = tk.Listbox(words_frame, height=6)
        self.word_list.pack(fill=tk.X, pady=6)
        ttk.Button(words_frame, text="Supprimer le mot sélectionné",
                   command=self.remove_word).pack(anchor="w")

        # --- Sauvegarde ---
        ttk.Button(main, text="💾 Sauvegarder et fermer", command=self.save_and_close).pack(pady=10)

        self.words = [tuple(w) for w in self.cfg["words"]]
        self.refresh_words()

    # ---------------- words ----------------
    def refresh_words(self):
        self.word_list.delete(0, tk.END)
        for word, start, direction in self.words:
            d = DIRECTIONS.get(direction, direction)
            self.word_list.insert(tk.END, f"{word.upper()}  — départ {start.upper()}  — {d}")

    def add_word(self):
        word = self.word_var.get().strip().upper()
        start = self.start_var.get().strip().upper()
        direction = "H" if self.dir_var.get().startswith("Horizontal") else "V"
        if not word or not start:
            messagebox.showerror("Champ manquant", "Remplis le mot ET sa case de départ.")
            return
        self.words.append((word, start, direction))
        self.word_var.set("")
        self.start_var.set("")
        self.refresh_words()

    def remove_word(self):
        sel = self.word_list.curselection()
        if sel:
            self.words.pop(sel[0])
            self.refresh_words()

    # ---------------- corners ----------------
    def calibrate_corners(self):
        cap = cv2.VideoCapture(self.cfg["cam_index"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg["cam_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg["cam_height"])
        ok, frame = cap.read()
        if not ok:
            cap.release()
            messagebox.showerror("Caméra", "Pas de caméra trouvée (vérifie cam_index dans config).")
            return

        pts = []
        win = "Calibration : clique haut-gauche, haut-droit, bas-droit, bas-gauche"

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
                pts.append((x, y))

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, on_click)
        print("Clique les 4 coins puis appuie sur 's' pour valider, 'r' pour reset, 'q' pour annuler.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            disp = frame.copy()
            for i, p in enumerate(pts):
                cv2.circle(disp, p, 8, (0, 255, 255), 2)
                cv2.putText(disp, str(i + 1), (p[0] + 10, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            if len(pts) == 4:
                cv2.polylines(disp, [np.array(pts, np.int32)], True, (0, 255, 0), 2)
            cv2.putText(disp, f"Coins cliques : {len(pts)}/4 — 's' valider, 'r' reset, 'q' annuler",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(win, disp)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('s') and len(pts) == 4:
                src = np.array(pts, dtype=np.float32)
                dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
                self.cfg["homography"] = cv2.getPerspectiveTransform(src, dst).tolist()
                self.calib_status.set("Homographie : calibrée ✔")
                break
            elif key == ord('r'):
                pts.clear()
            elif key in (ord('q'), 27):
                break
        cap.release()
        cv2.destroyAllWindows()

    def _try_show_existing(self):
        pass

    # ---------------- save ----------------
    def save_and_close(self):
        self.cfg["cols"] = int(self.cols_var.get())
        self.cfg["rows"] = int(self.rows_var.get())
        self.cfg["words"] = [list(w) for w in self.words]
        save_config(self.cfg)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    CalibrationApp(root)
    root.mainloop()
