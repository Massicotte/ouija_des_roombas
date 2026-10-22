"""
ROOOMBA TRACKER 🤖💥 — Ouija des Roumbas (ArUco)
Caméra fixe au-dessus d'une grille -> track 2 Roombas par leurs marqueurs ArUco,
détecte les collisions, joue l'animation d'explosion (sprites pixel-art) EXACTEMENT
au point de collision, enregistre la vidéo + un CSV des impacts.
Panneau en haut à droite : liste des cases/mots touchés (reset avec 'r').

Install:
    pip install -r requirements.txt

Usage:
    python make_markers.py     # 1) imprimer les 2 marqueurs ArUco
    python calibrate.py       # 2) grille + mots + coins + taille d'explosion
    python roomba_tracker.py  # 3) tracking
"""

import csv
import time
import datetime
from pathlib import Path

import cv2
import numpy as np

from config import load_config, cell_name, LETTERS
import aruco_detect

CFG = load_config()

ROBOT_A = {"name": "Roumba A", "draw": (60, 60, 230)}
ROBOT_B = {"name": "Roumba B", "draw": (230, 120, 40)}
ROBOTS = [ROBOT_A, ROBOT_B]

PANEL_FRACTION = 1 / 8
PANEL_BG = (25, 25, 25)
PANEL_TEXT = (255, 255, 255)
MAX_WORDS_SHOWN = 40


# ============================================================
# ANIMATION D'EXPLOSION : sprites (pixel-art) au point de collision
# ============================================================
class ExplosionAnimation:
    """Joue les frames du sprite-sheet, centrées sur le point de collision.
    `scale` (réglé dans calibrate.py) contrôle la taille de l'explosion."""

    def __init__(self, sprites, pos, scale, fps):
        self.frames = sprites        # liste de frames RGBA (np.uint8)
        self.pos = (int(pos[0]), int(pos[1]))
        self.scale = scale
        self.t0 = time.time()
        self.frame_dur = 1.0 / fps

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
        # le sprite fait ~137px de base ; scale=1.0 -> ~un tiers de case visible
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


def load_sprites(path):
    """Charge le sprite-sheet et découpe en frames BGRA."""
    sheet = cv2.imread(str(Path(__file__).parent / path), cv2.IMREAD_UNCHANGED)
    if sheet is None:
        raise SystemExit(f"Sprite sheet introuvable : {path}")
    if sheet.ndim == 2:
        sheet = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGRA)
    elif sheet.shape[2] == 3:
        sheet = cv2.cvtColor(sheet, cv2.COLOR_BGR2BGRA)
    h = sheet.shape[0]
    n = sheet.shape[1] // h  # frames carrées côte à côte
    frames = [sheet[:, i * h:(i + 1) * h] for i in range(n)]
    return frames


# ============================================================
# PANNEAU LATÉRAL
# ============================================================
class WordPanel:
    def __init__(self, frame_w, frame_h):
        self.w = max(200, int(frame_w * PANEL_FRACTION))
        self.h = frame_h
        self.words = []
        self.reset_btn = None

    def add(self, cell, ts):
        self.words.append((cell, ts))
        self.words = self.words[-MAX_WORDS_SHOWN:]

    def reset(self):
        self.words.clear()

    def draw(self, img):
        cv2.rectangle(img, (img.shape[1] - self.w, 0), (img.shape[1], self.h), PANEL_BG, -1)
        x = img.shape[1] - self.w + 10
        y = 28
        cv2.putText(img, "COLLISIONS", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        y += 14
        cv2.line(img, (x, y), (img.shape[1] - 10, y), (90, 90, 90), 1)
        y += 24
        for cell, ts in reversed(self.words):
            cv2.putText(img, cell, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 120), 2)
            y += 12
            cv2.putText(img, ts, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1)
            y += 22
            if y > self.h - 60:
                break
        self.reset_btn = (img.shape[1] - self.w + 10, self.h - 40,
                          img.shape[1] - 10, self.h - 10)
        cv2.rectangle(img, (self.reset_btn[0], self.reset_btn[1]),
                      (self.reset_btn[2], self.reset_btn[3]), (80, 80, 80), -1)
        cv2.putText(img, "RESET (touche R)", (self.reset_btn[0] + 8, self.reset_btn[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, PANEL_TEXT, 1)
        return img


# ============================================================
# HOMOGRAPHIE
# ============================================================
class GridMapper:
    def __init__(self, H):
        self.H = np.array(H, dtype=np.float32)

    def to_grid(self, pt):
        p = np.array([[pt]], dtype=np.float32)
        g = cv2.perspectiveTransform(p, self.H)[0][0]
        return g[0] * CFG["cols"], g[1] * CFG["rows"]

    def warped(self, frame):
        w, h = CFG["warp_size"]
        S = np.array([[w, 0, 0], [0, h, 0], [0, 0, 1]], dtype=np.float32)
        return cv2.warpPerspective(frame, S @ self.H, (w, h))


def draw_grid_overlay(img):
    w, h = img.shape[1], img.shape[0]
    for i in range(CFG["cols"] + 1):
        x = int(i * w / CFG["cols"])
        cv2.line(img, (x, 0), (x, h), (255, 255, 255), 1)
        if i < CFG["cols"]:
            cv2.putText(img, LETTERS[i], (x + 4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for j in range(CFG["rows"] + 1):
        y = int(j * h / CFG["rows"])
        cv2.line(img, (0, y), (w, y), (255, 255, 255), 1)
        if j < CFG["rows"]:
            cv2.putText(img, str(j + 1), (3, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def word_for_cell(cell):
    import re
    m = re.match(r"^([A-Z])(\d+)$", cell)
    if not m:
        return None
    col = ord(m.group(1)) - ord("A")
    row = int(m.group(2)) - 1
    for word, start, direction in CFG["words"]:
        sm = re.match(r"^([A-Z])(\d+)$", str(start).upper())
        if not sm:
            continue
        sc, sr = ord(sm.group(1)) - ord("A"), int(sm.group(2)) - 1
        wlen = len(str(word).replace(" ", ""))
        if direction == "H":
            if row == sr and sc <= col < sc + wlen:
                return str(word).upper()
        else:
            if col == sc and sr <= row < sr + wlen:
                return str(word).upper()
    return None


# ============================================================
# MAIN
# ============================================================
def main():
    if not CFG.get("homography"):
        raise SystemExit("Grille non calibrée ! Lance d'abord : python calibrate.py")

    cap = cv2.VideoCapture(CFG["cam_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG["cam_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG["cam_height"])
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Pas de caméra trouvée, vérifie cam_index dans la config !")

    mapper = GridMapper(CFG["homography"])
    panel = WordPanel(frame.shape[1], frame.shape[0])
    sprites = load_sprites(CFG["explosion_sprites"])

    win = "RoombaTracker"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    writer = None
    if CFG["record"]:
        Path(CFG["record_dir"]).mkdir(exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(f"{CFG['record_dir']}/session_{stamp}.mp4",
                                 cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (vw, vh))
        print(f"Enregistrement vidéo : {CFG['record_dir']}/session_{stamp}.mp4")

    events_path = Path(CFG["events_csv"])
    log_exists = events_path.exists()
    events_file = open(events_path, "a", newline="")
    log = csv.writer(events_file)
    if not log_exists:
        log.writerow(["timestamp", "robot_a", "robot_b", "cellule", "col", "row", "mot"])

    explosions = []
    last_collision_t = -999.0
    collision_count = 0
    missing_frames = 0

    print("Tracking lancé. 'r' = reset liste des mots, 'q' = quitter")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        view = frame.copy()

        # --- détection ArUco (uniquement les 2 marqueurs A et B) ---
        detections = aruco_detect.detect(frame)
        posA = detections.get(aruco_detect.ROBOT_A_ID)
        posB = detections.get(aruco_detect.ROBOT_B_ID)
        if posA:
            cv2.circle(view, (int(posA[0]), int(posA[1])), 12, ROBOT_A["draw"], 3)
            cv2.putText(view, ROBOT_A["name"], (int(posA[0]) + 14, int(posA[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, ROBOT_A["draw"], 2)
        if posB:
            cv2.circle(view, (int(posB[0]), int(posB[1])), 12, ROBOT_B["draw"], 3)
            cv2.putText(view, ROBOT_B["name"], (int(posB[0]) + 14, int(posB[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, ROBOT_B["draw"], 2)
        if not (posA and posB):
            missing_frames += 1
            if missing_frames == 30:
                print("⚠ Les 2 marqueurs ArUco ne sont pas visibles ensemble.")
        else:
            missing_frames = 0

        if posA and posB:
            gA = mapper.to_grid((posA[0], posA[1]))
            gB = mapper.to_grid((posB[0], posB[1]))

            warped = mapper.warped(frame)
            draw_grid_overlay(warped)
            cv2.imshow("Grille", warped)

            dist = np.hypot(gA[0] - gB[0], gA[1] - gB[1])
            now = time.time()
            if dist < CFG["collision_dist_cells"] and now - last_collision_t > CFG["collision_cooldown_s"]:
                last_collision_t = now
                collision_count += 1
                # explosion EXACTEMENT au point médian entre les 2 marqueurs
                mid_px = ((posA[0] + posB[0]) / 2, (posA[1] + posB[1]) / 2)
                mid = ((gA[0] + gB[0]) / 2, (gA[1] + gB[1]) / 2)
                cell = cell_name(mid[0], mid[1], CFG["cols"], CFG["rows"])
                word = word_for_cell(cell)
                explosions.append(ExplosionAnimation(sprites, mid_px,
                                                    CFG["explosion_scale"],
                                                    CFG["explosion_fps"]))
                ts = datetime.datetime.now().isoformat(timespec="milliseconds")
                log.writerow([ts, "Roumba A", "Roumba B", cell,
                              round(mid[0], 2), round(mid[1], 2), word or ""])
                events_file.flush()
                panel.add(cell + (f"  {word}" if word else ""), ts[11:19])
                print(f"💥 BOOM n°{collision_count} — case {cell}" + (f" (mot: {word})" if word else ""))

            for i, (robot, g) in enumerate([(ROBOT_A, gA), (ROBOT_B, gB)]):
                cv2.putText(view, f"{robot['name']} -> {cell_name(g[0], g[1], CFG['cols'], CFG['rows'])}",
                            (10, 60 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, robot["draw"], 2)

        explosions = [e for e in explosions if e.alive()]
        for e in explosions:
            e.draw(view)

        cv2.putText(view, f"Collisions: {collision_count}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        panel.draw(view)

        if writer:
            writer.write(view)
        cv2.imshow(win, view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            panel.reset()
            print("Liste des mots remise à zéro.")

    cap.release()
    if writer:
        writer.release()
    events_file.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
