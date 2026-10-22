"""
ROOOMBA TRACKER 🤖💥 — Ouija des Roumbas
Caméra fixe au-dessus d'une grille -> track 2 robots par leur forme
(disques sombres : seuillage + circularité), détecte les collisions,
affiche une explosion, enregistre la vidéo + un log CSV des impacts.
Un panneau en haut à droite affiche les cases/mots où il y a eu collision,
avec un bouton "Reset".

Install:
    pip install -r requirements.txt

Usage:
    python calibrate.py       # 1) régler grille + mots + coins (fenêtre dédiée)
    python roomba_tracker.py  # 2) tracking
"""

import csv
import time
import datetime
from pathlib import Path

import cv2
import numpy as np

from config import load_config, cell_name, LETTERS

CFG = load_config()

ROBOT_A = {"name": "Roumba A", "draw": (60, 60, 230)}
ROBOT_B = {"name": "Roumba B", "draw": (230, 120, 40)}
ROBOTS = [ROBOT_A, ROBOT_B]

# ---------- panneau latéral (1/8e d'écran à droite) ----------
PANEL_FRACTION = 1 / 8          # largeur du panneau = 1/8 de la frame
PANEL_BG = (25, 25, 25)
PANEL_TEXT = (255, 255, 255)
MAX_WORDS_SHOWN = 40


class WordPanel:
    """Panneau dessiné SUR la frame (donc inclus dans l'enregistrement vidéo)."""

    def __init__(self, frame_w, frame_h):
        self.w = max(200, int(frame_w * PANEL_FRACTION))
        self.h = frame_h
        self.words = []      # [(cell, timestamp_str)]
        self.dirty = True

    def add(self, cell, ts):
        self.words.append((cell, ts))
        self.words = self.words[-MAX_WORDS_SHOWN:]
        self.dirty = True

    def reset(self):
        self.words.clear()
        self.dirty = True

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
        # bouton reset (zone cliquable, gérée dans main)
        self.reset_btn = (img.shape[1] - self.w + 10, self.h - 40,
                          img.shape[1] - 10, self.h - 10)
        cv2.rectangle(img, (self.reset_btn[0], self.reset_btn[1]),
                      (self.reset_btn[2], self.reset_btn[3]), (80, 80, 80), -1)
        cv2.putText(img, "RESET (touche R)", (self.reset_btn[0] + 8, self.reset_btn[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, PANEL_TEXT, 1)
        return img


# ============================================================
# TRACKING PAR LA FORME : seuil -> contours -> blobs ronds
# ============================================================
def find_robots_by_shape(gray):
    if CFG["dark_on_light"]:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (CFG["min_area"] < area < CFG["max_area"]):
            continue
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        circularity = 4 * np.pi * area / (perim * perim)
        if circularity < CFG["min_circularity"]:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        found.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), area))
    found.sort(key=lambda f: f[2], reverse=True)
    return found[:2]


def assign_identities(detections, prev):
    assigned = {}
    remaining = list(detections)
    for name, p in prev.items():
        if not remaining:
            break
        i = min(range(len(remaining)),
                key=lambda k: np.hypot(remaining[k][0] - p[0], remaining[k][1] - p[1]))
        assigned[name] = (remaining[i][0], remaining[i][1])
        remaining.pop(i)
    for robot in ROBOTS:
        if robot["name"] not in assigned and remaining:
            x, y, _ = remaining.pop(0)
            assigned[robot["name"]] = (x, y)
    return assigned


# ============================================================
# EXPLOSION : particules
# ============================================================
class Explosion:
    def __init__(self, pos_px):
        self.t0 = time.time()
        self.pos = pos_px
        rng = np.random.default_rng()
        angles = rng.uniform(0, 2 * np.pi, CFG["n_particles"])
        speeds = rng.uniform(40, 160, CFG["n_particles"])
        colors = rng.uniform(0, 255, (CFG["n_particles"], 3)).tolist()
        self.particles = list(zip(angles.tolist(), speeds.tolist(), colors))
        self.duration = CFG["explosion_duration_s"]

    def alive(self):
        return time.time() - self.t0 < self.duration

    def draw(self, img):
        k = (time.time() - self.t0) / self.duration
        for ang, spd, col in self.particles:
            r = spd * k
            x = int(self.pos[0] + r * np.cos(ang))
            y = int(self.pos[1] + r * np.sin(ang))
            size = max(1, int(8 * (1 - k)))
            cv2.circle(img, (x, y), size, tuple(int(c) for c in col), -1)
        cv2.circle(img, self.pos, int(30 * (1 - k)) + 4, (0, 0, 255), 2)


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
    """Renvoie le mot qui couvre la case `cell`, ou None."""
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
        log.writerow(["timestamp", "cellule", "col", "row", "mot"])

    explosions = []
    last_collision_t = -999.0
    collision_count = 0
    prev_positions = {}

    print("Tracking lancé. 'r' = reset liste des mots, 'q' = quitter")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        view = frame.copy()

        detections = find_robots_by_shape(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        positions = assign_identities(detections, prev_positions)
        prev_positions.update(positions)
        for robot in ROBOTS:
            if robot["name"] in positions:
                px = positions[robot["name"]]
                cv2.circle(view, px, 12, robot["draw"], 3)
                cv2.putText(view, robot["name"], (px[0] + 14, px[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, robot["draw"], 2)

        if len(positions) == 2:
            gA = mapper.to_grid(positions["Roumba A"])
            gB = mapper.to_grid(positions["Roumba B"])

            warped = mapper.warped(frame)
            draw_grid_overlay(warped)
            cv2.imshow("Grille", warped)

            dist = np.hypot(gA[0] - gB[0], gA[1] - gB[1])
            now = time.time()
            if dist < CFG["collision_dist_cells"] and now - last_collision_t > CFG["collision_cooldown_s"]:
                last_collision_t = now
                collision_count += 1
                mid = ((gA[0] + gB[0]) / 2, (gA[1] + gB[1]) / 2)
                cell = cell_name(mid[0], mid[1], CFG["cols"], CFG["rows"])
                word = word_for_cell(cell)
                mid_px = (int((positions["Roumba A"][0] + positions["Roumba B"][0]) / 2),
                          int((positions["Roumba A"][1] + positions["Roumba B"][1]) / 2))
                explosions.append(Explosion(mid_px))
                ts = datetime.datetime.now().isoformat(timespec="milliseconds")
                log.writerow([ts, cell, round(mid[0], 2), round(mid[1], 2), word or ""])
                events_file.flush()
                label = f"case {cell}" + (f" (mot: {word})" if word else "")
                panel.add(cell + (f"  {word}" if word else ""), ts[11:19])
                print(f"💥 BOOM n°{collision_count} — {label}")

            for i, robot in enumerate(ROBOTS):
                if robot["name"] in positions:
                    col, row = mapper.to_grid(positions[robot["name"]])
                    cv2.putText(view, f"{robot['name']} -> {cell_name(col, row, CFG['cols'], CFG['rows'])}",
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
