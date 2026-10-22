# 🤖💥 Ouija des Roumbas

Caméra fixe au-dessus d'une grille. Deux Roombas se baladent, se cognent, et à chaque
collision on repère la case (ex: `B3`), on affiche une explosion, on l'enregistre
(vidéo + CSV), et un panneau en haut à droite affiche la liste des cases/mots touchés.

## C'est faisable ? Oui, et c'est simple

Le pipeline complet tient en 4 briques éprouvées d'OpenCV :

| Étape | Technique | Fiabilité |
|---|---|---|
| 1. Grille virtuelle | Calibration : nb de colonnes (lettres) / lignes (chiffres) | trivial |
| 2. Angle de caméra | **Homographie** (4 coins → `cv2.getPerspectiveTransform`) | très fiable |
| 3. Tracking des 2 robots | **ArUco** (marqueurs imprimés sur les Roombas) → identités A/B garanties, zéro faux positif | très fiable |
| 4. Collision | Distance entre les 2 centres < seuil (en unités de cases) + cooldown anti-spam | fiable |
| 5. Explosion | Particules dessinées sur la frame | cosmétique |

**Pourquoi ce n'est PAS fragile :** on ne track pas "deux objets parmi du bruit",
on track *exactement deux disques sombres et ronds* — tout le reste (ombres, grille,
mains) est rejeté par les filtres d'aire + circularité. Et comme la caméra est fixe,
l'homographie ne se recalcule jamais.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Calibration (fenêtre dédiée, pour "dumb user")

### 0. Imprimer les marqueurs ArUco (une seule fois)

```bash
python make_markers.py
```

Génère `markers/marker_A_id0.png` et `markers/marker_B_id1.png` — imprime-les à la
taille `marker_side_cm` de la config (8 cm par défaut) et colle-les bien à plat
sur chaque Roomba. Roumba A = id 0, Roumba B = id 1.

### 1. Calibration (fenêtre dédiée, pour "dumb user")

```bash
python calibrate.py
```

Interface Tkinter où tu règles SANS toucher au code :
- **Colonnes / lignes** de la grille (compteur, ex: 8×8 → A..H / 1..8)
- **Taille de l'explosion 💥** : slider (x0.2 → x5.0) avec bouton d'aperçu animé — l'explosion pixel-art se joue au point exact de collision
- **Les mots placés dans la grille** : mot + case de départ (ex: `B3`) + direction
  (horizontal/vertical) → le tracker affiche automatiquement le mot quand une
  collision tombe sur une case couverte par un mot
- **Les 4 coins de la grille** : bouton qui ouvre la caméra, tu cliques les 4 coins
  (haut-gauche → haut-droit → bas-droit → bas-gauche), `s` pour valider

Tout est sauvé dans `grid_config.json` (rechargé automatiquement par le tracker).

### 2. Tracking

```bash
python roomba_tracker.py
```

- 2 fenêtres : la caméra avec overlay + la grille redressée (`Grille`)
- Panneau **en haut à droite (1/8e de l'écran)** : liste des cases/mots de collision,
  visible aussi dans l'enregistrement vidéo
- Touche `r` : remet la liste des mots à zéro
- Touche `q` / `ESC` : quitte
- Sorties : `captures/session_*.mp4` (vidéo avec explosions + panneau) et
  `collisions.csv` (timestamp, cellule, coordonnées, mot)

## Réglages fins

Édite `grid_config.json` (généré par `calibrate.py`) :

| Clé | Effet |
|---|---|
| `aruco_dict` / `robot_a_id` / `robot_b_id` | type de marqueur et ids des 2 Roombas |
| `collision_dist_cells` | distance de collision en unités de cases (0.5 ≈ cases voisines) |
| `explosion_scale` | taille de l'animation d'explosion (x0.2 → x5.0, réglable dans calibrate.py) |
| `collision_cooldown_s` | anti-spam : délai minimum entre 2 explosions (secondes) |
| `cam_index` | 0 = webcam par défaut |

## Limites connues & astuces

- **Marqueurs bien à plat** : un marqueur gondolé ou très incliné se détecte moins bien. 8 cm de côté est un bon compromis vue de haut.
- **Si un marqueur est caché** (Roomba sous un obstacle), la collision n'est évaluée que quand les DEUX marqueurs sont visibles — pas de faux positif.
- Pour de meilleures perfs, la détection tourne sur du grayscale full-frame (~30 fps en 1080p sur un laptop moyen).

## Structure

```
roomba_tracker.py   # tracking ArUco + collisions + explosion sprite + panneau + enregistrement
calibrate.py        # outil de calibration (grille, mots, coins, taille explosion) — fenêtre séparée
make_markers.py     # génère les 2 marqueurs ArUco à imprimer
aruco_detect.py     # détection ArUco des 2 Roombas (uniquement ids A et B)
assets/             # sprite-sheet de l'animation d'explosion (pixel-art)
config.py           # config partagée, sauvegardée dans grid_config.json
```
