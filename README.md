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
| 3. Tracking des 2 robots | Seuillage Otsu + contours + filtre de circularité → blobs ronds uniquement | fiable si robots sombres sur grille claire |
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

```bash
python calibrate.py
```

Interface Tkinter où tu règles SANS toucher au code :
- **Colonnes / lignes** de la grille (compteur, ex: 8×8 → A..H / 1..8)
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
| `dark_on_light` | `true` = robots sombres sur grille claire |
| `min_area` / `max_area` | aire d'un robot en pixels (à ajuster selon ta caméra) |
| `min_circularity` | 1.0 = cercle parfait ; 0.6 est un bon défaut |
| `collision_dist_cells` | distance de collision en unités de cases (0.5 ≈ cases voisines) |
| `collision_cooldown_s` | anti-spam : délai minimum entre 2 explosions (secondes) |
| `cam_index` | 0 = webcam par défaut |

## Limites connues & astuces

- **Éclairage constant** : Otsu est un seuillage automatique par frame ; un changeur
  de luminosité brutal peut fausser 1-2 frames (le cooldown limite l'impact).
- **Ombres portées** : si les ombres des robots sont dures, baisse `min_circularity`
  ou augmente l'éclairage diffus.
- **Alternative plus robuste** : colle un marqueur **ArUco** sur chaque Roomba
  (`cv2.aruco.detectMarkers`) → identités A/B garanties, zéro faux positif. Le
  suivi par forme est le mode "zéro préparation" ; ArUco est le mode "zéro doute".
- Pour de meilleures perfs, la détection tourne sur du grayscale full-frame (~30 fps en 1080p sur un laptop moyen).

## Structure

```
roomba_tracker.py   # tracking + collisions + explosion + panneau + enregistrement
calibrate.py        # outil de calibration (grille, mots, coins) — fenêtre séparée
config.py           # config partagée, sauvegardée dans grid_config.json
```
